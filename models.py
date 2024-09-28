import copy
from einops import rearrange
import numpy as np
import torch
from torch import nn

import math
import networks
import tools
from torch import jit
from x_transformers import Encoder, Decoder
from typing import List, Tuple, Set
from kornia.augmentation import RandomCrop
from utils import ContrastModel, random_shift

to_np = lambda x: x.detach().cpu().numpy()


class RewardEMA:
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95]).to(device)

    def __call__(self, x, ema_vals):
        flat_x = torch.flatten(x.detach())
        x_quantile = torch.quantile(input=flat_x, q=self.range)
        # this should be in-place operation
        ema_vals[:] = self.alpha * x_quantile + (1 - self.alpha) * ema_vals
        scale = torch.clip(ema_vals[1] - ema_vals[0], min=1.0)
        offset = ema_vals[0]
        return offset.detach(), scale.detach()


class WorldModel(nn.Module):
    def __init__(self, obs_space, act_space, step, config):
        super(WorldModel, self).__init__()
        self._step = step
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        config.encoder["device"] = config.device
        config.encoder["use_mlr_loss"] = config.use_mlr_loss
        config.encoder["use_atp_loss"] = config.use_atp_loss
        self.encoder = networks.MultiEncoder(shapes, **config.encoder)
        self.embed_size = self.encoder.outdim
        self.dynamics = networks.RSSM(
            config.dyn_stoch,
            config.dyn_deter,
            config.dyn_hidden,
            config.dyn_rec_depth,
            config.dyn_discrete,
            config.act,
            config.norm,
            config.dyn_mean_act,
            config.dyn_std_act,
            config.dyn_min_std,
            config.unimix_ratio,
            config.initial,
            config.num_actions,
            self.embed_size,
            config.device,
        )

        self.use_pixel_shift = config.use_pixel_shift
        if self.use_pixel_shift:
            self.pixel_shift_prob = config.pixel_shift_prob
            print(f"pixel shift prob {self.pixel_shift_prob}")

        self.heads = nn.ModuleDict()
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter

        self._use_mlr_loss = config.use_mlr_loss
        if self._use_mlr_loss:
            print(f"obs space[image] shape: {obs_space['image'].shape}")
            self.transformation = [nn.Sequential(nn.ReplicationPad2d(4), RandomCrop((64, 64))), Intensity(scale=0.05)]
            image_size = obs_space["image"].shape[0] #if "image" in obs_space else obs_space["stoch"].shape[0]
            self.masker = networks.CubeMaskGenerator(
            input_size=image_size // config.patch_size, image_size=image_size, clip_size=config.batch_length, \
                block_size=config.batch_length // config.patch_size, mask_ratio=config.mask_ratio) 
            self.transformer = Encoder(
                dim=self.embed_size,
                heads=1,
                depth=2,
                layer_dropout=0.0,
            ).to(config.device)
            print(f"act_space action shape {act_space.shape}")
            self.action_embedding = nn.Linear(np.prod(act_space.shape), self.embed_size).to(config.device)
            self.position = PositionalEmbedding(self.embed_size)
            self.byol_loss = networks.SPRPred(input_size = self.embed_size, output_size = 256).to(config.device)

        self._use_atp_loss = config.use_atp_loss
        if self._use_atp_loss:
            self.encoder.set_tau(config.atp_tau)


        self._use_acro_loss = config.use_acro_loss
        if self._use_acro_loss:
            self.acro_K = config.acro_K
            self.bc_predictor = BehaviorCloneActionHead(
                feat_size,   
                act_space.shape,
                config.actor_layers,
                config.units,
                config.acro_K,
                config.use_bottleneck,
                config.bottleneck_params,
                config.use_count_based_exploration,
                config.act,
                config.acro_norm,
                config.actor_dist,
                unimix_ratio=config.action_unimix_ratio,
                device=config.device,
            )

        self._use_icm_loss = config.use_icm_loss
        if self._use_icm_loss:
            self.icm = ICMModel(
                feat_size,
                act_space.shape,
                config.actor_dist,
                config.device,
                config,
            )
            
        self.heads["decoder"] = networks.MultiDecoder(
            feat_size, shapes, **config.decoder
        )
        self.heads["reward"] = networks.MLP(
            feat_size,
            (255,) if config.reward_head["dist"] == "symlog_disc" else (),
            config.reward_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist=config.reward_head["dist"],
            outscale=config.reward_head["outscale"],
            device=config.device,
            name="Reward",
        )
        self.heads["cont"] = networks.MLP(
            feat_size,
            (),
            config.cont_head["layers"],
            config.units,
            config.act,
            config.norm,
            dist="binary",
            outscale=config.cont_head["outscale"],
            device=config.device,
            name="Cont",
        )
        for name in config.grad_heads:
            assert name in self.heads, name
        self._model_opt = tools.Optimizer(
            "model",
            self.parameters(),
            config.model_lr,
            config.opt_eps,
            config.grad_clip,
            config.weight_decay,
            opt=config.opt,
            use_amp=self._use_amp,
        )
        print(
            f"Optimizer model_opt has {sum(param.numel() for param in self.parameters())} variables."
        )
        # other losses are scaled by 1.0.
        self._scales = dict(
            reward=config.reward_head["loss_scale"],
            cont=config.cont_head["loss_scale"],
        )
    
    def random_pixel_shift(self, frames, shift_prob):
        # Generate a mask where each pixel is marked to be shifted or not based on probability
        mask = torch.rand(frames.shape) < shift_prob

        # Generate random values for the pixels that will be shifted
        random_values = torch.rand(frames.shape)

        # Apply random values where the mask is True (pixels to be shifted)
        shifted_frames = torch.where(mask, random_values, frames)

        return shifted_frames

    def _train(self, data):
        # action (batch_size, batch_length, act_dim)
        # image (batch_size, batch_length, h, w, ch)
        # reward (batch_size, batch_length)
        # discount (batch_size, batch_length)
        data = self.preprocess(data)

        with tools.RequiresGrad(self):
            with torch.amp.autocast("cuda", enabled=self._use_amp):
                _mets = {}
                if self._use_mlr_loss:
                    mask = self.masker().to(self._config.device)
                    mask = rearrange(mask, "l c h w -> l h w c")
                    # expand mask to B
                    B, T = data["image"].shape[:2]
                    mask = mask.expand(B, -1, -1, -1, -1)
                    images = self.apply_transformation(data["image"])
                    masked_obs = images * mask if "image" in data else data["stoch"] * mask
                    masked_data = {"image": masked_obs} if "image" in data else {"stoch": masked_obs, "deter": data["deter"]}
                    masked_latent, non_masked_latent = self.encoder.calculate_masked_non_masked_embeddings(masked_data, data)
                    position = self.position(T)
                    position = position.expand(B, T, -1)
                    masked_latent = masked_latent + position
                    action = data["action"].reshape(B, T, -1)
                    action_embed = self.action_embedding(action.clone())
                    action_embed_change = action_embed + position
                    x_full = torch.cat([masked_latent, action_embed_change], 1)
                    x_full = self.transformer(x_full)
                    masked_latent = x_full[:, :T]
                    _mets.update(self.byol_loss.calculate_byol_loss(masked_latent, non_masked_latent))
                    embed = non_masked_latent
                else:
                    embed = self.encoder(data)

                if self._use_atp_loss:
                    _mets.update(self.encoder.calculate_atc_loss(data["image"], K=4))
                
                # embed = self.encoder(data)
                post, prior = self.dynamics.observe(
                    embed, data["action"], data["is_first"]
                )

                feat = self.dynamics.get_feat(post)
                
                if self._use_acro_loss:
                    # TODO: what about prior
                    _mets.update(self.bc_predictor.calculate_loss(
                        feat, data["action"]
                    ))
                
                if self._use_icm_loss:
                    _mets.update(self.icm.calculate_loss(
                        feat[:, :-1], feat[:, 1:], data['action'][:, :-1]
                    ))
                

                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = self.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape
                preds = {}
                for name, head in self.heads.items():
                    grad_head = name in self._config.grad_heads
                    feat = self.dynamics.get_feat(post)
                    feat = feat if grad_head else feat.detach()
                    pred = head(feat)
                    if type(pred) is dict:
                        preds.update(pred)
                    else:
                        preds[name] = pred
                losses = {}
                for name, pred in preds.items():
                    loss = -pred.log_prob(data[name])
                    assert loss.shape == embed.shape[:2], (name, loss.shape)
                    losses[name] = loss
                scaled = {
                    key: value * self._scales.get(key, 1.0)
                    for key, value in losses.items()
                }
                model_loss = sum(scaled.values()) + kl_loss
                if self._use_acro_loss or self._use_icm_loss or self._use_mlr_loss or self._use_atp_loss:
                    for v in _mets.values():
                        model_loss += v
            metrics = self._model_opt(torch.mean(model_loss), self.parameters())

        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_free"] = kl_free
        metrics["dyn_scale"] = dyn_scale
        metrics["rep_scale"] = rep_scale
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl"] = to_np(torch.mean(kl_value))
        if self._use_acro_loss or self._use_icm_loss or self._use_mlr_loss or self._use_atp_loss:
                for k, v in _mets.items():
                    metrics[k] = to_np(v)
        with torch.cuda.amp.autocast(self._use_amp):
            metrics["prior_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(prior).entropy())
            )
            metrics["post_ent"] = to_np(
                torch.mean(self.dynamics.get_dist(post).entropy())
            )
            context = dict(
                embed=embed,
                feat=self.dynamics.get_feat(post),
                kl=kl_value,
                postent=self.dynamics.get_dist(post).entropy(),
            )
        # Doesn't work:(
        # if self._use_atp_loss:
        #     embed = self.encoder.forward_with_target(data)
        #     post, prior = self.dynamics.observe(
        #         embed, data["action"], data["is_first"]
        #     )
        post = {k: v.detach() for k, v in post.items()}
        return post, context, metrics

    def apply_transformation(self, images: torch.Tensor):
        img_shape = images.shape
        images = images.reshape(-1, *img_shape[-3:])
        images = rearrange(images, "B H W C -> B C H W")
        for t in self.transformation:
            images = t(images)
        
        images = rearrange(images, "B C H W -> B H W C")
        images = images.reshape(*img_shape)
        return images
    # this function is called during both rollout and training
    def preprocess(self, obs):
        obs = obs.copy()
        obs["image"] = torch.Tensor(obs["image"]) / 255.0
        if self.use_pixel_shift:
            obs["image"] = self.random_pixel_shift(obs["image"], self.pixel_shift_prob)
        if "discount" in obs:
            obs["discount"] *= self._config.discount
            # (batch_size, batch_length) -> (batch_size, batch_length, 1)
            obs["discount"] = torch.Tensor(obs["discount"]).unsqueeze(-1)
        # 'is_first' is necesarry to initialize hidden state at training
        assert "is_first" in obs
        # 'is_terminal' is necesarry to train cont_head
        assert "is_terminal" in obs
        obs["cont"] = torch.Tensor(1.0 - obs["is_terminal"]).unsqueeze(-1)
        obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
        return obs

    def video_pred(self, data):
        data = self.preprocess(data)
        embed = self.encoder(data)

        states, _ = self.dynamics.observe(
            embed[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5]
        )
        recon = self.heads["decoder"](self.dynamics.get_feat(states))["image"].mode()[
            :6
        ]
        reward_post = self.heads["reward"](self.dynamics.get_feat(states)).mode()[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.dynamics.imagine_with_action(data["action"][:6, 5:], init)
        openl = self.heads["decoder"](self.dynamics.get_feat(prior))["image"].mode()
        reward_prior = self.heads["reward"](self.dynamics.get_feat(prior)).mode()
        # observed image is given until 5 steps
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:6]
        model = model
        error = (model - truth + 1.0) / 2.0

        return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):
    def __init__(self, config, world_model):
        super(ImagBehavior, self).__init__()
        self._use_amp = True if config.precision == 16 else False
        self._config = config
        self._world_model = world_model
        if config.dyn_discrete:
            feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
        else:
            feat_size = config.dyn_stoch + config.dyn_deter
        self.actor = networks.MLP(
            feat_size,
            (config.num_actions,),
            config.actor["layers"],
            config.units,
            config.act,
            config.norm,
            config.actor["dist"],
            config.actor["std"],
            config.actor["min_std"],
            config.actor["max_std"],
            absmax=1.0,
            temp=config.actor["temp"],
            unimix_ratio=config.actor["unimix_ratio"],
            outscale=config.actor["outscale"],
            name="Actor",
        )
        self.value = networks.MLP(
            feat_size,
            (255,) if config.critic["dist"] == "symlog_disc" else (),
            config.critic["layers"],
            config.units,
            config.act,
            config.norm,
            config.critic["dist"],
            outscale=config.critic["outscale"],
            device=config.device,
            name="Value",
        )
        if config.critic["slow_target"]:
            self._slow_value = copy.deepcopy(self.value)
            self._updates = 0
        kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
        self._actor_opt = tools.Optimizer(
            "actor",
            self.actor.parameters(),
            config.actor["lr"],
            config.actor["eps"],
            config.actor["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer actor_opt has {sum(param.numel() for param in self.actor.parameters())} variables."
        )
        self._value_opt = tools.Optimizer(
            "value",
            self.value.parameters(),
            config.critic["lr"],
            config.critic["eps"],
            config.critic["grad_clip"],
            **kw,
        )
        print(
            f"Optimizer value_opt has {sum(param.numel() for param in self.value.parameters())} variables."
        )
        if self._config.reward_EMA:
            # register ema_vals to nn.Module for enabling torch.save and torch.load
            self.register_buffer("ema_vals", torch.zeros((2,)).to(self._config.device))
            self.reward_ema = RewardEMA(device=self._config.device)

    def _train(
        self,
        start,
        objective,
    ):
        self._update_slow_target()
        metrics = {}

        with tools.RequiresGrad(self.actor):
            with torch.cuda.amp.autocast(self._use_amp):
                imag_feat, imag_state, imag_action = self._imagine(
                    start, self.actor, self._config.imag_horizon
                )
                reward = objective(imag_feat, imag_state, imag_action)
                actor_ent = self.actor(imag_feat).entropy()
                state_ent = self._world_model.dynamics.get_dist(imag_state).entropy()
                # this target is not scaled by ema or sym_log.
                target, weights, base = self._compute_target(
                    imag_feat, imag_state, reward
                )
                actor_loss, mets = self._compute_actor_loss(
                    imag_feat,
                    imag_action,
                    target,
                    weights,
                    base,
                )
                actor_loss -= self._config.actor["entropy"] * actor_ent[:-1, ..., None]
                actor_loss = torch.mean(actor_loss)
                metrics.update(mets)
                value_input = imag_feat

        with tools.RequiresGrad(self.value):
            with torch.cuda.amp.autocast(self._use_amp):
                value = self.value(value_input[:-1].detach())
                target = torch.stack(target, dim=1)
                # (time, batch, 1), (time, batch, 1) -> (time, batch)
                value_loss = -value.log_prob(target.detach())
                slow_target = self._slow_value(value_input[:-1].detach())
                if self._config.critic["slow_target"]:
                    value_loss -= value.log_prob(slow_target.mode().detach())
                # (time, batch, 1), (time, batch, 1) -> (1,)
                value_loss = torch.mean(weights[:-1] * value_loss[:, :, None])

        metrics.update(tools.tensorstats(value.mode(), "value"))
        metrics.update(tools.tensorstats(target, "target"))
        metrics.update(tools.tensorstats(reward, "imag_reward"))
        if self._config.actor["dist"] in ["onehot"]:
            metrics.update(
                tools.tensorstats(
                    torch.argmax(imag_action, dim=-1).float(), "imag_action"
                )
            )
        else:
            metrics.update(tools.tensorstats(imag_action, "imag_action"))
        metrics["actor_entropy"] = to_np(torch.mean(actor_ent))
        with tools.RequiresGrad(self):
            metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
            metrics.update(self._value_opt(value_loss, self.value.parameters()))
        return imag_feat, imag_state, imag_action, weights, metrics

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}

        def step(prev, _):
            state, _, _ = prev
            feat = dynamics.get_feat(state)
            inp = feat.detach()
            action = policy(inp).sample()
            succ = dynamics.img_step(state, action)
            return succ, feat, action

        succ, feats, actions = tools.static_scan(
            step, [torch.arange(horizon)], (start, None, None)
        )
        states = {k: torch.cat([start[k][None], v[:-1]], 0) for k, v in succ.items()}

        return feats, states, actions

    def _compute_target(self, imag_feat, imag_state, reward):
        if "cont" in self._world_model.heads:
            inp = self._world_model.dynamics.get_feat(imag_state)
            discount = self._config.discount * self._world_model.heads["cont"](inp).mean
        else:
            discount = self._config.discount * torch.ones_like(reward)
        value = self.value(imag_feat).mode()
        target = tools.lambda_return(
            reward[1:],
            value[:-1],
            discount[1:],
            bootstrap=value[-1],
            lambda_=self._config.discount_lambda,
            axis=0,
        )
        weights = torch.cumprod(
            torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0
        ).detach()
        return target, weights, value[:-1]

    def _compute_actor_loss(
        self,
        imag_feat,
        imag_action,
        target,
        weights,
        base,
    ):
        metrics = {}
        inp = imag_feat.detach()
        policy = self.actor(inp)
        # Q-val for actor is not transformed using symlog
        target = torch.stack(target, dim=1)
        if self._config.reward_EMA:
            offset, scale = self.reward_ema(target, self.ema_vals)
            normed_target = (target - offset) / scale
            normed_base = (base - offset) / scale
            adv = normed_target - normed_base
            metrics.update(tools.tensorstats(normed_target, "normed_target"))
            metrics["EMA_005"] = to_np(self.ema_vals[0])
            metrics["EMA_095"] = to_np(self.ema_vals[1])

        if self._config.imag_gradient == "dynamics":
            actor_target = adv
        elif self._config.imag_gradient == "reinforce":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
        elif self._config.imag_gradient == "both":
            actor_target = (
                policy.log_prob(imag_action)[:-1][:, :, None]
                * (target - self.value(imag_feat[:-1]).mode()).detach()
            )
            mix = self._config.imag_gradient_mix
            actor_target = mix * target + (1 - mix) * actor_target
            metrics["imag_gradient_mix"] = mix
        else:
            raise NotImplementedError(self._config.imag_gradient)
        actor_loss = -weights[:-1] * actor_target
        return actor_loss, metrics

    def _update_slow_target(self):
        if self._config.critic["slow_target"]:
            if self._updates % self._config.critic["slow_target_update"] == 0:
                mix = self._config.critic["slow_target_fraction"]
                for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

class ICMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: tuple,
        dist: str,
        device: str,
        config,
        unimix_ratio: float=0.01,
    ):
        super(ICMModel, self).__init__()

        self.inp_dim = input_dim  # s_t, s_{t + k}, embedding_k
        self._size = np.prod(action_shape)
        self._action_shape = action_shape
        self.device = device
        self._config = config
        self._dist = dist
        self._unimix_ratio = unimix_ratio

        self.inverse_net = nn.Sequential(
            nn.Linear(self.inp_dim * 2, 512), nn.ReLU(), nn.Linear(512, self._size)
        )

        self.residual = [
            nn.Sequential(
                nn.Linear(self._size + self.inp_dim, 256),
                nn.LeakyReLU(),
                nn.Linear(256, self.inp_dim),
            ).to(self.device)
        ] * 8

        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.inp_dim + self._size, self.inp_dim), nn.LeakyReLU()
        )
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.inp_dim + self._size, self.inp_dim),
        )

        self.eta = config.eta

        for p in self.modules():
            if isinstance(p, nn.Linear):
                torch.nn.init.kaiming_uniform_(p.weight, a=1.0)
                p.bias.data.zero_()
        
        print(f"Total number of parameters for ICM in Megabytes is: {sum(p.numel() for p in self.parameters()) * 4 / 1024 / 1024}")


    def forward(self, inputs: list) -> tuple:
        state, next_state, action = inputs

        encode_state = state
        encode_next_state = next_state
        # get pred action
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action)
        if self._dist == "onehot":
            pred_action = tools.OneHotDist(pred_action, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_categorical":
            pred_action = pred_action.reshape(list(pred_action.shape[:-1]) + list(self._action_shape))
            pred_action = tools.FlattenDist(
                tools.OneHotDist(pred_action, unimix_ratio=self._unimix_ratio)
            )
        # ---------------------

        # get pred next state
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig)

        # residual
        for i in range(4):
            pred_next_state_feature = self.residual[i * 2](
                torch.cat((pred_next_state_feature_orig, action), 1)
            )
            pred_next_state_feature_orig = (
                self.residual[i * 2 + 1](
                    torch.cat((pred_next_state_feature, action), 1)
                )
                + pred_next_state_feature_orig
            )

        pred_next_state_feature = self.forward_net_2(
            torch.cat((pred_next_state_feature_orig, action), 1)
        )

        real_next_state_feature = encode_next_state
        return real_next_state_feature, pred_next_state_feature, pred_action

    def compute_intrinsic_reward(
        self,
        state: torch.FloatTensor,
        next_state: torch.FloatTensor,
        action: torch.FloatTensor,
    ):
        real_next_state_feature, pred_next_state_feature, pred_action = self(
            [state, next_state, action]
        )
        intrinsic_reward = self.eta * F.mse_loss(
            real_next_state_feature, pred_next_state_feature, reduction="none"
        ).mean(-1)
        return intrinsic_reward

    def calculate_loss(
        self,
        s_batch: torch.FloatTensor,
        next_s_batch: torch.FloatTensor,
        action_batch: torch.FloatTensor,
    ) -> dict:
        _s_batch = s_batch.reshape(-1, s_batch.shape[-1])
        _next_s_batch = next_s_batch.reshape(-1, next_s_batch.shape[-1])
        _action_batch = action_batch.reshape(-1, action_batch.shape[-1])

        # ce = nn.MSELoss() if self._config.action_encode else nn.CrossEntropyLoss()
        forward_mse = nn.MSELoss()

        real_next_state_feature, pred_next_state_feature, pred_action_dist = self(
            [_s_batch, _next_s_batch, _action_batch]
        )

        inverse_loss = -pred_action_dist.log_prob(_action_batch).mean()
        # inverse_loss = ce(pred_action, _action_batch)

        forward_loss = forward_mse(
            pred_next_state_feature, real_next_state_feature.detach()
        )

        metrics = {"forward_loss_icm": forward_loss, "inverse_loss_icm": inverse_loss}
        # print(f"forward_loss: {forward_loss}, inverse_loss: {inverse_loss}")
        return metrics


class BehaviorCloneActionHead(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        action_shape: tuple,
        layers,
        units,
        K: int = 16,
        use_bottleneck: bool = False,
        bottleneck_params: dict = None,
        use_count_based_exploration: bool = False,
        act=nn.ELU,
        norm=nn.LayerNorm,
        dist="onehot",
        outscale=1.0,
        unimix_ratio=0.01,
        device="cuda",
    ):
        super(BehaviorCloneActionHead, self).__init__()
        self.inp_dim = inp_dim  # s_t, s_{t + k}, embedding_k
        self._size = np.prod(action_shape)
        self._action_shape = action_shape
        self._layers = layers
        self._units = units
        self._dist = dist
        act = getattr(torch.nn, act)
        norm = getattr(torch.nn, norm)
        self._unimix_ratio = unimix_ratio
        self._botlleneck_params = bottleneck_params
        self._use_bottleneck = use_bottleneck

        self.K = K
        self.time_embedding = nn.Embedding(K, inp_dim).to(device)
        self.time_embedding.apply(tools.weight_init)

        pre_layers = []

        start_dim = 3 * inp_dim
        for index in range(self._layers):
            pre_layers.append(nn.Linear(start_dim, self._units, bias=False))
            pre_layers.append(norm(self._units, eps=1e-03))
            pre_layers.append(act())
            if index == 0:
                start_dim = self._units
        self._pre_layers = nn.Sequential(*pre_layers)
        self._pre_layers.apply(tools.weight_init)

        self._dist_layer = nn.Linear(self._units, self._size)
        self._dist_layer.apply(tools.uniform_weight_init(outscale))

        if self._use_bottleneck:
            # consts
            self._tau = bottleneck_params["tau"]
            self._topk = bottleneck_params["topk"]
            self._num_protos = bottleneck_params["num_protos"]
            self._queue_size = bottleneck_params["queue_size"]

            # models
            self._projector = Projector(inp_dim, bottleneck_params["proj_dim"]).to(
                device
            )
            self._projector.apply(tools.weight_init)

            self._protos = nn.Linear(inp_dim, self._num_protos, bias=False).to(device)
            self._protos.apply(tools.weight_init)

            if use_count_based_exploration:
                self._queue = torch.zeros(
                    self._queue_size, inp_dim, device="cpu"
                )  # too big, can't fit in GPU
                self._queue_ptr = 0

        print(f"Total number of parameters for ACRO in Megabytes is: {sum(p.numel() for p in self.parameters()) * 4 / 1024 / 1024}")


    def forward(self, features, dtype=None):
        x_t = features[:, : -self.K, :]

        # pick random index {0, ..., K - 1} for B batches
        idx = torch.randint(
            0, self.K, (features.shape[0], features.shape[1] - self.K)
        ).to(features.device)
        # get proper index from idx
        indices = torch.arange(features.shape[1] - self.K).to(features.device) + idx
        # pick x_{t+K} batch-wise with the indices
        x_tplusk = features[torch.arange(features.shape[0]).unsqueeze(1), indices, :]
        # get embedding for idx
        time_embed = self.time_embedding(idx)

        # concat x_t, x_{t+K}, embedding
        x = torch.cat([x_t, x_tplusk, time_embed], dim=-1)
        x = self._pre_layers(x)
        x = self._dist_layer(x)

        if self._dist == "onehot":
            dist = tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
        elif self._dist == "onehot_categorical":
            x = x.reshape(list(x.shape[:-1]) + list(self._action_shape))
            dist = tools.FlattenDist(
                tools.OneHotDist(x, unimix_ratio=self._unimix_ratio)
            )
        return dist

    def calculate_loss(
        self, features: torch.FloatTensor, actions: torch.LongTensor
    ) -> dict:
        dist = self.forward(features)
        acstate_loss = -dist.log_prob(actions[:, : -self.K]).mean()

        if self._use_bottleneck:
            bottleneck_loss = self._bottleneck_loss(features)

        metrics = {
            "acstate_loss": acstate_loss,
        }

        if self._use_bottleneck:
            metrics["bottleneck_loss"] = bottleneck_loss

        return metrics

    def _normalize_protos(self):
        C = self._protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self._protos.weight.data.copy_(C)

    def _bottleneck_loss(self, features: torch.FloatTensor) -> torch.FloatTensor:
        # normalize prototypes
        self._normalize_protos()
        s = self._projector(features[:, :-1])
        s = F.normalize(s, dim=-1, p=2)
        scores_s = self._protos(s)
        log_p_s = F.log_softmax(scores_s / self._tau, dim=-1)

        with torch.no_grad():
            s_prime = F.normalize(features[:, 1:])
            scores_s_prime = self._protos(s_prime)
            q_t = sinkhorn_knopp(scores_s_prime / self._tau)

        loss = -torch.sum(q_t * log_p_s) / s.shape[0]
        return loss

    def compute_intr_reward(self, obs, step):
        pass


class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(pred_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, pred_dim)
        )

        self.apply(tools.weight_init)

    def forward(self, x):
        return self.trunk(x)


@jit.script
def sinkhorn_knopp(Q):
    Q_shape = Q.shape
    Q = Q.reshape(-1, Q_shape[-1])
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
    for it in range(3):
        u = Q.sum(dim=1)
        u = r / u
        Q *= u.unsqueeze(dim=1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
    Q = Q / Q.sum(dim=0, keepdim=True)

    return Q.T.reshape(Q_shape)


class GRUCell(nn.Module):
    def __init__(self, inp_size, size, norm=False, act=torch.tanh, update_bias=-1):
        super(GRUCell, self).__init__()
        self._inp_size = inp_size
        self._size = size
        self._act = act
        self._norm = norm
        self._update_bias = update_bias
        self._layer = nn.Linear(inp_size + size, 3 * size, bias=False)
        if norm:
            self._norm = nn.LayerNorm(3 * size, eps=1e-03)

    @property
    def state_size(self):
        return self._size

    def forward(self, inputs, state):
        state = state[0]  # Keras wraps the state in a list.
        parts = self._layer(torch.cat([inputs, state], -1))
        if self._norm:
            parts = self._norm(parts)
        reset, cand, update = torch.split(parts, [self._size] * 3, -1)
        reset = torch.sigmoid(reset)
        cand = self._act(reset * cand)
        update = torch.sigmoid(update + self._update_bias)
        output = update * cand + (1 - update) * state
        return output, [output]

@jit.script
def get_target_patches(patch_h: int, patch_w: int, block_h:int, block_w: int, M: int)->torch.Tensor:
    start_patches_h = torch.randint(0, patch_h - block_h + 1, (M,))
    start_patches_w = torch.randint(0, patch_w - block_w + 1, (M,))
    start_patches = start_patches_h * patch_w + start_patches_w
    target_patches = torch.arange(block_h).repeat_interleave(block_w) * patch_w + torch.arange(block_w).repeat(block_h) + start_patches[:, None]
    return target_patches


class IJEPA(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        proj_dim: int,
        n_patches: int,
        embed_dim: int,
        enc_heads: int,
        enc_depth: int,
        decoder_depth: int,
        num_target: int,
        layer_dropout: float,
        post_emb_norm: bool,
        target_aspect_ratio: tuple,
        target_scale: tuple,
        context_aspect_ratio: tuple,
        context_scale: tuple,
        m: float,
        m_start_end: tuple,
        should_patch: bool,
        device: str,
    ):
        super().__init__()
 
        self.inp_dim = inp_dim
        self.proj_dim = proj_dim
        self.n_patches = n_patches
        self.embed_dim = embed_dim
        self.enc_heads = enc_heads
        self.enc_depth = enc_depth
        self.decoder_depth = decoder_depth
        self.num_target = num_target
        self.layer_dropout = layer_dropout
        self.post_emb_norm = post_emb_norm
        self.target_aspect_ratio = target_aspect_ratio
        self.target_scale = target_scale
        self.context_aspect_ratio = context_aspect_ratio
        self.context_scale = context_scale
        self.m = m
        self.m_start_end = m_start_end
        self._should_patch = should_patch
        


        self._device = device
        self.pos_embedding = nn.Embedding(n_patches, embed_dim).to(device)

        # define the cls and mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim)).to(device)
        nn.init.trunc_normal_(self.mask_token, 0.02)

        # define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=enc_heads,
            depth=enc_depth,
            layer_dropout=self.layer_dropout,
        ).to(device)
        self.student_encoder = copy.deepcopy(self.teacher_encoder).to(device)
        self.projector = nn.Linear(inp_dim, proj_dim).to(
            device
        )  # inp_dim -> proj_dim -> (n_patches, proj_dim / n_patches)
        self.predictor = Predictor(embed_dim, enc_heads, decoder_depth)
        # total params
        print(f"Total number of parameters for IJEPA in Megabytes is: {sum(p.numel() for p in self.parameters()) * 4 / 1024 / 1024}")

    # @jit.script
    @torch.no_grad()
    def get_target_block(
        self,
        target_encoder: Encoder,
        x: torch.Tensor,
        aspect_ratio: float,
        scale: float,
    ) -> Tuple[torch.Tensor, List[int], Set[int]]:

        # get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)
        x = self.norm(x)
        # get the number of patches
        num_patches = self.n_patches
        # get the number of patches in the target block
        num_patches_block = int(num_patches * scale)
        # get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        # get the patches in the target block
        M = self.num_target
        target_block = torch.zeros(
            (M, x.shape[0], block_h * block_w, x.shape[2]), device=self._device
        )
        patch_h = patch_w = int(torch.sqrt(torch.tensor(num_patches)))
        # for z in range(M):
        #     # get the starting patch
        #     start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
        #     start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
        #     start_patch = start_patch_h * patch_w + start_patch_w

        #     # get the patches in the target block
        #     patches = (
        #         torch.arange(block_h).repeat_interleave(block_w) * patch_w
        #         + torch.arange(block_w).repeat(block_h)
        #         + start_patch
        #     )

        #     # get the target block
        #     target_patches.append(patches)
        #     target_block[z] = x[:, patches, :]
        target_patches = get_target_patches(patch_h, patch_w, block_h, block_w, M)
        target_block = x[:, target_patches, :].permute(1, 0, 2, 3)

        all_patches = target_patches.reshape(-1).unique().tolist()

        return target_block, target_patches, all_patches
    
    
        
    # @jit.script
    def get_context_block(
        self,
        x: torch.Tensor,
        patch_dim: int,
        aspect_ratio: float,
        scale: float,
        target_patches: List[int],
    ) -> torch.Tensor:

        patch_h = patch_w = int(torch.sqrt(torch.tensor(patch_dim)))
        # get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        # get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        # get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h + 1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w + 1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        # get the patches in the context_block
        patches = (
            torch.arange(block_h).repeat_interleave(block_w) * patch_w
            + torch.arange(block_w).repeat(block_h)
            + start_patch
        ).tolist()

        patches = [patch for patch in patches if patch not in target_patches]

        return x[:, patches, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # start with B, T, E
        B, T, E = x.shape
        x = rearrange(x, "b t e -> (b t) e")
        x = self.projector(x)  # BT, proj_dim
        x = rearrange(x, "(bt) e -> (bt) p e", p=self.n_patches)
        x = x + self.pos_embedding
        x = self.post_emb_norm(x)
        return self.student_encoder(x)

    # @jit.script
    def _convert_embed_to_patches(self, embed: torch.Tensor) -> torch.Tensor:
        B, T, E = embed.shape
        embed = rearrange(embed, "b t e -> (b t) e")
        embed = self.projector(embed)
        patches = rearrange(
            embed, "bt (p e) -> bt p e", p=self.n_patches, e=self.embed_dim
        )
        return patches

    # @jit.script
    def _project_embed(self, embed: torch.Tensor) -> torch.Tensor:
        self.n_patches = embed.shape[1]
        return self.projector(embed)

    def compute_prediction_and_target(
        self,
        x: torch.Tensor,
        target_aspect_ratio: float,
        target_scale: float,
        context_aspect_ratio: int,
        context_scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the patch embeddings
        x = self._convert_embed_to_patches(x) if self._should_patch else self._project_embed(x)
        # add the positional embeddings
        x = x + self.pos_embedding.weight
        # normalize the embeddings
        x = self.post_emb_norm(x)
        # #get target embeddings
        target_blocks, target_patches, all_patches = self.get_target_block(
            self.teacher_encoder,
            x,
            target_aspect_ratio,
            target_scale,
        )
        m, b, n, e = target_blocks.shape
        # get context embedding

        context_block = self.get_context_block(
            x, self.n_patches, context_aspect_ratio, context_scale, all_patches
        )
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)

        # prediction_blocks = torch.zeros((m, b, n, e)).to(x.device)
        # get the prediction blocks, predict each target block separately
        # for i in range(m):
        #     target_masks = self.mask_token.repeat(b, n, 1)
        #     target_pos_embedding = self.pos_embedding.weight[None, target_patches[i], :]
        #     target_masks = target_masks + target_pos_embedding
        #     prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        target_masks = self.mask_token.repeat(m, b, n, 1)
        target_pos_embedding = self.pos_embedding.weight[target_patches, :].unsqueeze(1)
        target_masks = target_masks + target_pos_embedding
        prediction_blocks = self.predictor(context_encoding.repeat(m, 1, 1, 1), target_masks)

        return prediction_blocks, target_blocks

    def calculate_loss(self, embed: torch.Tensor) -> dict:
        #generate random target and context aspect ratio and scale
        target_aspect_ratio = np.random.uniform(self.target_aspect_ratio[0], self.target_aspect_ratio[1])
        target_scale = np.random.uniform(self.target_scale[0], self.target_scale[1])
        context_aspect_ratio = self.context_aspect_ratio
        context_scale = np.random.uniform(self.context_scale[0], self.context_scale[1])
        
        prediction_blocks, target_blocks = self.compute_prediction_and_target(embed, target_aspect_ratio, target_scale, context_aspect_ratio, context_scale)
        loss = nn.MSELoss()(prediction_blocks, target_blocks)
        return {"IJEPA_loss": loss} if self._should_patch else {"TJEPA_loss": loss}


    def on_after_backward(self):
        self.update_momentum(self.m)
        self.m += (self.m_start_end[1] - self.m_start_end[0]) / 1e6
    
    def update_momentum(self, m):
        student_model = self.student_encoder.eval()
        teacher_model = self.teacher_encoder.eval()
        with torch.no_grad():
            for student_param, teacher_param in zip(student_model.parameters(), teacher_model.parameters()):
                teacher_param.data.mul_(other=m).add_(other=student_param.data, alpha=1 - m)

        


class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()

        self.predictor = Decoder(dim=embed_dim, depth=depth, heads=num_heads)

    # def forward(self, context_encoding, target_masks):
    #     x = torch.cat((context_encoding, target_masks), dim=1)
    #     x = self.predictor(x)
    #     # return last len(target_masks) tokens
    #     l = x.shape[1]
    #     return x[:, l - target_masks.shape[1] :, :]

    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim=-2)
        m, b, n, e = x.shape

        x = rearrange(x, "m b n e -> (m b) n e")
        x = self.predictor(x)
        # return last len(target_masks) tokens
        l = x.shape[-2]
        return x[:, l - target_masks.shape[-2] :, :].reshape(m, b, target_masks.shape[-2], e)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, length):
        return self.pe[:, :length]

class Intensity(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))
        return x * noise