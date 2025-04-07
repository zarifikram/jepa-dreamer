from abc import abstractmethod
import copy
import torch
from utils import random_shift
from einops import rearrange
import torch.nn.functional as F
from sklearn.cluster import KMeans


perception_archs = {
    "a": [
        ("conv", 256, 128, 3, 1, 1),
        ("conv", 128, 64, 3, 1, 1),
        ("linear", 1024, 256, -1, -1, -1),
        ("linear", 256, 128, -1, -1, -1),
    ],
    "b": [
        ("linear", 256, 128, -1, -1, -1),
        ("linear", 128, 128, -1, -1, -1),
        ("linear", 128, 64, -1, -1, -1),
    ],
    "c": [
        ("linear", 1536, 512, -1, -1, -1),
        ("linear", 512, 256, -1, -1, -1),
        ("linear", 256, 256, -1, -1, -1),
        ("linear", 256, 128, -1, -1, -1),
    ],
    "d": [
        ("conv", 2, 4, 5, 1, 1),  # 36
        ("conv", 4, 8, 5, 1, 1),  # 32
        ("conv", 8, 16, 6, 2, 1),  # 15
        ("conv", 16, 32, 5, 4, 1),  #
        ("linear", 512, 256, -1, -1, -1),
        ("linear", 256, 64, -1, -1, -1),
        ("linear", 64, 2, -1, -1, -1),
    ],
    "e": [
        ("conv", 2, 4, 6, 2, 1),  # 19
        ("conv", 4, 8, 5, 2, 1),  # 9
        ("conv", 8, 16, 3, 2, 1),  # 5
        ("linear", 400, 256, -1, -1, -1),
        ("linear", 256, 64, -1, -1, -1),
        ("linear", 64, 2, -1, -1, -1),
    ],
    "f": [
        ("conv", 2, 32, 3, 1, 1),  # 40
        ("maxpool", -1, -1, 2, 2, 0),  # 20
        ("conv", 32, 32, 3, 1, 1),  # 20
        ("maxpool", -1, -1, 2, 2, 0),  # 10
        ("conv", 32, 32, 3, 1, 1),  # 10
        ("maxpool", -1, -1, 2, 2, 0),  # 5
        ("conv", 32, 32, 3, 1, 1),  # 5
        ("maxpool", -1, -1, 2, 2, 0),  # 2
        ("linear", 128, 2, -1, -1, -1),
    ],
}


def get_layers(arch, final_dim=None, act=torch.nn.ReLU):
    arch = perception_archs[arch]
    conv_layers, mlp_layers = [], []
    for i, (type_layer, in_dim, out_dim, kernel, stride, padding) in enumerate(arch):
        if type_layer == "conv":
            conv_layers.append(
                torch.nn.Conv2d(in_dim, out_dim, kernel, stride, padding)
            )  # don't change width and height
        elif type_layer == "linear":
            if i == len(arch) - 1 and final_dim:
                out_dim = final_dim
            mlp_layers.append(torch.nn.Linear(in_dim, out_dim))
        elif type_layer == "maxpool":
            conv_layers.append(torch.nn.MaxPool2d(kernel, stride, padding))

        if i < len(arch) - 1:
            mlp_layers.append(act())

    return torch.nn.ModuleDict(
        {
            "conv": torch.nn.Sequential(*conv_layers),
            "mlp": torch.nn.Sequential(*mlp_layers),
        }
    )


class ContrastiveEvaluator(torch.nn.Module):
    # uses a GAN-like discrimator which is learnt through contrastive learning
    def __init__(
        self,
        evaluator_config,
    ):
        super(ContrastiveEvaluator, self).__init__()
        self.config = evaluator_config
        self.assumed_height = evaluator_config["assumed_height"]
        self.layers = get_layers(evaluator_config["arch"], act=torch.nn.LeakyReLU)
        print(
            f"total number of parameters in the model is {sum(p.numel() for p in self.parameters())}"
        )

    def calculate_multihead_error(self, data: dict, kl_loss=None):
        base_feats, positive_feats, negative_feats = (
            self._calculate_positive_and_negative_samples(data)
        )
        positive_scores, negative_scores = self(base_feats, positive_feats), self(
            base_feats, negative_feats
        )
        # cross entropy loss
        positive_loss = torch.nn.functional.cross_entropy(
            positive_scores,
            torch.ones(positive_scores.shape[0], device=positive_scores.device).long(),
        )
        if kl_loss is not None:
            considered_loss = kl_loss[:, 1:]
            z = (considered_loss - considered_loss.mean()) / considered_loss.std()
            gt_negative_scores = (z < -1.645).long().reshape(-1)
            negative_loss = torch.nn.functional.cross_entropy(
                negative_scores,
                gt_negative_scores,
            )
        else:
            negative_loss = torch.nn.functional.cross_entropy(
                negative_scores,
                torch.zeros(negative_scores.shape[0], device=negative_scores.device).long(),
            )
        return {"evaluator_loss": positive_loss + negative_loss}

    def _calculate_positive_and_negative_samples(self, data):
        base_feats = self._get_base_samples(data)
        positive_feats = self._get_positive_samples(data)
        negative_feats = self._get_negative_samples(data)
        return base_feats.detach(), positive_feats.detach(), negative_feats.detach()

    def _get_base_samples(self, data):
        post_feats = data["embed"]
        return post_feats[:, :-1].reshape(-1, post_feats.shape[-1])

    def _get_positive_samples(self, data):
        post_feats = data["embed"]
        return post_feats[:, 1:].reshape(-1, post_feats.shape[-1])

    def _get_negative_samples(self, data):
        rand_number = torch.rand(1)
        if rand_number < 1.0:
            return data["goal_embed"]  # already reshaped
        else:
            return self._get_prior_negative_samples(data)

    def _get_prior_negative_samples(self, data):
        stoch, deter = data["prior"]["stoch"], data["prior"]["deter"]
        B, T = deter.shape[:2]
        prior_feat = torch.cat([stoch.reshape(B, T, -1), deter.reshape(B, T, -1)], -1)
        indices = self._get_indices(B, T, prior_feat.device)
        return prior_feat[
            torch.arange(B, device=indices.device).repeat_interleave(T - 1),
            indices.reshape(-1),
        ]

    def _get_indices(self, B, T, device):
        indices = torch.arange(T, device=device).repeat(T - 1, 1)
        remove_indices = torch.arange(1, T, device=device).unsqueeze(1)
        # remove the remove_indices from the indices
        indices = indices[indices != remove_indices].reshape(T - 1, -1)
        # now pick one random index from each row and do this for B times
        indices = indices[
            torch.arange(T - 1, device=device).repeat(B),
            torch.randint(0, T - 1, (B * (T - 1),), device=device),
        ].reshape(B, -1)
        return indices

    def forward(self, base_feats, other_feats):
        base_feats, other_feats = self._padding_transform(
            base_feats
        ), self._padding_transform(other_feats)
        x = torch.cat([base_feats, other_feats], 1)
        x = self.layers["conv"](x)
        x = x.view(x.size(0), -1)
        x = self.layers["mlp"](x)
        return x

    @torch.no_grad()
    def calculate_comparison_embedding(self, base_feats, other_feats):
        base_feats, other_feats = self._padding_transform(
            base_feats
        ), self._padding_transform(other_feats)
        x = torch.cat([base_feats, other_feats], 1)
        x = self.layers["conv"](x)
        x = x.view(x.size(0), -1)
        return x

    def _padding_transform(self, feats):
        B, D = feats.shape
        extra_to_add = self.assumed_height**2 - D
        if extra_to_add > 0:
            feats = torch.cat(
                [feats, torch.zeros(B, extra_to_add, device=feats.device)], -1
            )

        return feats.reshape(B, 1, self.assumed_height, self.assumed_height)

    @torch.no_grad()
    def calculate_rejection_mask_and_distance_from_generated_outputs(
        self, context_feats, generated_feats
    ):
        scores = self(context_feats, generated_feats).argmax(-1).bool()
        # flip the scores
        scores = ~scores
        return scores, None


class FeasibilityEvaluator(torch.nn.Module):
    def __init__(self, evaluator_config, encoder, action_space):
        super(FeasibilityEvaluator, self).__init__()
        self.config = evaluator_config
        self.encoder = encoder
        self.action_space = action_space
        self.rejection_tau = self.config["rejection_tau"]
        self.local_feature_extrator = LocalPerception(evaluator_config)
        self.discount_predictor = DiscountValue(
            evaluator_config["discount_config"], action_space.shape, encoder.device
        )
        self.histogram_converter = HistogramConverter(
            evaluator_config["histogram_config"]
        )
        self.is_latent_goal = evaluator_config["is_latent_goal"]
        self.goal_reached_threshold = evaluator_config["goal_reach_threshold"]
        self.histogram_converter.to(encoder.device)

    def forward(
        self,
    ):
        pass

    def calculate_multihead_error(self, data: dict):
        (
            batch_obs_curr,
            batch_action,
            batch_obs_next,
            batch_done,
            batch_obs_targ,
            batch_state_curr,
            batch_state_next,
            batch_state_targ,
        ) = self._get_relevant_info(data)

        assert self.is_latent_goal == True or batch_obs_targ is not None

        size_batch = batch_state_curr.shape[0]
        state_local_curr, state_local_next, state_local_targ = (
            self.calculate_batched_local_state(
                batch_state_curr, batch_state_next, batch_state_targ
            )
        )

        if self.goal_reached_threshold:
            if self.is_latent_goal:
                batch_targ_reached = (batch_state_next - batch_state_targ).pow(2).sum(
                    (1)
                ) < self.goal_reached_threshold
            else:
                batch_targ_reached = (batch_obs_next - batch_obs_targ).pow(2).sum(
                    (1, 2, 3)
                ) < self.goal_reached_threshold
        else:
            batch_targ_reached = (
                (batch_obs_next == batch_obs_targ).reshape(size_batch, -1).all(-1)
            )

        # log the mse between the target and the next state
        # with torch.no_grad():
        #     err = ((batch_obs_next - batch_obs_targ) **2).sum((1,2,3))
        #     print(f"for threhold of {1} the rate of success is {((err < 1).sum() / size_batch) * 100}%")
        #     print(f"for threhold of {5} the rate of success is {((err < 5).sum() / size_batch) * 100}%")
        #     print(f"for threhold of {10} the rate of success is {((err < 10).sum() / size_batch) * 100}%")
        #     print()

        predicted_discount = self.discount_predictor(
            state_local_curr, state_local_targ, batch_action
        )

        with torch.no_grad():
            action_next = self._calculate_best_action_for_next_step(
                state_local_next, state_local_targ
            )
            target_distance = self._calculate_binned_target_distance(
                state_local_curr,
                state_local_targ,
                action_next,
                batch_done,
                batch_targ_reached,
            )

        # discount_logits_curr = predicted_discount.reshape(size_batch, -1)
        loss_discount = torch.nn.functional.kl_div(
            torch.log_softmax(predicted_discount, -1),
            target_distance.detach(),
            reduction="none",
        ).sum(-1)
        return {"evaluator_loss": loss_discount.sum()}

    def _calculate_binned_target_distance(
        self, state_local_curr, state_local_targ, action, batch_done, batch_targ_reached
    ):
        weighted_distances = self.discount_predictor._calculate_weighted_distance(
            state_local_curr, state_local_targ, action
        )
        target_distance = self._get_target_distance(
            weighted_distances, batch_done, batch_targ_reached
        )
        return self._calculate_binned_distance(target_distance)

    def _get_target_distance(self, distance, batch_done, batch_targ_reached):
        distance[batch_done] = 1000.0
        distance[batch_targ_reached] = 0.0
        return distance + 1

    def _calculate_binned_distance(self, distance):
        return self.histogram_converter.to_histogram(distance)

    def _calculate_best_action_for_next_step(self, state_local_next, state_local_targ):
        return self.discount_predictor._get_best_action_by_discount(
            state_local_next, state_local_targ
        )

    def calculate_batched_local_state(
        self,
        batch_state_curr,
        batch_state_next,
        batch_state_targ,
    ):
        batched_input = torch.cat(
            [batch_state_curr, batch_state_next, batch_state_targ], 0
        )
        batched_local = self.local_feature_extrator(batched_input)
        size_batch = batch_state_curr.shape[0]
        state_local_curr, state_local_next, state_local_targ = torch.split(
            batched_local, [size_batch, size_batch, size_batch], dim=0
        )
        return state_local_curr, state_local_next, state_local_targ

    def _get_relevant_info(self, data):
        batch_obs_curr, batch_obs_next = self._get_batched_current_and_next(
            data["image"]
        )
        batch_state_curr, batch_state_next = self._get_batched_current_and_next(
            data["embed"]
        )
        batch_action, _ = self._get_batched_current_and_next(data["action"])
        _, batch_done = self._get_batched_current_and_next(data["is_terminal"])
        batch_done = batch_done.bool()
        batch_obs_targ, batch_state_targ = self._get_goal_obs_and_latent(data)

        return (
            batch_obs_curr,
            batch_action,
            batch_obs_next,
            batch_done,
            batch_obs_targ,
            batch_state_curr,
            batch_state_next,
            batch_state_targ,
        )

    def _get_batched_current_and_next(self, batched_data):
        T = batched_data.shape[1]  # (B, T, ...)
        rest_of_dims = batched_data.shape[2:]
        return batched_data[:, : T - 1].reshape(-1, *rest_of_dims), batched_data[
            :, 1:
        ].reshape(-1, *rest_of_dims)

    def _get_goal_obs_and_latent(self, data):
        """Selects random goal observations and latent states.

        This method selects random observations and latent states from the next batch
        to be used as targets/goals.

        Args:
            data: A dictionary containing batched data.

        Returns:
            A tuple containing the target observations and latent states.
        """
        if self.is_latent_goal:
            batch_state_targ = data["goal_embed"]
            return None, batch_state_targ
        batch_obs_curr, batch_obs_next = self._get_batched_current_and_next(
            data["image"]
        )
        batch_state_curr, batch_state_next = self._get_batched_current_and_next(
            data["embed"]
        )
        B = batch_obs_next.shape[0]
        # we want B random indices to choose from batch_state_next
        random_indices = torch.randint(0, B, (B,))
        batch_obs_targ, batch_state_targ = (
            batch_obs_next[random_indices],
            batch_state_next[random_indices],
        )
        return batch_obs_targ, batch_state_targ

    @torch.no_grad()
    def calculate_rejection_mask_and_distance_from_generated_outputs(
        self, context_feats, generated_feats
    ):
        local_state, generated_local_state = self.local_feature_extrator(
            context_feats
        ), self.local_feature_extrator(generated_feats)
        predicted_discount = (
            self.discount_predictor(local_state, generated_local_state)
            .softmax(-1)
            .max(-2)[0]
        )
        return predicted_discount[:, 0] < self.rejection_tau, predicted_discount


class LocalPerception(torch.nn.Module):
    def __init__(self, config):
        super(LocalPerception, self).__init__()
        self.config = config
        self.layers = get_layers(config["local_arch"])
        # TODO: very hard coded
        self.conv_shape = (
            256,
            4,
            4,
        )

    def forward(self, state):
        x = state
        if len(self.layers["conv"]) > 0:
            x = x.reshape(-1, *self.conv_shape)
            x = self.layers["conv"](x)
            x = x.view(x.size(0), -1)
        x = self.layers["mlp"](x)
        return x


class DiscountValue(torch.nn.Module):
    # Almost like a value function to determine whether we "reach the goal" in a few steps or not
    def __init__(self, config, num_actions, device):
        super(DiscountValue, self).__init__()
        assert len(num_actions) == 1, "Only single action space is supported"
        self.config = config
        self.device = device
        self.num_bins = config["n_bins"]
        self.num_actions = num_actions
        self.gamma = config["gamma"]
        self.support_distance = torch.arange(
            1, self.num_bins + 1, dtype=torch.float32, device=device
        )
        self.support_discount = torch.pow(self.gamma, self.support_distance)
        self.layers = get_layers(
            config["arch"], final_dim=self.num_bins * self.num_actions[0]
        )

    def forward(self, state, goal_state, action=None):
        x = torch.cat([state, goal_state], -1)
        x = self.layers["mlp"](x)  # get combined representation
        if action is not None:
            return self._calculate_discount_for_action(x, action)
        else:
            return self._calculate_discount_for_all_actions(x)

    def _calculate_discount_for_action(self, representation, action):
        size_batch = representation.shape[0]
        representation = representation.reshape(size_batch, -1, self.num_bins)
        representation = representation[
            torch.arange(size_batch, device=representation.device), action.argmax(-1)
        ]
        return representation.contiguous()

    def _calculate_discount_for_all_actions(self, representation):
        size_batch = representation.shape[0]
        representation = representation.reshape(
            size_batch, -1, self.num_bins
        ).contiguous()
        return representation

    def _get_best_action_by_discount(self, state, goal_state):
        weighted_distances = self._calculate_weighted_discount(state, goal_state)
        best_actions = torch.argmax(weighted_distances, dim=-1)
        action_one_hot = torch.nn.functional.one_hot(
            best_actions, num_classes=self.num_actions[0]
        )
        return action_one_hot

    def _calculate_weighted_discount(self, state, goal_state):
        dist_discounts = self(state, goal_state).softmax(-1)
        weighted_distances = dist_discounts @ self.support_discount
        return weighted_distances

    def _calculate_weighted_distance(self, state, goal_state, action):
        dist_discounts = self(state, goal_state, action).softmax(-1)
        weighted_distances = dist_discounts @ self.support_distance
        return weighted_distances


class HistogramConverter(torch.nn.Module):
    """
    consistent scalar <-> histogram converter for distributional outputs
    """

    def __init__(self, config):
        super(HistogramConverter, self).__init__()
        self.register_buffer("value_min", torch.tensor(config["value_min"]))
        self.register_buffer("value_max", torch.tensor(config["value_max"]))
        self.atoms = config["n_bins"]
        self.value_span = config["value_max"] - config["value_min"]
        const_norm = torch.tensor((self.atoms - 1) / self.value_span)
        self.register_buffer("const_norm", const_norm)
        support = torch.arange(self.atoms).float()
        self.register_buffer("support", support)

    def to(self, device):
        super().to(device)
        self.value_min = self.value_min.to(device)
        self.value_max = self.value_max.to(device)
        self.const_norm = self.const_norm.to(device)
        self.support = self.support.to(device)

    @torch.no_grad()
    def to_histogram(self, value):
        value = value.clamp(self.value_min, self.value_max).unsqueeze(
            -1
        )  # NO in-place clipping!!! Do not alter the original
        value_normalized = (
            value - self.value_min
        ) * self.const_norm  # normalize to [0, atoms - 1] range
        value_normalized.clamp_(0, self.atoms - 1)
        upper, lower = value_normalized.ceil().long(), value_normalized.floor().long()
        upper_weight = value_normalized % 1
        lower_weight = 1 - upper_weight
        dist = torch.zeros(
            value.shape[0], self.atoms, device=value.device, dtype=value.dtype
        )
        dist.scatter_add_(-1, lower, lower_weight)
        dist.scatter_add_(-1, upper, upper_weight)
        return dist


class ATCLoss(torch.nn.Module):
    def __init__(self, encoder, config):
        super(ATCLoss, self).__init__()
        self.config = config
        self.K = config["K"]
        self.pad = config["pad"]
        self.encoder = encoder
        self.cnn = encoder._cnn
        self.cnn_dim = config["cnn_dim"]
        self.proj_dim = config["proj_dim"]
        self.std_margin = config["std_margin"]
        self._build_model()

    def _build_model(self):
        self.projector = torch.nn.Linear(self.cnn_dim, self.proj_dim)
        self.anchor_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.proj_dim, self.proj_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.proj_dim, self.proj_dim),
        )
        self.W = torch.nn.Linear(self.proj_dim, self.proj_dim, bias=False)

    def calculate_loss(self, obs):
        anchor_embed, positive_embed = self._get_anchor_and_positive_embedding(obs)
        atc_loss = self._calculate_atc_loss(anchor_embed, positive_embed)
        vicreg_loss = self._calculate_vicreg_loss(anchor_embed, positive_embed)
        return {"atc_loss": atc_loss, **vicreg_loss}

    def _calculate_vicreg_loss(self, anchor, positive):
        variance_loss = self._calculate_variance_loss(anchor, positive)
        invariance_loss = self._calculate_invariance_loss(anchor, positive)
        covariance_loss = self._calculate_covariance_loss(anchor, positive)
        return {"variance_loss": variance_loss, "invariance_loss": invariance_loss, "covariance_loss": covariance_loss}
    
    def _calculate_variance_loss(self, anchor, positive):
        return (self._variance_loss(anchor) + self._variance_loss(positive)) / 2
        
    def _variance_loss(self, x):
        x = x - x.mean(dim=0, keepdim=True)
        x_std = torch.sqrt(x.var(dim=0) + 0.0001)
        return torch.mean(F.relu(self.std_margin - x_std))

    def _calculate_invariance_loss(self, anchor, positive):
        return torch.nn.functional.mse_loss(anchor, positive)

    def _calculate_covariance_loss(self, anchor, positive):
        return (self._covariance_loss(anchor) + self._covariance_loss(positive)) / 2

    def _covariance_loss(self, x):
        B, D = x.shape
        x = x - x.mean(dim=0, keepdim=True)
        x_cov = torch.matmul(x.T, x) / (B - 1) # divide by B - 1 to get unbiased estimate
        x_diagonals = torch.einsum("ii->i", x_cov).pow(2).sum(dim=0)
        return (x_cov.pow(2).sum() - x_diagonals).div(D * (D - 1))

    def _calculate_atc_loss(self, anchor, positive):
        labels = torch.arange(anchor.shape[0], dtype=torch.long, device=anchor.device)
        logits = self._calculate_atc_logits(anchor, positive)
        return torch.nn.functional.cross_entropy(logits, labels)

    def _calculate_atc_logits(self, anchor, positive):
        anchor_proj = self.W(anchor + self.anchor_mlp(anchor))
        logits = torch.matmul(anchor_proj, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        return logits

    def _get_anchor_and_positive_embedding(self, obs):
        anchor, positive = self._get_anchor_and_positive(obs)
        anchor, positive = self._calculate_rearranged_obs(
            anchor
        ), self._calculate_rearranged_obs(positive)
        return self.projector(self.cnn(anchor)), self.projector(self.cnn(positive))

    def _calculate_rearranged_obs(self, obs):
        # assume obs (B T H W C)
        obs = rearrange(obs, "B T H W C -> (B T) C H W")
        obs = random_shift(obs.cpu(), pad=self.pad).to(obs.device)
        obs = rearrange(obs, "BT C H W -> BT H W C")
        return obs

    def _get_anchor_and_positive(self, obs):
        return obs[:, : -self.K], obs[:, self.K :]

    def forward(self, data):
        return self.atc(data)

class WeightPredictor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(WeightPredictor, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, output_dim),
            torch.nn.Softplus()  # Softplus ensures the output weights are positive
        )
    
    def forward(self, x):
        return self.fc(x)

class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, config):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = config["temperature"]
        self.n_labels = config["n_labels"]
        self.latent_dim = config["latent_dim"]
        self.n_views = config["n_views"]

    def calculate_loss(self, features, encoder=None, images=None):
        # features should be of shape (batch_size, time, feature_dim)
        features = F.normalize(features, dim=-1)
        labels = self._calculate_labels(features)
        if encoder == None:
            loss = self._calculate_loss(features, labels)
        else:
            loss = self._calculate_loss_for_encoder(images, labels, encoder)
        return {"supervised_contrastive_loss": loss}
    
    @abstractmethod
    def _calculate_labels(self, features):
        # This method should be implemented in subclasses. Returns B, T, 1 labels
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def _calculate_loss(self, features, labels):
        # This method should be implemented in subclasses. Returns the loss value
        raise NotImplementedError("Subclasses should implement this method.")
   

    def forward(self, features, labels=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        
        mask = self._calculate_mask(labels)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = (torch.exp(logits) + 1e-6) * logits_mask 
        
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = -(mask * log_prob).sum(1) / mask_pos_pairs

        loss = mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
    
    def _calculate_mask(self, labels):
        return torch.eq(labels, labels.T).float().to(labels.device)
    
    @staticmethod
    def build(config):
        loss_type = config["loss_type"]
        if loss_type == "kmeans":
            return KMeansContrastiveLoss(config)
        elif loss_type == "prototype":
            return PrototypeContrastiveLoss(config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    
class KMeansContrastiveLoss(SupervisedContrastiveLoss):
    def __init__(self, config):
        super(KMeansContrastiveLoss, self).__init__(config)
        self.init_centers = None

    def _calculate_labels(self, features):
        # Reshape features to (batch_size * time, feature_dim)
        reshaped_features = features.view(-1, self.latent_dim).detach().cpu().numpy()
        if self.init_centers is None:
            kmeans = KMeans(n_clusters=self.n_labels, random_state=0).fit(reshaped_features)
        else:
            kmeans = KMeans(n_clusters=self.n_labels, init=self.init_centers, n_init=1).fit(reshaped_features)
        
        labels = torch.tensor(kmeans.labels_, device=features.device).view(features.shape[0], -1, 1)
        self.init_centers = kmeans.cluster_centers_
        
        return labels
    
    def _calculate_loss(self, features, labels):
        # Calculate the contrastive loss using the labels generated by KMeans
        x, y = features.view(-1, self.latent_dim), labels.view(-1, 1)
        x = self._generate_views(x)
        return super().forward(x, y)

    def _generate_views(self, x):
        x = x.unsqueeze(1).repeat(1, self.n_views, 1)
        x[:, 1:] = x[:, 1:] + torch.randn_like(x[:, 1:]) * 0.01 # leave the first view unchanged
        return x
    
    

class PrototypeContrastiveLoss(SupervisedContrastiveLoss):
    def __init__(self, config):
        super(PrototypeContrastiveLoss, self).__init__(config)
        self.prototypes = torch.nn.Parameter(torch.randn(self.n_labels, self.latent_dim))
        self.pad = config["pad"]

    def _calculate_labels(self, features):
        reshaped_features = features.view(-1, self.latent_dim)
        prototypes = F.normalize(self.prototypes, dim=1)
        logits = torch.matmul(reshaped_features, prototypes.t()) / self.temperature
        labels = torch.argmax(logits, dim=1).view(features.shape[0], -1, 1)
        return labels

    def _calculate_loss(self, features, labels):
        # Calculate the contrastive loss using the labels generated by KMeans
        x, y = features.view(-1, self.latent_dim), labels.view(-1, 1)
        x = self._generate_views(x)
        return super().forward(x, y) + 3*self._prototype_orthogonality_loss()
    
    def _calculate_loss_for_encoder(self, images, labels, encoder):
        x = self._generate_views_with_encoder(images, encoder)
        
        y = labels.view(-1, 1)
        return super().forward(x, y) + 3*self._prototype_orthogonality_loss()

    def _generate_views(self, x):
        x = x.unsqueeze(1).repeat(1, self.n_views, 1)
        x[:, 1:] = x[:, 1:] + torch.randn_like(x[:, 1:]) * 0.01 # leave the first view unchanged
        return x
    
    def _generate_views_with_encoder(self, images, encoder):
        B, T, H, W, C = images.shape
        x = images.reshape((B*T, 1, H, W, C))
        x = x.repeat(1, self.n_views, 1, 1, 1)
        x[:, 1:] = self._calculate_rearranged_obs(x[:, 1:])
        return encoder._cnn(x)

    def _calculate_rearranged_obs(self, obs):
        B, T, H, W, C = obs.shape
        obs = rearrange(obs, "B T H W C -> (B T) C H W")
        obs = random_shift(obs.cpu(), pad=self.pad).to(obs.device)
        obs = rearrange(obs, "(B T) C H W -> B T H W C", B=B, T=T)
        return obs 

    def _prototype_orthogonality_loss(self):
        prototypes = F.normalize(self.prototypes, dim=1)
        sim = torch.matmul(prototypes, prototypes.t())
        k = prototypes.shape[0]
        diag = torch.eye(k, device=prototypes.device)
        return ((sim - diag)**2).sum() / (k*(k-1))

class TemporalConsistancyLoss(torch.nn.Module):
    def __init__(self, config):
        super(TemporalConsistancyLoss, self).__init__()
        self.config = config
        self.latent_dim = config["latent_dim"]
        self.k = config["k"]
        self.tau = config["tau"]
        self.loss_type = config["loss_type"]
        self.delta = config["delta"]
        if self.loss_type == "predictive":
            self.predictor = torch.nn.Linear(self.latent_dim, 2*self.latent_dim) # mu and logvar
        self.mu, self.std = None, None

    def calculate_loss(self, features):
        # features should be of shape (batch_size, time, feature_dim)
        features = F.normalize(features, dim=-1)
        tc_loss = self._temporal_contrastive_loss(features) if self.loss_type == "contrastive" else self._temporal_predictive_loss(features)
        return {"temporal_contrastive_loss": tc_loss}

    def _temporal_predictive_loss(self, x):
        # anchor and positive used loosely to allude to contrastive learning and not the actual meaning
        anchor = x[:, :-self.k].reshape(-1, self.latent_dim)  # Shape: (B*(T-k), D)
        positive = self._calculate_positive(x)  # Shape: (B*(T-k), k, D)
        predicted_positive = self._calculate_predicted_positive(anchor)  # Shape: (B*(T-k), k, D)
        return self._calculate_weighted_mse_loss(predicted_positive, positive)
    
    def _calculate_weighted_mse_loss(self, predicted_positive, positive):
        weights = torch.exp(-self.delta * torch.arange(self.k, device=predicted_positive.device))
        weights = weights / weights.sum()
        mse_loss = torch.nn.functional.mse_loss(predicted_positive, positive, reduction='none')
        weighted_mse_loss = (mse_loss * weights.unsqueeze(-1).unsqueeze(0)).sum(dim=1).mean(-1)
        self.mu, self.std = weighted_mse_loss.mean(), weighted_mse_loss.std()
        return weighted_mse_loss.mean()
    
    def _calculate_positive(self, x):
        B, T, D = x.shape
        positive_idx = self._calculate_next_k_indices(B, T, x.device)  # (B*(T-k), k)
        positive = x.reshape(-1, D)[positive_idx.reshape(-1)].reshape(-1, self.k, D)  # (B*(T-k), k, D)
        return positive

    def _calculate_predicted_positive(self, anchor):
        mu, logvar = self.predictor(anchor).chunk(2, dim=-1)
        mu = mu.unsqueeze(1).repeat(1, self.k, 1)  # Shape: (B*(T-k), k, D)
        logvar = logvar.unsqueeze(1).repeat(1, self.k, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # Shape: (B*(T-k), k, D)


    def _temporal_contrastive_loss(self, x):
        B, T, D = x.shape
        sim = self._calculate_cosine_similarity(x)  # (B*(T-k), B*T)
        
        postive_idx = self._calculate_next_k_indices(B, T, x.device)  # (B*(T-k), k)
        positive_similarities = sim.gather(dim=1, index=postive_idx)  # (B*(T-k), k)

        all_offset = torch.arange(0, B * T, device=x.device).view(B, T)
        all_idx = all_offset.unsqueeze(1).repeat(1, T - self.k, 1).reshape(-1, T)  # (B*(T-k), T)         
        all_similarities = sim.gather(dim=1, index=all_idx)  # (B*(T-k), T) # we care about the features _in_ the same sequence
                
        loss = torch.logsumexp(all_similarities, dim=1) - torch.logsumexp(positive_similarities, dim=1)
        self.mu, self.std = positive_similarities.mean(-1).mean(), positive_similarities.mean(-1).std()
        return loss.mean()
    
    def _calculate_cosine_similarity(self, x):
        B, T, D = x.shape
        anchors = x[:, :-self.k].reshape(-1, D)  # Shape: (B*(T-k), D)
        candidates = x.reshape(B * T, D) # Shape: (B*T, D)
        
        return torch.matmul(anchors, candidates.T) / self.tau  # Shape: (B*(T-k), B*T)
    
    def _calculate_next_k_indices(self, B, T, device):
        b_idx = torch.arange(B, device=device).unsqueeze(1).repeat(1, T - self.k)  # (B, T-k)
        t_idx = torch.arange(T - self.k, device=device).unsqueeze(0).repeat(B, 1)    # (B, T-k)
        
        # For each anchor, calculate positive indices in the flattened candidate space.
        # For an anchor at time t in batch b, its positive indices are: b * T + (t+1, ..., t+k).
        pos_offset = torch.arange(1, self.k + 1, device=device).view(1, 1, -1)  # (1, 1, k)
        pos_idx = b_idx.unsqueeze(-1) * T + (t_idx.unsqueeze(-1) + pos_offset)  # (B, T-k, k)
        pos_idx = pos_idx.reshape(-1, self.k)  # (B*(T-k), k)
        return pos_idx
    
    @torch.no_grad()
    def calculate_rejection_mask_and_distance_from_generated_outputs(
        self, context_feats, generated_feats
    ):
        # normalize the features
        s_t, s_t_plus_one = F.normalize(context_feats, dim=-1), F.normalize(generated_feats, dim=-1)
        if self.loss_type == "predictive":
            # calculate the distance
            rejection_mask = self._calculate_rejection_mask_predictive(s_t, s_t_plus_one)
        elif self.loss_type == "contrastive":
            rejection_mask = self._calculate_rejection_mask_contrastive(s_t, s_t_plus_one)
        return rejection_mask, None

    def _calculate_rejection_mask_predictive(self, s_t, s_t_plus_one):
        deter_t, deter_t_plus_one = s_t[:, -self.latent_dim:], s_t_plus_one[:, -self.latent_dim:]
        predicted_deter_t_plus_one = self._calculate_predicted_positive(deter_t) # (B, k, D)
        prediction_error = (deter_t_plus_one.unsqueeze(1) - predicted_deter_t_plus_one).pow(2).mean(dim=[1, 2])
        prediction_error = (prediction_error - self.mu) / self.std
        # reject ones with 5% of the distribution
        rejection_mask = prediction_error > 1.645
        return rejection_mask

    def _calculate_rejection_mask_contrastive(self, s_t, s_t_plus_one):
        # calculate the rejection mask based on the cosine similarity
        # both s_t and s_t plus one is (B, d). Calculate cosine similarity to get (B,)
        cosine_similarity = (torch.nn.functional.cosine_similarity(s_t, s_t_plus_one, dim=-1) / self.tau - self.mu) / self.std

        # reject ones with 5% of the distribution
        rejection_mask = cosine_similarity < -1.645
        return rejection_mask


        