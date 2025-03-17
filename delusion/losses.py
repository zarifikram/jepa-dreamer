import torch


perception_archs = {
    "a": [
        ("conv", 256, 128),
        ("conv", 128, 64),
        ("linear", 1024, 256),
        ("linear", 256, 128),
    ],
    "b": [
        ("linear", 256, 128),
        ("linear", 128, 128),
        ("linear", 128, 64),
    ],
}


def get_layers(arch, final_dim=None):
    arch = perception_archs[arch]
    conv_layers, mlp_layers = [], []
    for i, (type_layer, in_dim, out_dim) in enumerate(arch):
        if type_layer == "conv":
            conv_layers.append(
                torch.nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
            )  # don't change width and height
        elif type_layer == "linear":
            if i == len(arch) - 1 and final_dim:
                out_dim = final_dim
            mlp_layers.append(torch.nn.Linear(in_dim, out_dim))

        if i < len(arch) - 1:
            mlp_layers.append(torch.nn.ReLU())

    return torch.nn.ModuleDict(
        {
            "conv": torch.nn.Sequential(*conv_layers),
            "mlp": torch.nn.Sequential(*mlp_layers),
        }
    )


class FeasibilityEvaluator(torch.nn.Module):
    def __init__(self, evaluator_config, encoder, action_space):
        super(FeasibilityEvaluator, self).__init__()
        self.config = evaluator_config
        self.encoder = encoder
        self.action_space = action_space
        self.local_feature_extrator = LocalPerception(evaluator_config)
        self.discount_predictor = DiscountValue(
            evaluator_config["discount_config"], action_space.shape, encoder.device
        )
        self.histogram_converter = HistogramConverter(
            evaluator_config["histogram_config"]
        )
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

        size_batch = batch_obs_curr.shape[0]
        state_local_curr, state_local_next, state_local_targ = (
            self.calculate_batched_local_state(
                batch_state_curr, batch_state_next, batch_state_targ
            )
        )

        batch_targ_reached = (
            (batch_obs_next == batch_obs_targ).reshape(size_batch, -1).all(-1)
        )

        predicted_discount = self.discount_predictor(
            state_local_curr, state_local_targ, batch_action
        )

        with torch.no_grad():
            action_next = self._calculate_best_action_for_next_step(
                state_local_next.detach(), state_local_targ.detach()
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
        print(loss_discount.sum())
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
        target_distance = distance + 1  # next step is 1 step away
        return target_distance

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

    def calculate_loss(obs, actions, next_obs, dones, goals, encoder):
        states, next_states, goal_states = self._get_latent_representations(
            obs, next_obs, goals
        )

    def _get_latent_representations(self, obs, next_obs, goals):
        encoder.eval()

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
        # TODO: Update it
        batch_obs_curr, batch_obs_next = self._get_batched_current_and_next(
            data["image"]
        )
        batch_state_curr, batch_state_next = self._get_batched_current_and_next(
            data["embed"]
        )
        return batch_obs_next, batch_state_next


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
        x = state.reshape(-1, *self.conv_shape)
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
