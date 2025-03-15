import torch

class FeasibilityEvaluator(torch.nn.Module):
    def __init__(evaluator_config, ):
        super(FeasibilityEvaluator, self).__init__(
        
    def forward()

    def calculate_loss(obs, actions, next_obs, dones, goals, encoder):
        states, next_states, goal_states = self._get_latent_representations(obs, next_obs, goals)

    def _get_latent_representations(obs, next_obs, goals):
        encoder.eval()
