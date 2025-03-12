import torch
from torch import nn
from torch.distributions import Normal

from environment.VectorizedEnvironment import VectorizedEnvironment

from model.BaseAgent import BaseAgent

class ContinuousAgent(BaseAgent):
    """
    ContinuousAgent class for discrete action space environments. Pretty standard multi-layer perceptron with
    hidden layer of size 64 and tanh activation function.
    Inherits from BaseAgent and implements the setup for discrete action spaces.
    """
    
    def __init__(self, env: VectorizedEnvironment):
        super(ContinuousAgent, self).__init__(env)
        
    def _setup_actor(self, env: VectorizedEnvironment):
        """
        Set up the actor network for discrete action spaces.
        """
        self.std_actor = nn.Parameter(torch.zeros(env.get_action_space_parameters_count()))

        return nn.Sequential(
            nn.Linear(env.get_observation_space_parameters_count(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, env.get_action_space_parameters_count()),
        )
        
    def _setup_critic(self, env: VectorizedEnvironment):
        """
        Set up the critic network for discrete action spaces.
        """
        return nn.Sequential(
            nn.Linear(env.get_observation_space_parameters_count(), 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def predict_action(self, state: torch.Tensor) -> tuple:
        """
        Evaluate the action of a state.
        :param state: The state to evaluate.
        :return: The action and its probability.
        """
        logits = self.actor(state)
        sampler = Normal(logits, self.std_actor.exp())
        action = sampler.sample()
        if action.dim() > 0:
            action_prob = sampler.log_prob(action)
        else:
            action_prob = sampler.log_prob(action).sum(dim=-1)
        action_prob = torch.exp(action_prob)
        return action, action_prob

    def eval_action(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Evaluate the action of a state.
        :param state: The state to evaluate.
        :param action: The action to evaluate.
        :return: The action probability and entropy.
        """
        logits = self.actor(state)
        sampler = Normal(logits, self.std_actor.exp())
        action_prob = sampler.log_prob(action)
        entropy = sampler.entropy()
        return action_prob, entropy