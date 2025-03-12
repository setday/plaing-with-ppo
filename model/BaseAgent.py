import torch
from torch import nn

from environment.VectorizedEnvironment import VectorizedEnvironment

class BaseAgent(nn.Module):
    """
    Base class for all agents (critic + actor) in the environment.
    """
    def __init__(self, env: VectorizedEnvironment):
        super(BaseAgent, self).__init__()
        
        self.setup(env)

    def setup(self, env: VectorizedEnvironment):
        """
        Set up the agent with the environment.
        """
        self.env = env

        self.actor = self._setup_actor(env)
        self.critic = self._setup_critic(env)
        
    def _setup_actor(self, env: VectorizedEnvironment) -> nn.Module:
        """
        Set up the actor network.
        """
        raise NotImplementedError("Actor setup not implemented.")
    
    def _setup_critic(self, env: VectorizedEnvironment) -> nn.Module:
        """
        Set up the critic network.
        """
        raise NotImplementedError("Critic setup not implemented.")

    def eval_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the value of a state.
        :param state: The state to evaluate.
        :return: The value of the state.
        """
        return self.critic(state).squeeze(1)
    
    def predict_action(self, state: torch.Tensor) -> tuple:
        """
        Evaluate the action of a state.
        :param state: The state to evaluate.
        :return: The action and its probability.
        """
        raise NotImplementedError("Action prediction not implemented.")

    def eval_action(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Evaluate the action of a state.
        :param state: The state to evaluate.
        :param action: The action to evaluate.
        :return: The action probability and entropy.
        """
        raise NotImplementedError("Action evaluation not implemented.")
