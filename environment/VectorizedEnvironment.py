import jax.numpy as jnp

import gymnasium as gym

from environment.BaseEnvironment import BaseEnvironment

class VectorizedEnvironment(BaseEnvironment):
    """
    Vectorized environment for multiple instances.
    """

    def __init__(self, gym_env_name: str, num_envs: int, allow_recording: bool = False):
        """
        Initialize the vectorized environment.
        :param gym_env_name: Name of the Gym environment.
        :param num_envs: Number of parallel environments.
        :param allow_recording: Flag to allow video recording of the environment.
        :return: None
        """

        self.gym_env_name = gym_env_name
        self.allow_recording = allow_recording
        
        self.num_envs = num_envs
        self.env = self._setup_vectorized_env()

    def _setup_vectorized_env(self) -> gym.Env:
        """
        Set up the vectorized environment.
        :return: The vectorized Gym environment instance.
        """
        
        envs = [self._setup_env for _ in range(self.num_envs)]
        env = gym.vector.SyncVectorEnv(envs)
        
        return env
    
    def _setup_env(self) -> gym.Env:
        """
        Set up a single environment instance.
        :return: The Gym environment instance.
        """
        env = super()._setup_env()

        self.allow_recording = False # Disable recording for individual environments in vectorized setup

        return env
    
    def get_environment_count(self) -> int:
        """
        Get the number of environments.
        :return: The number of environments.
        """

        return self.num_envs
    
    def get_action_count(self) -> int:
        """
        Get the action space of the environment.
        :return: The action space of the environment.
        """

        return int(self.env.single_action_space.n)
    
    def get_action_space_shape(self):
        """
        Get the action space shape of the environment.
        :return: The action space shape of the environment.
        """

        return self.env.single_action_space.shape
    
    def get_action_space_parameters_count(self) -> int:
        """
        Get the number of parameters in the action space.
        :return: The number of parameters in the action space.
        """

        return jnp.array(self.get_action_space_shape()).prod()
    
    def get_observation_space_shape(self) -> tuple:
        """
        Get the observation space shape of the environment.
        :return: The observation space shape of the environment.
        """
        
        return self.env.single_observation_space.shape
    
    def get_observation_space_parameters_count(self) -> int:
        """
        Get the number of parameters in the observation space.
        :return: The number of parameters in the observation space.
        """

        return jnp.array(self.get_observation_space_shape()).prod()

    def is_discrete(self):
        """
        Check if the environment has a discrete action space.
        :return: True if the action space is discrete, False otherwise.
        """

        return isinstance(self.env.single_action_space, gym.spaces.Discrete)
