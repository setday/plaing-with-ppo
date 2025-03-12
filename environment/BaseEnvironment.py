from typing import Optional

import jax.numpy as jnp

import gymnasium as gym

class BaseEnvironment:
    """
    Base class for all environments.
    """

    def __init__(self, gym_env_name: str, allow_recording: bool = False):
        """
        Initialize the environment.
        :param gym_env_name: Name of the Gym environment.
        :param allow_recording: Flag to allow video recording of the environment.
        :return: None
        """

        self.gym_env_name = gym_env_name
        self.allow_recording = allow_recording

        self.env = self._setup_env()

    def _setup_env(self) -> gym.Env:
        """
        Set up the environment.
        :return: The Gym environment instance.
        """

        env = gym.make(self.gym_env_name, render_mode=("rgb_array" if self.allow_recording else None))
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if self.allow_recording:
            env = gym.wrappers.RecordVideo(env, f"examples/{self.gym_env_name}", episode_trigger=lambda x: x % 100 == 0)

        return env
    
    def get_wrapped_env(self) -> gym.Env:
        """
        Get the wrapped environment.
        :return: The wrapped Gym environment instance.
        """

        return self.env
    
    def get_action_count(self) -> int:
        """
        Get the action space of the environment.
        :return: The action space of the environment.
        """

        return self.env.action_space.n
    
    def get_observation_space_shape(self) -> tuple:
        """
        Get the observation space shape of the environment.
        :return: The observation space shape of the environment.
        """

        return self.env.observation_space.shape
    
    def reset(self) -> tuple:
        """
        Reset the environment.
        :return: Tuple of (observation, info).
        """

        return self.env.reset()
    
    def step(self, action: Optional[int] = None) -> tuple:
        """
        Step the environment.
        :param action: Action to take in the environment.
        :return: Tuple of (observation, reward, done, info).
        """

        return self.env.step(action)
    
    def close(self) -> None:
        """
        Close the environment.
        :return: None
        """

        self.env.close()

    def get_action_count(self) -> int:
        """
        Get the action space of the environment.
        :return: The action space of the environment.
        """

        return self.env.action_space.n
    
    def get_action_space_shape(self) -> tuple:
        """
        Get the action space shape of the environment.
        :return: The action space shape of the environment.
        """

        return self.env.action_space.shape
    
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

        return self.env.observation_space.shape
    
    def get_observation_space_parameters_count(self) -> int:
        """
        Get the number of parameters in the observation space.
        :return: The number of parameters in the observation space.
        """

        return jnp.array(self.get_observation_space_shape()).prod()
    
    def is_discrete(self) -> bool:
        """
        Check if the action space is discrete.
        :return: True if the action space is discrete, False otherwise.
        """

        return isinstance(self.env.action_space, gym.spaces.Discrete)
