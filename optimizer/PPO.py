from typing import Optional

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.BaseAgent import BaseAgent
from environment.VectorizedEnvironment import VectorizedEnvironment

class PPOConfig:
    """
    Configuration class for Proximal Policy Optimization (PPO).
    This class contains hyperparameters and settings for the PPO algorithm.
    """

    def __init__(self,
                 learning_rate: float = 3e-3, learning_rate_desc: float = 0.99,
                 gamma: float = 0.99, clip_range: float = 0.2,
                 num_epochs: int = 8, num_steps: int = 129, batch_size: float = 16,
                 ent_coef: float = 0.05, vf_coef: float = 0.25):
        """
        Initialize the PPO configuration with hyperparameters.
        :param learning_rate: Learning rate for the optimizer.
        :param learning_rate_desc: Learning rate decay factor.
        :param gamma: Discount factor for future rewards.
        :param clip_range: Clipping range for PPO.
        :param num_epochs: Number of epochs for training.
        :param num_steps: Number of steps to collect before updating the policy.
        :param batch_size: Batch size for training.
        :param ent_coef: Coefficient for entropy loss.
        :param vf_coef: Coefficient for value function loss.
        """

        self.learning_rate = learning_rate
        self.learning_rate_desc = learning_rate_desc
        self.gamma = gamma
        self.clip_range = clip_range
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm.
    This class implements the PPO algorithm for reinforcement learning.
    """

    def __init__(self, envs: VectorizedEnvironment, actor_critic: BaseAgent, config: PPOConfig, tensorboard_writer: Optional[SummaryWriter] = None):
        """
        Initialize the PPO algorithm with the environment, actor-critic model, and configuration.
        :param env: The environment for training.
        :param actor_critic: The actor-critic model.
        :param config: Configuration for PPO.
        :param tensorboard_writer: Optional TensorBoard writer for logging.
        """
        self.env = envs
        self.actor_critic = actor_critic
        self.config = config
        self.tensorboard_writer = tensorboard_writer
        
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer)

        self.actor_critic.train()

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """
        Set up the optimizer for the actor-critic model.
        """
        return torch.optim.AdamW(self.actor_critic.parameters(), lr=self.config.learning_rate)
    
    def _setup_scheduler(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
        """
        Set up the learning rate scheduler for the optimizer.
        """
        return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: self.config.learning_rate_desc)
    
    def train(self, num_milestones: int = 100):
        """
        Train the PPO algorithm.
        """
        start_observation = torch.tensor(self.env.reset()[0])

        for milestone in tqdm(range(num_milestones), desc="Training PPO"):
            # Collect trajectories
            observations, actions, rewards, dones, values, probs = self._compute_trajectories(start_observation, milestone)
            start_observation = observations[-1]

            # Compute masks for done states
            masks = 1 - dones

            # Compute advantages and returns
            advantages = self._compute_advantages(rewards, values, masks)

            # Flatten the data for training to make it "time independent"
            observations = observations[:-1].reshape(-1, *observations.shape[2:])
            actions = actions[:-1].reshape(-1, *actions.shape[2:])
            advantages = advantages[:-1].reshape(-1)
            probs = probs[:-1].reshape(-1, *probs.shape[2:])
            rewards = rewards[:-1].reshape(-1)

            # Update policy using the collected trajectories
            value_target = rewards + advantages
            for epoch in tqdm(range(self.config.num_epochs), desc="Updating Policy", leave=False):
                self._update_policy(observations, actions, advantages, value_target, probs, milestone, epoch)

            # Update the learning rate
            self.scheduler.step()

            # Log the training progress
            if self.tensorboard_writer:
                self.tensorboard_writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], milestone)

    def _compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Compute advantages using Generalized Advantage Estimation (GAE).
        PS: Last element will have no advantage
        :param rewards: Rewards from the environment.
        :param values: Values from the critic.
        :param masks: Masks for done states.
        :return: Computed advantages.
        """

        # A = r_t + gamma * V(s_{t+1}) - V(s_t) + gamma * A_{t+1}
        advantages = rewards - values
        advantages[len(advantages)-1] = 0.0  # Last element has no advantage

        for t in reversed(range(len(rewards)-1)):
            advantages[t] += self.config.gamma * masks[t] * (advantages[t + 1] + values[t + 1])

        return advantages
    
    def _compute_value_loss(self, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute value loss.
        :param values: Values from the critic.
        :param returns: Computed returns.
        :return: Computed value loss.
        """
        value_loss = torch.mean((values - returns) ** 2)
        return value_loss
    
    def _compute_policy_loss(self, probs: torch.Tensor, old_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss.
        :param probs: Probabilities from the actor.
        :param old_probs: Old probabilities for importance sampling.
        :param advantages: Computed advantages.
        :return: Computed policy loss.
        """
        ratio = probs / (old_probs + 1e-10)
        surr1 = ratio * advantages
        surr2 = torch.clip(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * advantages
        policy_loss = -torch.mean(torch.minimum(surr1, surr2))
        return policy_loss
    
    def _compute_full_loss(self, policy_loss: torch.Tensor, value_loss: torch.Tensor, entropy_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute the full loss for the PPO update.
        :param policy_loss: Computed policy loss.
        :param value_loss: Computed value loss.
        :param entropy_loss: Computed entropy loss.
        :return: Computed full loss.
        """
        full_loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy_loss
        return full_loss

    def _compute_trajectories(self, start_observation: torch.Tensor, milestone: int = 0) -> tuple:
        """
        Compute trajectories from the environment.
        :param start_observation: Initial observation for the environment.
        :param milestone: Current milestone for logging.
        :return: Trajectories from the environment (observations, rewards, done, actions, values, probs).
        """
        array_shape = (self.config.num_steps, self.env.get_environment_count())

        observations = torch.zeros(array_shape + self.env.get_observation_space_shape())
        rewards = torch.zeros(array_shape)
        dones = torch.zeros(array_shape)

        actions = torch.zeros(array_shape + self.env.get_action_space_shape())
        probs = torch.zeros(array_shape + self.env.get_action_space_shape())
        values = torch.zeros(array_shape)

        for step in range(self.config.num_steps):
            if step == 0:
                observations[step] = start_observation
            step_observations = observations[step]
                
            with torch.no_grad():
                step_actions, step_probs = self.actor_critic.predict_action(step_observations)
                step_values = self.actor_critic.eval_value(step_observations)

            next_observations, step_rewards, step_terminated, step_truncated, step_info = self.env.step(step_actions.cpu().numpy())
            step_dones = step_terminated | step_truncated

            rewards[step], dones[step] = torch.tensor(step_rewards), torch.tensor(step_dones)
            values[step], actions[step], probs[step] = step_values, step_actions, step_probs

            if step < self.config.num_steps - 1:
                observations[step + 1] = torch.tensor(next_observations)

            if self.tensorboard_writer and "episode" in step_info:
                time_step = milestone * self.config.num_steps + step

                episode_rewards = step_info["episode"]["r"]
                episode_lengths = step_info["episode"]["l"]

                reward_s = sum(episode_rewards)
                length_s = sum(episode_lengths)
                episode_count = sum(length > 0 for length in episode_lengths)

                self.tensorboard_writer.add_scalar('episode/avg_episode_reward', reward_s / episode_count, time_step)
                self.tensorboard_writer.add_scalar('episode/avg_episode_length', length_s / episode_count, time_step)
                

        return observations, actions, rewards, dones, values, probs
    
    def _update_policy(self, observations, actions, advantages, value_targets, old_probs, milestone: int = 0, epoch: int = 0):
        """
        Update the policy using the collected trajectories.
        :param observations: Observations from the environment.
        :param actions: Actions taken in the environment.
        :param advantages: Computed advantages.
        :param value_targets: Computed value_targets.
        :param values: Values from the critic.
        :param old_probs: Old probabilities for importance sampling.
        :param milestone: Current milestone for logging.
        :param epoch: Current epoch for logging.
        """

        sample_loader = DataLoader(
            list(zip(observations, actions, value_targets, advantages, old_probs)),
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for batch in sample_loader:
            # Unpack batch
            observations_batch, actions_batch, value_targets_batch, advantages_batch, old_probs_batch = batch

            # Compute policy loss
            probs, entropy = self.actor_critic.eval_action(observations_batch, actions_batch)
            values_batch = self.actor_critic.eval_value(observations_batch)
            
            # Compute loss
            policy_loss = self._compute_policy_loss(probs, old_probs_batch, advantages_batch)
            value_loss = self._compute_value_loss(values_batch, value_targets_batch)
            entropy_loss = entropy.mean()
            full_loss = self._compute_full_loss(policy_loss, value_loss, entropy_loss)
            
            # Backpropagation and optimization step
            self.optimizer.zero_grad()
            full_loss.backward()
            self.optimizer.step()

        # Log the training progress
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar('loss/policy_loss', policy_loss.item(), milestone * self.config.num_epochs + epoch)
            self.tensorboard_writer.add_scalar('loss/value_loss', value_loss.item(), milestone * self.config.num_epochs + epoch)
            self.tensorboard_writer.add_scalar('loss/entropy_loss', entropy_loss.item(), milestone * self.config.num_epochs + epoch)
            self.tensorboard_writer.add_scalar('loss/full_loss', full_loss.item(), milestone * self.config.num_epochs + epoch)
