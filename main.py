import click

import torch
from torch.utils.tensorboard import SummaryWriter

from model.DiscreteAgent import DiscreteAgent
from model.ContinuousAgent import ContinuousAgent
from environment.VectorizedEnvironment import VectorizedEnvironment
from optimizer.PPO import PPO, PPOConfig

# environments to test:
# MountainCar-v0
# CartPole-v1
# Acrobot-v1
# MountainCarContinuous-v0
# Pendulum-v1

@click.command()
@click.option('--env', default='Pendulum-v1', help='Gym environment name')
@click.option('--num_envs', default=4, help='Number of parallel environments')
@click.option('--allow_recording', is_flag=True, help='Allow video recording of the environment')
@click.option('--allow_tensorboard', is_flag=True, help='Allow tensorboard recording of the environment')
def main(env, num_envs, allow_recording, allow_tensorboard):
    """
    Main function to run the PPO agent on the specified environment.
    :param env: Gym environment name
    :param num_envs: Number of parallel environments
    :param allow_recording: Flag to allow video recording of the environment
    :return: None
    """

    # Set up tensorboard
    if allow_tensorboard:
        tensorboard_path = f"tensorboard/{env}_ppo"
        tensorboard_writer = SummaryWriter(tensorboard_path)
    else:
        tensorboard_writer = None
    
    # Set up the environment
    env = VectorizedEnvironment(env, num_envs, allow_recording)
    
    # Set up the agent
    if env.is_discrete():
        agent = DiscreteAgent(env)
    else:
        agent = ContinuousAgent(env)
    
    # Set up the PPO algorithm
    ppo = PPO(env, agent, PPOConfig(), tensorboard_writer)
    
    # Train the agent
    ppo.train()

    # Save the model
    torch.save(agent.state_dict(), f"examples/{env.gym_env_name}/{env.gym_env_name}_ppo_model.pth")

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
