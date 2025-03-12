import click

from tqdm import tqdm

import torch

from model.DiscreteAgent import DiscreteAgent
from model.ContinuousAgent import ContinuousAgent
from environment.BaseEnvironment import BaseEnvironment

@click.command()
@click.option('--env', default='CartPole-v1', help='Gym environment name')
@click.option('--num_steps', default=1000, help='Number of steps to run the agent')
@click.option('--allow_recording', is_flag=True, help='Allow video recording of the environment')
def test(env, num_steps, allow_recording):
    """
    Main function to run the PPO agent on the specified environment.
    :param env: Gym environment name
    :param num_steps: Number of steps to run the agent
    :param allow_recording: Flag to allow video recording of the environment
    :return: None
    """
    
    # Set up the environment
    env = BaseEnvironment(env, allow_recording)
    
    # Set up the agent
    if env.is_discrete():
        agent = DiscreteAgent(env)
    else:
        agent = ContinuousAgent(env)
    agent.load_state_dict(torch.load(f"examples/{env.gym_env_name}/{env.gym_env_name}_ppo_model.pth", weights_only=True))
    agent.eval()
    
    # Capture video of the agent's performance
    state = torch.Tensor(env.reset()[0])
    for step in tqdm(range(num_steps)):
        action = agent.predict_action(state)[0].item()
        state = torch.Tensor(env.step(action)[0])

    # Close the environment
    env.close()

if __name__ == "__main__":
    test()
