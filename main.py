import gymnasium as gym
import numpy as np
import os
import ale_py
from data_utils import ReplayMemory

gym.register_envs(ale_py)

# Initialize the Atari Breakout environment
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

memory = ReplayMemory()



while 1 == 1:# Reset the environment to start a new episode
    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        # Randomly select an action from the action space
        action = env.action_space.sample()

        # Take the action in the environment
        obs, reward, done, truncated, info = env.step(action)
        print(obs.shape)
        # Accumulate the reward
        total_reward += reward

        # Optional: Print current reward and info
        print(f"Action: {action}, Reward: {reward}, Total Reward: {total_reward}")

    # Print final score
    print(f"Game Over! Final Score: {total_reward}")

    # Close the environment
    env.close()


