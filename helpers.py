import glob
import os
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import numpy as np
from pettingzoo.mpe import simple_spread_v3
import random
import torch
def train(
    env, save_model_path, training_steps=10_000, num_cpus=1, num_instances=1
):

    # Pass the environment to SuperSuit after seeding
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_instances, num_cpus=num_cpus, base_class="stable_baselines3")


    # Create the PPO model without a seed argument
    model = PPO(
        MlpPolicy,
        env,
        verbose=3,
        learning_rate=1e-3,
        batch_size=256
    )

    model.learn(total_timesteps=training_steps)

    model.save(save_model_path)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def test(env, model_path, num_games=100):
    # Load the latest trained model
    try:
        latest_policy = max(glob.glob(model_path), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = PPO.load(latest_policy)  # Load PPO model

    # Evaluate the trained model
    rewards = []
    for i in range(num_games):
        obs, _ = env.reset()  # Reset environment with seed
        done = False
        episode_reward = 0
        while not done:
            actions = {}
            for agent in env.agents:
                actions[agent] = model.predict(obs[agent], deterministic=True)[0]
                
            obs, reward, termination, truncation, info = env.step(actions)

            # Sum the rewards that each agent received
            episode_reward += sum(reward.values())
            done = all(termination.values()) and all(truncation.values())
        rewards.append(episode_reward)
    
    avg_reward = sum(rewards) / len(rewards)
    
    return avg_reward