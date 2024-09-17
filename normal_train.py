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
from helpers import train, test
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--training_steps", type=int, default=1000)
    parser.add_argument("--number_of_tests", type=int, default=3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    SAMPLES = args.samples
    TRAIN_STEPS = args.training_steps
    number_of_tests = args.number_of_tests
    all_avg_rewards = []

    for i in range(SAMPLES):
        # Create a seeded environment
        env = simple_spread_v3.parallel_env(local_ratio=0, N=2)

        model_path = "simple_spread_normal/ppo/simple_spread_v3.zip"

        # Train and test
        train(env, model_path, training_steps=200, num_cpus=1, num_instances=1)

        # Test the model
        avg_reward = test(env, model_path, num_games=100)
        all_avg_rewards.append(avg_reward)
    
    
    print("Test Average Reward:", sum(all_avg_rewards) / len(all_avg_rewards))
    f = open("simple_spread_normal/ppo/avg_rewards.txt", "w")
    f.write(str(sum(all_avg_rewards) / len(all_avg_rewards)))
    f.close()

