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
import argparse
from helpers import train, test

def get_agent_i(env, i):
    env = env.unwrapped
    agent = env.world.agents[i].state
    radius = env.world.agents[i].size
    absolute_x = agent.p_pos[0]-radius
    absolute_y = agent.p_pos[1]-radius
    width = height = radius*2
    vx = agent.p_vel[0]
    vy = agent.p_vel[1]
    return absolute_x, absolute_y, absolute_y, width, height, vx, vy

def get_landmark_i(env, i):
    env = env.unwrapped
    landmark = env.world.landmarks[i].state
    radius = env.world.landmarks[i].size
    absolute_x = landmark.p_pos[0]-radius
    absolute_y = landmark.p_pos[1]-radius
    width = height = radius*2
    vx = landmark.p_vel[0]
    vy = landmark.p_vel[1]
    return absolute_x, absolute_y, width, height, vx, vy

def overwrite_and_pad_obs(env, obs, n_x, n_y, n_vx, n_vy):
    # Create a zero numpy array with the shape of obs
    new_obs = np.zeros(obs.shape[0])
    new_obs[0] = n_x
    new_obs[1] = n_y
    new_obs[2] = n_vx
    new_obs[3] = n_vy
    return new_obs




def decorate_step(env, original_step):
    def new_step(action):
        # Call the original step method
        obs, reward, termination, truncation, info = original_step(action)
        if "agent_0" in obs == False or "agent_1" in obs == False or obs == {}:
            return obs, reward, termination, truncation, info
       
        # TODO: Modify the observation here
        #print(get_agent_i(env, 0))
        #print(get_landmark_i(env, 0))

        #print(get_agent_i(env, 1))
        #print(get_landmark_i(env, 1))

        n_agent1_x = 0
        n_agent1_y = 0
        n_agent1_vx = 0
        n_agent1_vy = 0

        n_agent2_x = 0
        n_agent2_y = 0
        n_agent2_vx = 0
        n_agent2_vy = 0
        print(obs)
        print(termination, truncation)
        obs["agent_0"] = overwrite_and_pad_obs(env, obs["agent_0"], n_agent1_x, n_agent1_y, n_agent1_vx, n_agent1_vy)
        obs["agent_1"] = overwrite_and_pad_obs(env, obs["agent_1"], n_agent2_x, n_agent2_y, n_agent2_vx, n_agent2_vy)
        print(env.current_step)
        
        # set old_obs
        env.old_obs = obs # If you need more than the last observation, you can store them in a list
        env.current_step += 1
        return obs, reward, termination, truncation, info

    return new_step


def decorate_reset(env, original_reset):
    def new_reset(seed=None, options=None):
        # Access environment internal state before reset
        env.old_obs = None
        env.current_step = 0
        
        # Call the original reset method
        obs, info = original_reset(seed=seed, options=options)
        if "agent_0" in obs == False or "agent_1" in obs == False:
            return obs, info
        
        # TODO: Modify the observation here
        #print(get_agent_i(env, 0))
        #print(get_landmark_i(env, 0))

        #print(get_agent_i(env, 1))
        #print(get_landmark_i(env, 1))

        n_agent1_x = 0
        n_agent1_y = 0
        n_agent1_vx = 0
        n_agent1_vy = 0

        n_agent2_x = 0
        n_agent2_y = 0
        n_agent2_vx = 0
        n_agent2_vy = 0
        print(obs)
        
        obs["agent_0"] = overwrite_and_pad_obs(env, obs["agent_0"], n_agent1_x, n_agent1_y, n_agent1_vx, n_agent1_vy)
        obs["agent_1"] = overwrite_and_pad_obs(env, obs["agent_1"], n_agent2_x, n_agent2_y, n_agent2_vx, n_agent2_vy)
        #print(env.current_step)
        # set old_obs
        env.old_obs = obs # If you need more than the last observation, you can store them in a list
        env.current_step += 1
        return obs, info

    return new_reset



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

        # Decorate the environment
        env.step = decorate_step(env, env.step)
        env.reset = decorate_reset(env, env.reset)

        model_path = "simple_spread_modified/ppo/simple_spread_v3.zip"

        # Train and test
        train(env, model_path, training_steps=200, num_cpus=1, num_instances=1)

        
        
        env = simple_spread_v3.parallel_env(local_ratio=0, N=2)
        # Decorate the environment
        env.step = decorate_step(env, env.step)
        env.reset = decorate_reset(env, env.reset)
        # Test the model
        avg_reward = test(env, model_path, num_games=100)
        all_avg_rewards.append(avg_reward)
    
    
    print("Test Average Reward:", sum(all_avg_rewards) / len(all_avg_rewards))
    f = open("simple_spread_modified/ppo/avg_rewards.txt", "w")
    f.write(str(sum(all_avg_rewards) / len(all_avg_rewards)))
    f.close()