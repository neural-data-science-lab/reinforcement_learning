import argparse
import os
import time

import gymnasium as gym
from stable_baselines3 import PPO

from sb3_contrib import TRPO
from plot_results import get_experiment_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    args = parser.parse_args()

    experiment_path = args.fpath
    info = get_experiment_info(experiment_path)

    env = gym.make(info["env_type"], render_mode="human")

    model = PPO.load(os.path.join(experiment_path, "model"), print_system_info=True)

    obs, _ = env.reset()
    for i in range(3000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()

        if done:
            break
