import argparse
import os
import time

import gymnasium as gym
from stable_baselines3 import SAC
from sb3_contrib import CicDDPG

from sb3_contrib import TRPO
from plot_results import get_experiment_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--algo', default="CicDDPG", type=str, help="name of the algorithm")
    args = parser.parse_args()

    experiment_path = args.fpath
    info = get_experiment_info(experiment_path)

    env = gym.make(info["env"], render_mode="human")

    if args.algo == "CicDDPG":
        model = CicDDPG.load(os.path.join(experiment_path, "model"), print_system_info=True)
    else:
        raise KeyError(f"Algorithm {args.algo} unknown")

    obs, _ = env.reset()
    for i in range(3000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        env.render()

        if done:
            break
