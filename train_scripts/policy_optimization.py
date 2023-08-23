import datetime
import os
import random
from argparse import ArgumentParser

import numpy as np
import yaml

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure as sb3_configure_logger
import torch as th
from torch import nn

from sb3_contrib import TRPO, CustomTRPO


def write_info(experiment_path, info):
    with open(os.path.join(experiment_path, "info.yml"), "w") as f:
        f.write(yaml.dump(info))


def requires_map(env_name: str) -> bool:
    return env_name in ["AntMaze-UMazeDense-v3", "AntMaze-UMaze-v3"]


def train_sb3():
    parser = ArgumentParser()
    parser.add_argument('--algo', default="CustomTRPO", type=str, help="name of the algorithm")
    parser.add_argument('--div', default="IS", type=str, help="the divergence measure to use in TRPO")
    parser.add_argument('--env', default="HalfCheetah-v3", type=str, help="name of the gym environment")
    parser.add_argument('--seed', default=0, type=int, help="manual seed")
    args = parser.parse_args()

    with open(os.path.join("config", f'{args.algo}.yml'), "r") as f:
        try:
            params = yaml.load(f, yaml.Loader)[args.env]
            total_timesteps = params.pop("n_timesteps")
            if params.get("policy_kwargs"):
                params["policy_kwargs"] = exec(params["policy_kwargs"])
        except KeyError:
            print(f"No hyperparameters for {args.env} found")
            params = {'policy': "MultiInputPolicy"}
            total_timesteps = 1_000_000

    print(f'Training {args.algo} with params {params}')

    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    if requires_map(args.env):
        env = gym.make(args.env, maze_map=[[1, 1, 1, 1, 1],
                                           [1, 0, 0, 'g', 1],
                                           [1, 0, 1, 0, 1],
                                           [1, 'r', 0, 0, 1],
                                           [1, 1, 1, 1, 1]])
    else:
        env = gym.make(args.env)

    if args.algo == "TRPO":
        model = TRPO(env=env, **params)
    elif args.algo == "PPO":
        model = PPO(env=env, **params)
    elif args.algo == "CustomTRPO":
        model = CustomTRPO(env=env, div=args.div, **params)
    else:
        raise KeyError(f"Algorithm {args.algo} unknown")

    experiment_path = os.path.join("results", args.env, f"{args.algo}_{args.div}", f"seed_{args.seed}")
    os.makedirs(experiment_path, exist_ok=True)

    eval_callback = EvalCallback(env, log_path=experiment_path, eval_freq=10000, deterministic=True,
                                 render=False, n_eval_episodes=3, warn=False, verbose=False)

    os.makedirs(experiment_path, exist_ok=True)

    write_info(experiment_path, {
            "algo": args.algo,
            "env_type": args.env,
            "seed": args.seed,
            "training_start_time": datetime.datetime.now()
    })

    new_logger = sb3_configure_logger(experiment_path, ["csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(
        total_timesteps=total_timesteps,
        log_interval=10000,
        progress_bar=True,
        callback=[eval_callback]  # , state_trajectory_callback
    )

    model.save(os.path.join(experiment_path, "model.zip"))

    return model


if __name__ == '__main__':
    train_sb3()
