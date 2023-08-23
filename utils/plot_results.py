import os
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def find_paths(fpath):
    paths = [os.path.join(fpath, d) for d in os.listdir(fpath) if os.path.isdir(os.path.join(fpath, d))]
    print(f"Joining info from experiments")
    print(paths)
    return paths


def get_experiment_info(experiment_path):
    with open(os.path.join(experiment_path, 'info.yml'), "r") as f:
        info = yaml.load(f, yaml.Loader)

    return info


def plot_sb3_results(fpath, col, single):
    if single:
        frame = pd.read_csv(os.path.join(fpath, "progress.csv"))
        info = get_experiment_info(fpath)
        for key, value in info.items():
            frame[key] = value

    else:
        li = []

        for path in find_paths(fpath):
            try:
                df = pd.read_csv(os.path.join(path, "progress.csv"))
                info = get_experiment_info(path)
                for key, value in info.items():
                    df[key] = value
                li.append(df)
            except pd.errors.EmptyDataError:
                pass

        frame = pd.concat(li, axis=0, ignore_index=True)

    print("Available columns:", frame.columns)

    # y="eval/mean_ep_length" is also useful
    sns.lineplot(frame, x="time/total_timesteps", y=col)
    plt.title(col)
    plt.savefig(os.path.join(fpath, f"{col.replace('/', '_')}.png"))
    plt.show()


def plot_custom_results(exp_dir, col):
    li = []

    for path in find_paths(exp_dir):
        df = pd.read_csv(os.path.join(path, "progress.txt"), delimiter="\t")
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)

    sns.lineplot(frame, x="TotalEnvInteracts", y=col)
    plt.title(col)
    plt.savefig(os.path.join(exp_dir, f"{col}.png"))
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str, nargs='+')
    parser.add_argument('--col', '-c', type=str, default="eval/mean_reward")
    parser.add_argument('--single', '-s', action='store_true', help="no averaging, "
                                                                    "interpret 'fpath' as a single experiment")
    args = parser.parse_args()

    plot_sb3_results(args.fpath, args.col, args.single)
