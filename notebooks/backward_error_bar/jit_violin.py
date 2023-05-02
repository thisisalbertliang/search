import glob
import numpy as np
import matplotlib.pyplot as plt
from Pk_library import PKL
from tqdm import tqdm
import argparse
import os
import pickle
import torch


class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout-prob', type=float, required=True)
    parser.add_argument('--pow-spec-last', action='store_true', default=False)
    return parser.parse_args()


def plot_log_pow_spec_of_mean_lin(dropout_prob: float):
    box_length: int = 1000 # Mpc/h
    axis: int = 0 # 0, 1, 2 for x, y, z respectively

    pred_lin_files = sorted(glob.glob(
        f'/user_data/ajliang/Quijote/LH0663/mc_dropout/{dropout_prob}/*/lin_out.npy'
    ))
    sample_size = len(pred_lin_files)

    res_file_path = f'cache/pow_spec_last/dropout-prob-{dropout_prob:.0%}-sample-size-{sample_size}.pkl'
    if os.path.exists(res_file_path):
        print(f'{res_file_path} exists. Loading directly...')
        with open(f'{res_file_path}', 'rb') as f:
            res = pickle.load(f)
    else:
        running_stats = RunningStats()
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for pred_lin_file in tqdm(pred_lin_files):
            pred_lin = np.load(pred_lin_file)

            # if device.type == 'cuda':
            #     pred_lin = torch.from_numpy(pred_lin).to(device)

            running_stats.push(pred_lin)

        mean = running_stats.mean()
        std = running_stats.standard_deviation()

        # if device.type == 'cuda':
        #     mean = mean.cpu().numpy()
        #     std = std.cpu().numpy()

        res = {"mean": mean, "std": std}
        with open(f'{res_file_path}', 'wb') as f:
            pickle.dump(res, f)

    mean, std = res['mean'], res['std']

    mean_xpk = PKL.XPk([mean[axis]], box_length, 0, (None, None))
    mean_pow_spec = mean_xpk.Pk[:, 0, 0]
    upper_bound_xpk = PKL.XPk([mean[axis] + std[axis]], box_length, 0, (None, None))
    upper_bound_pow_spec = upper_bound_xpk.Pk[:, 0, 0]
    lower_bound_xpk = PKL.XPk([mean[axis] - std[axis]], box_length, 0, (None, None))
    lower_bound_pow_spec = lower_bound_xpk.Pk[:, 0, 0]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    wave_num = mean_xpk.k3D

    ax.plot(wave_num, mean_pow_spec, label='Power Spectrum of Predicted Linear Field')
    ax.fill_between(
        wave_num, lower_bound_pow_spec, upper_bound_pow_spec,
        alpha=0.2, color='blue',
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wave Number k (log scale)')
    ax.set_ylabel('Power Spectrum P(k) (log scale)')
    ax.legend()
    ax.set_title(f'Dropout Probability: {dropout_prob:.0%}')

    fig.tight_layout()
    fig.savefig(f'figs/pow_spec_last/pow-spec-last_dropout-prob-{int(100 * dropout_prob)}.png')


def plot_log_mean_pow_spec(dropout_prob: float):
    box_length: int = 1000 # Mpc/h
    axis: int = 0 # 0, 1, 2 for x, y, z respectively

    pred_lin_files = sorted(glob.glob(
        f'/user_data/ajliang/Quijote/LH0663/mc_dropout/{dropout_prob}/*/lin_out.npy'
    ))

    sample_size = len(pred_lin_files)
    res_file_path = f'cache/pow_spec_first/dropout-prob-{dropout_prob:.0%}-sample-size-{sample_size}.pkl'
    if os.path.exists(f'{res_file_path}'):
        print(f'{res_file_path} exists. Loading directly...')
        with open(f'{res_file_path}', 'rb') as f:
            res = pickle.load(f)
    else:
        pred_lin_pow_specs = []
        for pred_lin_file in tqdm(pred_lin_files):
            pred_lin = np.load(pred_lin_file)

            xpk = PKL.XPk([pred_lin[axis]], box_length, 0, (None, None))

            pred_lin_auto = xpk.Pk[:, 0, 0]

            pred_lin_pow_specs.append(pred_lin_auto)

        pred_lin_pow_specs = np.vstack(pred_lin_pow_specs)

        res = {'pred_lin_pow_specs': pred_lin_pow_specs, 'wave_num': xpk.k3D}
        with open(f'{res_file_path}', 'wb') as f:
            pickle.dump(res, f)

    pred_lin_pow_specs, wave_num = res['pred_lin_pow_specs'], res['wave_num']

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    # ax.loglog(
    #     wave_num,
    #     np.mean(pred_lin_pow_spec, axis=0),
    #     label='Power Spectrum of Predicted Linear Field'
    # )

    mean = np.mean(pred_lin_pow_specs, axis=0)
    std = np.std(pred_lin_pow_specs, axis=0)
    # max = np.max(pred_lin_pow_specs, axis=0)
    # min = np.min(pred_lin_pow_specs, axis=0)
    # std = 100

    ax.plot(
        wave_num,
        mean,
        label='Power Spectrum of Predicted Linear Field'
    )
    # ax.fill_between(
    #     wave_num,
    #     np.mean(pred_lin_pow_spec, axis=0) - 10,
    #     np.mean(pred_lin_pow_spec, axis=0) + 10,
    #     alpha=0.2,
    #     color='blue'
    # )
    ax.fill_between(
        wave_num,
        mean - std, # min,
        mean + std, # max,
        alpha=0.2,
        color='blue'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Wave Number k (log scale)')
    ax.set_ylabel('Power Spectrum P(k) (log scale)')
    ax.legend()
    ax.set_title(f'Dropout Probability: {dropout_prob:.0%}')

    fig.tight_layout()
    fig.savefig(f'figs/pow_spec_first/pow-spec-first_dropout-prob-{int(100 * dropout_prob)}.png')


if __name__ == '__main__':
    args = parse_args()
    if not args.pow_spec_last:
        plot_log_mean_pow_spec(dropout_prob=args.dropout_prob)
    else:
        plot_log_pow_spec_of_mean_lin(dropout_prob=args.dropout_prob)
