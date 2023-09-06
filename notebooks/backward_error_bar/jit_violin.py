import numpy as np; np.random.seed(10620)
import matplotlib.pyplot as plt
from Pk_library import PKL
from tqdm import tqdm
import argparse
import os
from itertools import islice
import sys; sys.path.append('/ocean/projects/cis230021p/lianga/search')

import torch; torch.manual_seed(10620)
from torch.utils.data import DataLoader

from search.map2map.map2map.data import FieldDataset
from search.map2map.map2map.models import StyledVNet, narrow_cast
from search.map2map.map2map.utils import load_model_state_dict


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
    # uncertainty
    parser.add_argument('--dropout-prob', type=float, required=True)
    parser.add_argument('--pow-spec-last', action='store_true', default=False)
    parser.add_argument('--sample-size', type=int, default=30)
    # data
    parser.add_argument('--simulation-number', type=int, default=1045)
    parser.add_argument('--data-dir', type=str, default='/ocean/projects/cis230021p/lianga/quijote')
    parser.add_argument('--sample-index', type=int, default=150)
    # misc
    parser.add_argument('--in-norms', type=str, default='cosmology.dis')
    parser.add_argument('--tgt-norms', type=str, default='cosmology.dis')
    parser.add_argument('--callback-at', type=str, default='.')
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--in-pad', type=int, default=48)
    parser.add_argument('--tgt-pad', type=int, default=48)
    parser.add_argument('--scale-factor', type=int, default=1)
    # model
    parser.add_argument('--model-state-dict', type=str, default='/ocean/projects/cis230021p/lianga/search/model_weights/backward_model.pt')
    # misc
    parser.add_argument('--experiment-name', type=str, default=None)

    return parser.parse_args()


def get_data_loader(args):
    data_dir = os.path.join(args.data_dir, f'LH{args.simulation_number}')

    dataset = FieldDataset(
        style_pattern=os.path.join(data_dir, 'params.npy'),
        in_patterns=[os.path.join(data_dir, 'nonlin.npy')],
        tgt_patterns=[os.path.join(data_dir, 'lin.npy')],
        in_norms=[args.in_norms],
        tgt_norms=[args.tgt_norms],
        callback_at=args.callback_at,
        crop=args.crop,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
    )

    args.style_size = dataset.style_size
    args.in_chan = dataset.in_chan
    args.out_chan = dataset.tgt_chan

    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    return loader


def get_pow_spec(pred: torch.Tensor, target: torch.Tensor, box_length: int, axis: int):
    assert pred.shape == target.shape

    pred = pred.squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()
    xpk = PKL.XPk((pred[axis], target[axis]), box_length, 0, (None, None))

    pred_auto = xpk.Pk[:,0,0] #auto ps of predicted field
    target_auto = xpk.Pk[:,0,1] # auto ps of original field
    cross = xpk.XPk[:,0][:,0] # cross ps
    wave_num = xpk.k3D

    return pred_auto, target_auto, cross, wave_num

    # xpk = PKL.XPk([X[axis]], box_length, 0, (None, None))
    # auto = xpk.Pk[:, 0, 0]
    # wave_num = xpk.k3D
    # return auto, wave_num


def plot_log_pow_spec_of_mean_lin(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    box_length: int = 1000 # Mpc/h
    axis: int = 0 # 0, 1, 2 for x, y, z respectively

    # load data
    loader = get_data_loader(args)
    # get the nth sample from the loader
    sample = next(islice(loader, args.sample_index, None))
    input, target, style = sample['input'].to(device), sample['target'].to(device), sample['style'].to(device)
    print(f'input shape: {input.shape}')
    print(f'target shape: {target.shape}')
    print(f'style shape: {style.shape}')

    # load model
    model = StyledVNet(args.style_size, sum(args.in_chan), sum(args.out_chan),
                       dropout_prob=args.dropout_prob,
                    scale_factor=args.scale_factor)
    state = torch.load(args.model_state_dict, map_location=device)
    load_model_state_dict(model, state['model'], strict=True)
    model.to(device)

    model.eval()

    pow_spec_stats = RunningStats()
    tf_error_stats = RunningStats()
    stocs_stats = RunningStats()
    with torch.no_grad():
        for i in tqdm(range(args.sample_size)):
            pred = model(input, style)

            if i == 0:
                print(f'pred_lin shape: {pred.shape}')
                _, target = narrow_cast(pred, target)

            pred_auto, target_auto, cross, wave_num = get_pow_spec(pred, target, box_length, axis)

            pow_spec_stats.push(pred_auto) # power spectrum of predicted linear field

            tf_error = np.sqrt(pred_auto / target_auto) - 1 # transfer function fractional error
            tf_error_stats.push(tf_error)

            stoc = 1 - (cross / np.sqrt(pred_auto * target_auto))**2 # stochastoicity
            stocs_stats.push(stoc)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

    def plot_power_spectrum_error_bar():
        pow_spec_mean = pow_spec_stats.mean()
        pow_spec_std = pow_spec_stats.standard_deviation()

        ax1.plot(
            wave_num,
            pow_spec_mean,
            label='Power Spectrum of Predicted Linear Field'
        )
        ax1.fill_between(
            wave_num,
            pow_spec_mean - pow_spec_std, # min,
            pow_spec_mean + pow_spec_std, # max,
            alpha=0.2,
            color='blue'
        )
        # plot power spectrum of target
        ax1.plot(
            wave_num, target_auto, label='Power Spectrum of True Linear Field'
        )

        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax1.set_xlabel('Wave Number k (log scale)')
        ax1.set_ylabel('Power Spectrum P(k) (log scale)')
        ax1.legend()
        ax1.set_title(f'Dropout Probability: {args.dropout_prob:.0%}')

    def plot_transfer_function_fractional_error():
        tf_error_mean = tf_error_stats.mean()
        tf_error_std = tf_error_stats.standard_deviation()

        ax2.semilogx(wave_num, tf_error_mean, color='red')
        ax2.fill_between(
            wave_num, tf_error_mean - tf_error_std, tf_error_mean + tf_error_std,
            alpha=0.2, color='red'
        )
        ax2.set_xlabel('k')
        ax2.set_ylabel('Transfer function fractional errors')
        ax2.set_title('Transfer function fractional errors vs k')

    def plot_stochasticity():
        stoc_mean = stocs_stats.mean()
        stoc_std = stocs_stats.standard_deviation()

        ax3.semilogx(wave_num, stoc_mean, color='green')
        ax3.fill_between(
            wave_num, stoc_mean - stoc_std, stoc_mean + stoc_std,
            alpha=0.2, color='green'
        )
        ax3.set_xlabel('k')
        ax3.set_ylabel('1 - (r(k)^2)')
        ax3.set_title('Stochasticty vs k')

    plot_power_spectrum_error_bar()
    plot_transfer_function_fractional_error()
    plot_stochasticity()

    fig.tight_layout()
    fig_path = os.path.join(
        'figs',
        f'LH{args.simulation_number}',
        args.experiment_name if args.experiment_name is not None else '',
        f'dropout-prob-{args.dropout_prob:.0}-sample-size-{args.sample_size}.png'
    )
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)
    print(f'Figure saved to {fig_path}')


if __name__ == '__main__':
    args = parse_args()
    plot_log_pow_spec_of_mean_lin(args)
