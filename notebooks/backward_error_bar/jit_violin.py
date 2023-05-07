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


def get_pow_spec(X: torch.Tensor, box_length: int, axis: int):
    # X shape: (batch, channel, x, y, z)
    X = X.squeeze(0).cpu().numpy()
    xpk = PKL.XPk([X[axis]], box_length, 0, (None, None))
    auto = xpk.Pk[:, 0, 0]
    wave_num = xpk.k3D
    return auto, wave_num


def plot_log_pow_spec_of_mean_lin(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    box_length: int = 1000 # Mpc/h
    axis: int = 0 # 0, 1, 2 for x, y, z respectively

    loader = get_data_loader(args)
    # get the nth sample from the loader
    sample = next(islice(loader, args.sample_index, None))
    input, target, style = sample['input'].to(device), sample['target'].to(device), sample['style'].to(device)
    print(f'input shape: {input.shape}')
    print(f'target shape: {target.shape}')
    print(f'style shape: {style.shape}')

    model = StyledVNet(args.style_size, sum(args.in_chan), sum(args.out_chan),
                       dropout_prob=args.dropout_prob,
                    scale_factor=args.scale_factor)
    state = torch.load(args.model_state_dict, map_location=device)
    load_model_state_dict(model, state['model'], strict=True)
    model.to(device)

    model.eval()

    stats = RunningStats()
    with torch.no_grad():
        for i in tqdm(range(args.sample_size)):
            pred_lin = model(input, style)

            if i == 0:
                print(f'pred_lin shape: {pred_lin.shape}')

            pred_lin_auto, wave_num = get_pow_spec(pred_lin, box_length, axis)
            stats.push(pred_lin_auto)

        mean = stats.mean()
        std = stats.standard_deviation()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # plot power spectrum error bars
    ax.plot(
        wave_num,
        mean,
        label='Power Spectrum of Predicted Linear Field'
    )
    ax.fill_between(
        wave_num,
        mean - std, # min,
        mean + std, # max,
        alpha=0.2,
        color='blue'
    )
    # plot power spectrum of target
    pred_lin, target = narrow_cast(pred_lin, target)
    pow_spec_target, wave_num = get_pow_spec(target, box_length, axis)
    ax.plot(
        wave_num, pow_spec_target, label='Power Spectrum of True Linear Field'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Wave Number k (log scale)')
    ax.set_ylabel('Power Spectrum P(k) (log scale)')
    ax.legend()
    ax.set_title(f'Dropout Probability: {args.dropout_prob:.0%}')

    fig.tight_layout()
    fig_path = f'figs/LH{args.simulation_number}/dropout-prob-{args.dropout_prob:.0}-sample-size-{args.sample_size}.png'
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    fig.savefig(fig_path)


if __name__ == '__main__':
    args = parse_args()
    plot_log_pow_spec_of_mean_lin(args)
