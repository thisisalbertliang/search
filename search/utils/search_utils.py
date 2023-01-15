import argparse
from typing import Tuple
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

import Pk_library as PKL

from map2map.map2map.data.fields import FieldDataset
from map2map.map2map.utils import plt_slices, plt_power

from utils.common_utils import load_model


def initialize_style_input_target(
    args: argparse.Namespace, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        # in_norms=['cosmology.dis'],   # no normalization for now
        callback_at='/home/ajliang/search/search/map2map',
        crop=args.crop,
        in_pad=48,
    )

    crop_index = 0 # arbitrarily choose the first crop of the nonlinear field as the target

    args.style_size = forward_dataset.style_size
    args.in_chan = forward_dataset.in_chan
    args.out_chan = forward_dataset.tgt_chan

    if args.load_backward_model_state:
        init_input = _init_input_with_backward_model(args, device, crop_index)
    else:
        # arbitrarily choose a crop at the middle of the input linear field as the initial input
        random_crop_index = len(forward_dataset) // 2
        init_input = forward_dataset[random_crop_index]['input']
        init_input = init_input.unsqueeze(0) # add batch dimension
        init_input = init_input.to(device)

    target: torch.Tensor = forward_dataset[crop_index]['target']
    target = target.unsqueeze(0) # add batch dimension
    target = target.to(device)

    true_input: torch.Tensor = forward_dataset[crop_index]['input']
    true_input = true_input.unsqueeze(0) # add batch dimension
    true_input = true_input.to(device)

    style: torch.Tensor = forward_dataset[crop_index]['style']
    style = style.unsqueeze(0) # add batch dimension
    style = style.to(device)

    if args.verbose:
        print("************* Loaded Style, Initial Input, Target, True Input *************")
        print(f"style shape: {style.shape}")
        print(f"init_inpput shape: {init_input.shape}")
        print(f"target shape: {target.shape}")
        print(f"true input shape: {true_input.shape}")
        print("*******************************************************************")

    return style, init_input, target, true_input


def _init_input_with_backward_model(
    args: argparse.Namespace, device: torch.device, crop_index: int
) -> torch.Tensor:
    # load backward model
    backward_model = load_model(args, device=device, path_to_model_state=args.load_backward_model_state)
    # load the dataset for the backward model
    backward_dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        # in_norms=['cosmology.dis'],   # no normalization for now
        callback_at='/home/ajliang/search/search/map2map',
        crop=args.crop,
        tgt_pad=96, # pad target by 96 so that the backward model can output an input of shape (N, C, crop + 96, crop + 96, crop + 96)
    )
    backward_target = backward_dataset[crop_index]['target']
    backward_target = backward_target.unsqueeze(0) # add batch dimension
    backward_target = backward_target.to(device) # move to device

    style = backward_dataset[crop_index]['style']
    style = style.unsqueeze(0) # add batch dimension
    style = style.to(device) # move to device
    with torch.no_grad():
        backward_input = backward_model(backward_target, style)

    assert backward_input.shape == torch.Size([1, 3, args.crop + 96, args.crop + 96, args.crop + 96])
    return backward_input


def plot_results(
    logger: SummaryWriter,
    input: torch.Tensor, true_input: torch.Tensor,
    lag_out: torch.Tensor, lag_tgt: torch.Tensor,
    eul_out: torch.Tensor, eul_tgt: torch.Tensor,
    iteration: int, log_folder: str = None,
):
    log_folder = f"fig/{log_folder}" if log_folder else "fig"

    # plot slices of input, true_input, lag_out, lag_tgt, eul_out, eul_tgt
    fig = plt_slices(
        input[-1], true_input[-1],
        lag_out[-1], lag_tgt[-1], lag_out[-1] - lag_tgt[-1],
        eul_out[-1], eul_tgt[-1], eul_out[-1] - eul_tgt[-1],
        title=['in', 'true_in',
               'lag_out', 'lag_tgt', 'lag_out - lag_tgt',
               'eul_out', 'eul_tgt', 'eul_out - eul_tgt'],
    )
    logger.add_figure(
        f"{log_folder}/slices/iteration_{iteration}", fig, global_step=iteration)
    fig.clf()

    # plot power spectrum of input, lag_out, lag_tgt
    fig = plt_power(input, lag_out, lag_tgt, label=[
                    'in', 'lag_out', 'lag_tgt'])
    logger.add_figure(
        f'{log_folder}/power/lag/iteration_{iteration}', fig, global_step=iteration)
    fig.clf()

    # plot power spectrum of input, eul_out, eul_tgt
    fig = plt_power(1.0,
                    dis=[input, lag_out, lag_tgt],
                    label=['in', 'eul_out', 'eul_tgt'],
                    )
    logger.add_figure(
        f'{log_folder}/power/eul/iteration_{iteration}', fig, global_step=iteration)
    fig.clf()

    # Setup for plotting power spectrums
    axis = 2

    input = input.squeeze(0) # remove batch dimension
    input = input.cpu().detach().numpy() # convert to numpy
    true_input = true_input.squeeze(0) # remove batch dimension
    true_input = true_input.cpu().detach().numpy() # convert to numpy

    # compute the auto power spectrum of input and true_input and the their cross power spectrum
    xpk = PKL.XPk(
        (input[axis], true_input[axis]),
        BoxSize=1e3/512 * 128,
        axis=0,
        MAS=(None, None)
    )
    wave_num = xpk.k3D

    input_power_spec = xpk.Pk[:, 0, 0]
    true_input_power_spec = xpk.Pk[:, 0, 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    # plot power spectrum of input and true_input
    ax1.loglog(wave_num, input_power_spec,
            color="orange", label="Learned Linear Input")
    ax1.loglog(wave_num, true_input_power_spec,
            color="blue", label="True Linear Input")
    ax1.set_xlabel("k (log scale)")
    ax1.set_ylabel("P(k) (log scale)")
    ax1.legend()
    ax1.set_title("Power vs Wave Number")

    # plot transfer function fractional errors between input and true_input
    tf_error = np.sqrt(input_power_spec / true_input_power_spec) - 1 # fractional error in transfer function
    ax2.semilogx(wave_num, tf_error, color="red")
    ax2.set_xlabel('k')
    ax2.set_ylabel('Transfer function fractional errors')
    ax2.set_title('Transfer function fractional errors vs k')

    # plot stochasticity
    cross = xpk.XPk[:, 0][:, 0] # cross power spectrum
    stoc = 1 - (cross / np.sqrt(input_power_spec * true_input_power_spec))**2 # stochasticity
    ax3.semilogx(wave_num, stoc, color="green")
    ax3.set_xlabel('k')
    ax3.set_ylabel('1 - (r(k)^2)')
    ax3.set_title('Stochasticty vs k')

    fig.tight_layout()
    logger.add_figure(
        f"{log_folder}/lin_pow_spec_&_transfer_&_stoc/iteration_{iteration}", fig, global_step=iteration)
    fig.clf()
