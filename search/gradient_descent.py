import argparse
from typing import Tuple
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common_utils import load_model, get_logger
from utils.search_utils import initialize_style_input_target

from map2map.map2map.models import lag2eul
from map2map.map2map.utils import plt_slices, plt_power


def plot_slices_and_power(
    logger: SummaryWriter,
    input: torch.Tensor, true_input: torch.Tensor,
    lag_out: torch.Tensor, lag_tgt: torch.Tensor,
    eul_out: torch.Tensor, eul_tgt: torch.Tensor,
    iteration: int
):
    # plot slices of input, true_input, lag_out, lag_tgt, eul_out, eul_tgt
    fig = plt_slices(
        input[-1], true_input[-1],
        lag_out[-1], lag_tgt[-1], lag_out[-1] - lag_tgt[-1],
        eul_out[-1], eul_tgt[-1], eul_out[-1] - eul_tgt[-1],
        title=['in', 'true_in', 
               'lag_out', 'lag_tgt', 'lag_out - lag_tgt',
               'eul_out', 'eul_tgt', 'eul_out - eul_tgt'],
    )
    logger.add_figure(f"fig/slices/iteration_{iteration}", fig, global_step=iteration)
    fig.clf()
    
    # plot power spectrum of input, lag_out, lag_tgt
    fig = plt_power(input, lag_out, lag_tgt, label=['in', 'lag_out', 'lag_tgt'])
    logger.add_figure(f'fig/power/lag/iteration_{iteration}', fig, global_step=iteration)
    fig.clf()
    
    # plot power spectrum of input, eul_out, eul_tgt
    fig = plt_power(1.0,
        dis=[input, lag_out, lag_tgt],
        label=['in', 'eul_out', 'eul_tgt'],
    )
    logger.add_figure(f'fig/power/eul/iteration_{iteration}', fig, global_step=iteration)
    fig.clf()


def compute_loss(
    lag_out: torch.Tensor, lag_tgt: torch.Tensor,
    eul_out: torch.Tensor, eul_tgt: torch.Tensor,
    criterion, logger: SummaryWriter, iteration: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:    
    lag_loss = criterion(lag_out, lag_tgt)
    eul_loss = criterion(eul_out, eul_tgt)
    lxe_loss = lag_loss * eul_loss
    
    logger.add_scalar("loss/lag", lag_loss.item(), global_step=iteration)
    logger.add_scalar("loss/eul", eul_loss.item(), global_step=iteration)
    logger.add_scalar("loss/lxe", lxe_loss.item(), global_step=iteration)
    
    train_loss = torch.log((lag_loss ** 3) * eul_loss)
    return lag_loss, eul_loss, lxe_loss, train_loss


def run_search(args: argparse.Namespace):
    logger = get_logger(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    style, input, target, true_input = initialize_style_input_target(args, device=device)
    input.requires_grad = True # turns on gradient tracking

    model = load_model(args, device=device)
    optimizer = torch.optim.SGD(
        params=[input],
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    save_dir = os.path.join("checkpoints", args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    # save style
    torch.save(style, os.path.join(save_dir, "style.pt"))

    iteration = 0
    while not torch.allclose(input, true_input, atol=1):
        output = model(input, style)

        lag_out, lag_tgt = output, target
        eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])
        _, _, lxe_loss, train_loss = compute_loss(
            lag_out, lag_tgt, eul_out, eul_tgt, criterion, logger, iteration
        )

        if iteration % args.log_interval == 0:
            print(f"After {iteration} steps: lxe loss = {lxe_loss}")
            plot_slices_and_power(
                logger, 
                input, true_input,
                lag_out, lag_tgt, 
                eul_out, eul_tgt, 
                iteration
            )
        if iteration % args.save_interval == 0:
            save_path = os.path.join(save_dir, f"input_{iteration}.pt")
            torch.save(input, save_path)
            print(f"After {iteration} steps: saved input to {save_path}")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        iteration += 1
