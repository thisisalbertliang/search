import argparse
from typing import Tuple
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.common_utils import load_model, get_logger
from utils.search_utils import initialize_style_input_target, plot_results

from map2map.map2map.models.lag2eul import lag2eul
from map2map.map2map.models import StyledVNet


def compute_and_log_loss(
    lag_out: torch.Tensor, lag_tgt: torch.Tensor,
    eul_out: torch.Tensor, eul_tgt: torch.Tensor,
    criterion, logger: SummaryWriter, iteration: int,
    log_folder: str = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lag_loss = criterion(lag_out, lag_tgt)
    eul_loss = criterion(eul_out, eul_tgt)
    lxe_loss = lag_loss * eul_loss

    log_folder = f"loss/{log_folder}" if log_folder else "loss"
    logger.add_scalar(f"{log_folder}/lag", lag_loss.item(), global_step=iteration)
    logger.add_scalar(f"{log_folder}/eul", eul_loss.item(), global_step=iteration)
    logger.add_scalar(f"{log_folder}/lxe", lxe_loss.item(), global_step=iteration)

    train_loss = torch.log((lag_loss ** 3) * eul_loss)
    return lag_loss, eul_loss, lxe_loss, train_loss


def plot_results_for_true_input(
    true_input: torch.Tensor,
    style: torch.Tensor,
    target: torch.Tensor,
    forward_model: StyledVNet,
    criterion,
    logger: SummaryWriter,
    save_dir: str
):
    with torch.no_grad():
        output = forward_model(true_input, style)
        lag_out, lag_tgt = output, target
        eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])

        compute_and_log_loss(
            lag_out=lag_out, lag_tgt=lag_tgt,
            eul_out=eul_out, eul_tgt=eul_tgt,
            criterion=criterion, logger=logger, iteration=0,
            log_folder="true_input",
        )

        plot_results(
            logger=logger, input=true_input, true_input=true_input,
            lag_out=lag_out, lag_tgt=lag_tgt, eul_out=eul_out, eul_tgt=eul_tgt,
            iteration=0, log_folder="true_input",
        )

        save_true_input_path = os.path.join(save_dir, "true_input.pt")
        torch.save(true_input, save_true_input_path)
        print(f"Saved true input to {save_true_input_path}", flush=True)


def run_gradient_descent(args: argparse.Namespace):
    logger = get_logger(args)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    style, input, target, true_input = initialize_style_input_target(args, device=device)
    input.requires_grad = True  # turns on gradient tracking

    forward_model = load_model(args, device=device, path_to_model_state=args.load_forward_model_state)
    optimizer = torch.optim.SGD(
        params=[input],
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    save_dir = os.path.join("checkpoints", args.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    # save style
    torch.save(style, os.path.join(save_dir, "style.pt"))

    plot_results_for_true_input(
        true_input=true_input, style=style, target=target,
        forward_model=forward_model, criterion=criterion,
        logger=logger, save_dir=save_dir
    )

    iteration = 0
    while not torch.allclose(input, true_input, atol=1):
        output = forward_model(input, style)

        lag_out, lag_tgt = output, target
        eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])
        _, _, lxe_loss, train_loss = compute_and_log_loss(
            lag_out, lag_tgt, eul_out, eul_tgt, criterion, logger, iteration
        )

        if iteration % args.log_interval == 0:
            print(f"After {iteration} steps: lxe loss = {lxe_loss}", flush=True)
            plot_results(
                logger,
                input, true_input,
                lag_out, lag_tgt,
                eul_out, eul_tgt,
                iteration
            )
        if iteration % args.save_interval == 0:
            save_path = os.path.join(save_dir, f"input_{iteration}.pt")
            torch.save(input, save_path)
            print(f"After {iteration} steps: saved input to {save_path}", flush=True)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        iteration += 1
