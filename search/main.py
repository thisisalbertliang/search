import argparse
from typing import Tuple
from time import strftime, gmtime
import torch
from torch.utils.tensorboard import SummaryWriter

from args import parse_args
from data.utils import initialize_style_input_target

from map2map.map2map.models import StyledVNet, lag2eul
from map2map.map2map.utils import load_model_state_dict
from map2map.map2map.utils import plt_slices, plt_power


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_logger(args: argparse.Namespace):
    experiment_name = [args.experiment_name]
    experiment_name.append(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    experiment_name = '_'.join(experiment_name)

    logger = SummaryWriter(f"runs/{experiment_name}")
    logger.add_text(
        tag="Arguments",
        text_string=str(vars(args)),
        global_step=0,
    )
    
    return logger


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


def get_model(args: argparse.Namespace, device: torch.device):
    model: StyledVNet = StyledVNet(
        style_size=args.style_size,
        in_chan=sum(args.in_chan),
        out_chan=sum(args.out_chan),
    )
    model.to(device)

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=True)
    
    print(
        f"Loaded model from {args.load_state}, which was trained for {state['epoch']} epochs.",
        flush=True
    )
    
    return model


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


def search(args: argparse.Namespace):
    logger = get_logger(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    style, input, target, true_input = initialize_style_input_target(args, device=device)
    input.requires_grad = True # turns on gradient tracking

    model = get_model(args, device=device)
    optimizer = torch.optim.SGD(
        params=[input],
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    iteration = 0
    while not torch.allclose(input, true_input, atol=1):
        output = model(input, style)

        lag_out, lag_tgt = output, target
        eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])

        _, _, lxe_loss, train_loss = compute_loss(
            lag_out, lag_tgt, eul_out, eul_tgt, criterion, logger, iteration
        )
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if iteration % args.log_interval == 0:
            print(f"Iteration: {iteration}, lxe loss: {lxe_loss}")
            plot_slices_and_power(
                logger, 
                input, true_input,
                lag_out, lag_tgt, 
                eul_out, eul_tgt, 
                iteration
            )
        iteration += 1


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    search(args)
