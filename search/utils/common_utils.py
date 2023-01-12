import argparse
from time import strftime, gmtime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from map2map.map2map.models import StyledVNet
from map2map.map2map.utils import load_model_state_dict


def load_model(args: argparse.Namespace, device: torch.device) -> StyledVNet:
    model: StyledVNet = StyledVNet(
        style_size=args.style_size,
        in_chan=sum(args.in_chan),
        out_chan=sum(args.out_chan),
        dropout_prob=args.dropout_prob,
    )
    model.to(device)

    state = torch.load(args.load_model_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=True)
    
    print(
        f"Loaded model from {args.load_model_state}, which was trained for {state['epoch']} epochs.",
        flush=True
    )
    
    return model


def get_logger(args: argparse.Namespace):
    experiment_name = [args.experiment_name]
    experiment_name.append(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    experiment_name = '_'.join(experiment_name)

    logger = SummaryWriter(
        os.path.join("runs", "temp", experiment_name)
    )
    logger.add_text(
        tag="Arguments",
        text_string=str(vars(args)),
        global_step=0,
    )
    
    args.experiment_name = experiment_name
    return logger


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device = {device}.", flush=True)
    return device
