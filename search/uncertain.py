import argparse
from itertools import islice
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from map2map.map2map.data import FieldDataset
from map2map.map2map.models import StyledVNet
from map2map.map2map.utils import load_model_state_dict

from utils.common_utils import get_device, get_logger


def run_uncertain(args: argparse.Namespace):
    # get tensorboard logger
    logger = get_logger(args)

    plot_uncertainty(args, logger, in_distribution=True)
    plot_uncertainty(args, logger, in_distribution=False)


def plot_uncertainty(args: argparse.Namespace, logger: SummaryWriter, in_distribution: bool):
    distribution = f"{'in' if in_distribution else 'out'}_distribution"
    data_dir = f"/user_data/ajliang/ood_data/{distribution}/train"

    for simulation in os.listdir(data_dir):
        simulation_dir = os.path.join(data_dir, simulation)
        # load dataset
        dataset = FieldDataset(
            style_pattern=f"{simulation_dir}/params.npy",
            in_patterns=[f"{simulation_dir}/lin.npy"],
            tgt_patterns=[f"{simulation_dir}/nonlin.npy"],
            in_norms=['cosmology.dis'],
            tgt_norms=['cosmology.dis'],
            crop=args.crop,
            in_pad=48,
            scale_factor=1,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        # only ploy uncertainty for first N crops
        loader = islice(loader, args.dataset_limit)

        # load model
        model = StyledVNet(
            dataset.style_size,
            sum(dataset.in_chan),
            sum(dataset.tgt_chan),
            dropout_prob=args.dropout_prob,
            scale_factor=1.0,
        )
        device = get_device(args.device_ordinal)
        model.to(device)
        state = torch.load(args.load_forward_model_state, map_location=device)
        load_model_state_dict(model, state["model"], strict=True)

        # Compute uncertainty for each crop
        # plot results in tensorboard
        variances = torch.zeros(len(dataset), device=device)
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(tqdm(loader)):
                style, input, = data["style"].to(device), data["input"].to(device)
                variances[i] = estimate_uncertainty(model, style, input, args.sample_size)

                logger.add_scalar(
                    tag=f"{distribution}/{simulation}/uncertainty",
                    scalar_value=variances[i],
                    global_step=i,
                )
                print(f"{distribution} {simulation} crop {i}: Uncertainty = {variances[i]}", flush=True)

            mean_variance = torch.mean(variances)
            logger.add_text(
                tag=f"{distribution}/{simulation}/uncertainty",
                text_string=f"{distribution} {simulation}: Mean Uncertainty = {mean_variance}",
                global_step=i,
            )
            print(f"{distribution} {simulation}: Mean Uncertainty = {mean_variance}", flush=True)


def estimate_uncertainty(
    model: torch.nn.Module,
    style: torch.Tensor, input: torch.Tensor,
    sample_size: int,
) -> torch.Tensor:
    outputs = None
    for s in range(sample_size):
        out = model(input, style)
        if outputs is None:
            outputs = torch.zeros((sample_size, *out.shape), device=out.device)
        outputs[s] = out

    variance = torch.var(outputs, dim=0)
    return variance.sum()





    # # # load saved input
    # # input = torch.load(args.input_path).to(device)
    # # initialize style in order to run forward pass
    # # initialize target in order to compute loss
    # style, input, _, _ = initialize_style_input_target(args, device)

    # # load model
    # model = load_model(args, device)
    # model.eval()  # turns on dropout
    # if args.verbose:
    #     summary(model, input_size=[
    #             input.shape, style.shape], depth=5, verbose=2, device=device)

    # criterion = torch.nn.MSELoss()
    # # run uncertainty estimation
    # losses = torch.zeros(size=(args.sample_size,), device=device)
    # for s in tqdm(range(args.sample_size)):
    #     with torch.no_grad():
    #         output = model(input, style)
    #         loss = criterion(output, target)
    #     losses[s] = loss
    # mean = torch.mean(losses)
    # variance = torch.var(losses, unbiased=False)

    # print("******************** Uncertainty Estimation ********************")
    # print("model:", args.load_model_state)
    # print("input path:", args.input_path)
    # print("dropout prob:", args.dropout_prob)
    # print("sample size:", args.sample_size)
    # print(f"mean: {mean.item()}")
    # print(f"variance: {variance.item()}")
    # print("***************************************************************")
