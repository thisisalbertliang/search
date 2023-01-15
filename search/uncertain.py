import argparse
from tqdm import tqdm

import torch
from torchinfo import summary

from utils.common_utils import load_model, get_device
from utils.search_utils import initialize_style_input_target


def run_uncertain(args: argparse.Namespace):
    device = get_device()

    # # load saved input
    # input = torch.load(args.input_path).to(device)
    # initialize style in order to run forward pass
    # initialize target in order to compute loss
    style, input, target, _ = initialize_style_input_target(args, device)

    # load model
    model = load_model(args, device)
    model.eval()  # turns on dropout
    if args.verbose:
        summary(model, input_size=[
                input.shape, style.shape], depth=5, verbose=2, device=device)

    criterion = torch.nn.MSELoss()
    # run uncertainty estimation
    losses = torch.zeros(size=(args.sample_size,), device=device)
    for s in tqdm(range(args.sample_size)):
        with torch.no_grad():
            output = model(input, style)
            loss = criterion(output, target)
        losses[s] = loss
    mean = torch.mean(losses)
    variance = torch.var(losses, unbiased=False)

    print("******************** Uncertainty Estimation ********************")
    print("model:", args.load_model_state)
    print("input path:", args.input_path)
    print("dropout prob:", args.dropout_prob)
    print("sample size:", args.sample_size)
    print(f"mean: {mean.item()}")
    print(f"variance: {variance.item()}")
    print("***************************************************************")
