from itertools import islice
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import numpy as np
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from map2map.map2map.data import FieldDataset
from map2map.map2map.models import StyledVNet
from map2map.map2map.utils import load_model_state_dict
from utils.common_utils import get_device, get_logger


def score(Y, Y_hat):
    return torch.mean((Y - Y_hat) ** 2)


def load_dataset(args, simulation_index):
    # Zero pad the simulation index to 4 digits
    simulation_index = str(simulation_index).zfill(4)
    # load dataset
    dataset = FieldDataset(
        style_pattern=f"{args.quijote_data_dir}/LH{simulation_index}/params.npy",
        in_patterns=[f"{args.quijote_data_dir}/LH{simulation_index}/nonlin.npy"],
        tgt_patterns=[f"{args.quijote_data_dir}/LH{simulation_index}/lin.npy"],
        in_norms=['cosmology.dis'],
        tgt_norms=['cosmology.dis'],
        crop=args.crop,
        in_pad=48,
        scale_factor=1,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )
    # use N crops as calibration data
    if args.dataset_limit:
        loader = islice(loader, args.dataset_limit)

    return dataset, loader


def load_model(args, dataset, device):
    # load model
    model = StyledVNet(
        dataset.style_size,
        sum(dataset.in_chan),
        sum(dataset.tgt_chan),
        dropout_prob=0.0,
        scale_factor=1.0,
    )
    model.to(device)
    state = torch.load(args.load_backward_model_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=True)

    return model


def run_conformal_prediction(args):
    # get tensorboard logger
    logger = get_logger(args)
    calibration_scores = compute_scores(args, logger, simulation_index=args.calibration_simulation_index)
    eval_conformal_prediction(args, logger, calibration_scores)


def compute_scores(args, logger: SummaryWriter, simulation_index: int) -> list[float]:
    simulation_index = str(simulation_index).zfill(4)
    scores_path = f"/ocean/projects/cis230021p/lianga/search/outputs/conformal_prediction/LH{simulation_index}_scores.pkl"

    def _log_scores(i):
        fig = plt.figure()
        plt.hist(scores, bins="auto")
        logger.add_figure(
            f"{args.experiment_name}/LH{simulation_index}/score_histogram",
            fig,
            global_step=i,
        )
        plt.clf()

    if os.path.exists(scores_path):
        with open(scores_path, "rb") as f:
            print(f"Loading scores from {scores_path}", flush=True)
            scores = pickle.load(f)
    else:
        # load dataset
        dataset, loader = load_dataset(args, simulation_index=simulation_index)
        # get device
        device = get_device()
        # load model
        model = load_model(args, dataset, device)

        # compute score for each crop
        # plot score distribution as histogram in tensorboard
        scores = []

        with torch.no_grad():
            model.eval()
            for i, data in tqdm(enumerate(loader)):
                style, input_, target = data["style"].to(device), data["input"].to(device), data["target"].to(device)
                output = model(input_, style)
                s = score(target, output)
                if args.verbose:
                    print(f"[LH{simulation_index}] [Iteration {i}] Score = {s}")
                s = s.cpu().numpy()
                scores.append(s)

                if args.log_interval and i % args.log_interval == 0:
                    _log_scores(i)

        # save scores to disk
        with open(scores_path, "wb") as f:
            print(f"Saving scores to {scores_path}", flush=True)
            pickle.dump(scores, f)

    _log_scores(len(scores))

    return scores


def eval_conformal_prediction(args, logger: SummaryWriter, calibration_scores: list[float]):
    calibration_scores.sort()

    def _get_quantile():
        n = len(calibration_scores)

        index = math.ceil((n + 1) * (1 - args.alpha)) - 1  # subtract 1 as Python indexing is 0-based
        return calibration_scores[index]

    quantile = _get_quantile()
    print(f"The {100 * (1 - args.alpha)}% quantile of the scores is {quantile}.")

    eval_scores = compute_scores(args, logger, simulation_index=args.evaluation_simulation_index)

    # line plot
    successes = []  # Keep track of whether each prediction was a success or not
    for i, s in enumerate(eval_scores):
        success = s <= quantile
        if args.verbose:
            print(
                f"[Evaluation iteration {i}]: \n"
                f"\t Prediction Set = {{ y : score(y) <= {quantile} }} \n"
                f"\t Actual Score = {s} \n"
                (f"\t PASS" if success else f"\t FAIL")
            )
        # line plot
        successes.append(success)

    def _plot_line(i):
        # Option 1: Line Plot
        plt.figure()
        plt.scatter(range(len(eval_scores)), eval_scores, c=['g' if x else 'r' for x in successes])
        plt.axhline(y=quantile, color='b', linestyle='--', label=f'Quantile: {quantile}')
        plt.legend()
        plt.title("Actual Scores and Quantile")
        plt.xlabel("Iteration")
        plt.ylabel("Score")

        # To log this figure to TensorBoard
        logger.add_figure("Actual Scores and Quantile", plt.gcf(), global_step=i)
        # Clear the figure to ensure no old data is plotted the next time
        plt.clf()

    def _plot_histogram(j):
        # Generate histogram data with automatic binning
        hist_data, bin_edges = np.histogram(eval_scores, bins='auto')

        # Initialize array for success/failure counts per bin
        success_failure_counts = np.zeros(len(bin_edges) - 1)

        # Count successes and failures per bin
        for s, success in zip(eval_scores, successes):
            for j in range(len(bin_edges) - 1):
                if bin_edges[j] <= s < bin_edges[j + 1]:
                    success_failure_counts[j] += 1 if success else -1
                    break

        # Plot histogram
        plt.figure()
        bars = plt.bar(
            range(len(success_failure_counts)),
            success_failure_counts,
            tick_label=[f"{round(bin_edges[j], 2)} - {round(bin_edges[j+1], 2)}" for j in range(len(bin_edges) - 1)],
            color=['g' if x >= 0 else 'r' for x in success_failure_counts]
        )
        plt.title("Success/Failure Histogram with Automatic Binning")
        plt.xlabel("Score Ranges")
        plt.ylabel("Net Successes/Failures")
        green_patch = plt.Rectangle((0,0),1,1,fc="g", edgecolor = 'none')
        red_patch = plt.Rectangle((0,0),1,1,fc='r', edgecolor = 'none')
        plt.legend([green_patch, red_patch], ['Success', 'Failure'], loc=2)

        # To log this figure to TensorBoard
        logger.add_figure("Success/Failure Histogram with Automatic Binning", plt.gcf(), global_step=j)
        # Clear the figure to ensure no old data is plotted the next time
        plt.clf()

    _plot_line(len(eval_scores))
    _plot_histogram(len(eval_scores))
