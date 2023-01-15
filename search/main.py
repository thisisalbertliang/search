import torch

from args import parse_args
from gradient_descent import run_gradient_descent
from uncertain import run_uncertain


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    if args.task == "gradient_descent":
        run_gradient_descent(args)
    elif args.task == "uncertain":
        run_uncertain(args)
    else:
        raise ValueError(f"Unknown subcommand: {args.subcommand}")
