import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gradient descent on Linear Displacement Field"
    )

    subparsers = parser.add_subparsers(dest='task', required=True)
    _add_gradient_descent_args(subparsers)
    _add_uncertain_args(subparsers)

    return parser.parse_args()


def _add_common_args(parser: argparse.ArgumentParser):
    # common arguments
    parser.add_argument(
        "--load-forward-model-state",
        default="/home/ajliang/search/model_weights/paper_fwd_d2d_weights.pt",
        type=str,
    )
    parser.add_argument(
        "--seed",
        default=10620,
        type=int,
    )
    parser.add_argument(
        "--style-path",
        default="/user_data/ajliang/Linear/val/LH0045/4/params.npy",
    )
    parser.add_argument(
        "--init-input-path",
        default="/user_data/ajliang/Linear/val/LH0045/4/dis.npy",
        type=str,
    )
    parser.add_argument(
        "--target-output-path",
        default="/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy",
        type=str,
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--crop",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--device-ordinal",
        default=0,
        type=int
    )
    parser.add_argument(
        "--experiment-name",
        default="SEARCH-WITH-DREW-FWD-MODEL",
        type=str,
    )


def _add_gradient_descent_args(subparsers: argparse._SubParsersAction):
    """Add arguments for search subcommand"""
    parser_gradient_descent = subparsers.add_parser("gradient_descent")
    _add_common_args(parser_gradient_descent)

    parser_gradient_descent.add_argument(
        "--lr",
        default=100,
        type=float,
    )
    parser_gradient_descent.add_argument(
        "--log-interval",
        default=1000,
        type=int,
    )
    parser_gradient_descent.add_argument(
        "--save-interval",
        default=1000,
        type=int,
    )
    parser_gradient_descent.add_argument(
        "--load-backward-model-state",
        default=None,
        type=str,
    )
    # a hacky solution for compatibility with uncertainty estimation
    parser_gradient_descent.add_argument(
        "--dropout-prob",
        default=0.0,
        type=float,
    )


def _add_uncertain_args(subparsers: argparse._SubParsersAction):
    """Add arguments for uncertain subcommand"""
    parser_uncertain = subparsers.add_parser("uncertain")
    _add_common_args(parser_uncertain)

    parser_uncertain.add_argument(
        "--input-path",
        default="/home/ajliang/search/checkpoints/SEARCH-WITH-DREW-FWD-MODEL_2022-12-16-17-22-12/input_371000.pt",
        type=str,
    )
    parser_uncertain.add_argument(
        "--sample-size",
        default=10,
        type=int,
    )
    parser_uncertain.add_argument(
        "--dropout-prob",
        default=0.1,
        type=float,
    )
    parser_uncertain.add_argument(
        "--dataset-limit",
        default=100,
        type=int,
    )
