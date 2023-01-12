import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient descent on Linear Displacement Field")
    
    # common arguments
    parser.add_argument(
        "--load-model-state",
        default="/home/ajliang/search/search/map2map/checkpoints/DREW-FWD-MODEL/d2d_weights.pt",
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
    
    subparsers = parser.add_subparsers(dest='task', required=True)
    _add_search_args(subparsers)
    _add_uncertain_args(subparsers)
    
    return parser.parse_args()


def _add_search_args(subparsers: argparse._SubParsersAction):
    """Add arguments for search subcommand"""
    parser_search = subparsers.add_parser("search")
    
    parser_search.add_argument(
        "--lr",
        default=100,
        type=float,
    )
    parser_search.add_argument(
        "--log-interval",
        default=1000,
        type=int,
    )
    parser_search.add_argument(
        "--save-interval",
        default=1000,
        type=int,
    )
    parser_search.add_argument(
        "--experiment-name",
        default="SEARCH-WITH-DREW-FWD-MODEL",
        type=str
    )


def _add_uncertain_args(subparsers: argparse._SubParsersAction):
    """Add arguments for uncertain subcommand"""
    parser_uncertain = subparsers.add_parser("uncertain")
    
    parser_uncertain.add_argument(
        "--input-path",
        default="/home/ajliang/search/checkpoints/SEARCH-WITH-DREW-FWD-MODEL_2022-12-16-17-22-12/input_371000.pt",
        type=str,
    )
    parser_uncertain.add_argument(
        "--sample-size",
        default=100,
        type=int,
    )
    parser_uncertain.add_argument(
        "--dropout-prob",
        default=0.5,
        type=float,
    )
