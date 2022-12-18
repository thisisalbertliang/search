import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient descent on Linear Displacement Field")
    
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
        "--load-state",
        default="/home/ajliang/search/search/map2map/checkpoints/DREW-FWD-MODEL/d2d_weights.pt",
        type=str,
    )
    parser.add_argument(
        "--lr",
        default=100,
        type=float,
    )
    parser.add_argument(
        "--seed",
        default=10620,
        type=int,
    )
    parser.add_argument(
        "--log-interval",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--save-interval",
        default=1000,
        type=int,
    )
    parser.add_argument(
        "--experiment-name",
        default="SEARCH-WITH-DREW-FWD-MODEL",
        type=str
    )
    
    return parser.parse_args()
