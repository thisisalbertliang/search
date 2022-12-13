import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SGD on Linear Displacement Field")
    
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
        required=True,
        type=str,
    )
    parser.add_argument(
        "--lr",
        default=0.01,
        type=float,
    )
    parser.add_argument(
        "--epochs",
        default=1000,
        type=int,
    )
    
    return parser.parse_args()
