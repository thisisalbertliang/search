import sys
import argparse
import torch

sys.path.append('/home/ajliang/search/search')

from utils.common_utils import load_model
from map2map.map2map.data.fields import FieldDataset

if __name__ == '__main__':
    """Load Dataset"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = argparse.Namespace(
        init_input_path="/user_data/ajliang/Linear/val/LH0045/4/dis.npy",
        style_path="/user_data/ajliang/Linear/val/LH0045/4/params.npy",
        target_output_path="/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy",
        load_backward_model_state="/home/ajliang/search/model_weights/backward_model.pt",
        verbose=False,
        dropout_prob=0.0,
    )

    dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        in_norms=['cosmology.dis'],
        callback_at='/home/ajliang/search/search/map2map',
        # crop=160,
        crop=40,
        in_pad=48,
    )

    args.style_size = dataset.style_size
    args.in_chan = dataset.in_chan
    args.out_chan = dataset.tgt_chan
    backward_model = load_model(args, device=device, path_to_model_state=args.load_backward_model_state)

    target = dataset[len(dataset) // 2]['target']
    target = target.unsqueeze(0) # add batch dimension
    target = target.to(device)

    input = dataset[len(dataset) // 2]['input']
    input = input.unsqueeze(0) # add batch dimension
    input = input.to(device)

    style: torch.Tensor = dataset[len(dataset) // 2]['style']
    style = style.unsqueeze(0) # add batch dimension
    style = style.to(device)

    import pdb; pdb.set_trace()

    backward_model(target, style)

    # style: torch.Tensor = dataset[len(dataset) // 2]['style']
    # style = style.unsqueeze(0) # add batch dimension

    # target: torch.Tensor = dataset[len(dataset) // 2]['target']
    # target = target.unsqueeze(0) # add batch dimension

    # backward_model = load_model(args, device=device, path_to_model_state=args.load_backward_model_state)
