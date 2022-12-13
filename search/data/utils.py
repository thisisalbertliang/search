import argparse
import torch

from map2map.map2map.data.fields import FieldDataset


def initialize_style_input_target(args: argparse.Namespace, device: torch.device):
    dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        in_norms=['cosmology.dis'],
        tgt_norms=None,
        callback_at='/home/ajliang/search/search/map2map',
        augment=True,
        aug_shift=16,
        aug_add=None,
        aug_mul=None,
        crop=40,
        crop_start=None,
        crop_stop=None,
        crop_step=40,
        in_pad=48,
        tgt_pad=0,
        scale_factor=1
    )
    
    args.style_size = dataset.style_size
    args.in_chan = dataset.in_chan
    args.out_chan = dataset.tgt_chan
    
    data = dataset[0]
    style: torch.Tensor = data['style']
    input: torch.Tensor = data['input']
    target: torch.Tensor = data['target']
    # Add batch dimension
    style = style.unsqueeze(0)
    input = input.unsqueeze(0)
    target = target.unsqueeze(0)
    
    return style.to(device), input.to(device), target.to(device)
