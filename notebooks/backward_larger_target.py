import sys

sys.path.append('/home/ajliang/search/search')
import argparse
import torch

from utils.common_utils import load_model
from map2map.map2map.data.fields import FieldDataset


import Pk_library as PKL
import matplotlib.pyplot as plt
import numpy as np

def plot_ps_trans_stoc(input, true_input):
    # Setup for plotting power spectrums
    axis = 2

    input = input.squeeze(0) # remove batch dimension
    input = input.cpu().detach().numpy() # convert to numpy
    true_input = true_input.squeeze(0) # remove batch dimension
    true_input = true_input.cpu().detach().numpy() # convert to numpy

    # compute the auto power spectrum of input and true_input and the their cross power spectrum
    xpk = PKL.XPk(
        (input[axis], true_input[axis]),
        BoxSize=1e3/512 * 128,
        axis=0,
        MAS=(None, None)
    )
    wave_num = xpk.k3D

    input_power_spec = xpk.Pk[:, 0, 0]
    true_input_power_spec = xpk.Pk[:, 0, 1]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))
    # plot power spectrum of input and true_input
    ax1.loglog(wave_num, input_power_spec,
            color="orange", label="Learned Linear Input")
    ax1.loglog(wave_num, true_input_power_spec,
            color="blue", label="True Linear Input")
    ax1.set_xlabel("k (log scale)")
    ax1.set_ylabel("P(k) (log scale)")
    ax1.legend()
    ax1.set_title("Power vs Wave Number")

    # plot transfer function fractional errors between input and true_input
    tf_error = np.sqrt(input_power_spec / true_input_power_spec) - 1 # fractional error in transfer function
    ax2.semilogx(wave_num, tf_error, color="red")
    ax2.set_xlabel('k')
    ax2.set_ylabel('Transfer function fractional errors')
    ax2.set_title('Transfer function fractional errors vs k')

    # plot stochasticity
    cross = xpk.XPk[:, 0][:, 0] # cross power spectrum
    stoc = 1 - (cross / np.sqrt(input_power_spec * true_input_power_spec))**2 # stochasticity
    ax3.semilogx(wave_num, stoc, color="green")
    ax3.set_xlabel('k')
    ax3.set_ylabel('1 - (r(k)^2)')
    ax3.set_title('Stochasticty vs k')

    fig.tight_layout()
    # plt.show()
    plt.savefig('/home/ajliang/search/notebooks/figures/backward_larger_target.png')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = argparse.Namespace(
        init_input_path="/user_data/ajliang/Linear/val/LH0045/4/dis.npy",
        style_path="/user_data/ajliang/Linear/val/LH0045/4/params.npy",
        target_output_path="/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy",
        load_backward_model_state="/home/ajliang/search/model_weights/backward_model.pt",
        verbose=False,
        dropout_prob=0.0,
        crop=32,
    )

    forward_dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        # in_norms=['cosmology.dis'],
        callback_at='/home/ajliang/search/search/map2map',
        crop=args.crop,
        in_pad=48,
    )

    crop_index = 0

    forward_data = forward_dataset[crop_index]
    input = forward_data['input']
    input = input.unsqueeze(0) # add batch dimension
    input = input.to(device)
    target = forward_data['target']
    target = target.unsqueeze(0) # add batch dimension
    target = target.to(device)
    style = forward_data['style']
    style = style.unsqueeze(0) # add batch dimension
    style = style.to(device)

    backward_dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.init_input_path],
        tgt_patterns=[args.target_output_path],
        # in_norms=['cosmology.dis'],
        callback_at='/home/ajliang/search/search/map2map',
        crop=args.crop,
        tgt_pad=96,
    )

    backward_data = backward_dataset[crop_index]

    backward_target = backward_data['target']
    backward_target = backward_target.unsqueeze(0) # add batch dimension
    backward_target = backward_target.to(device)
    backward_style = backward_data['style']
    backward_style = backward_style.unsqueeze(0) # add batch dimension
    backward_style = backward_style.to(device)

    args.style_size = backward_dataset.style_size
    args.in_chan = backward_dataset.in_chan
    args.out_chan = backward_dataset.tgt_chan
    backward_model = load_model(args, device=device, path_to_model_state=args.load_backward_model_state)

    with torch.no_grad():
        predicted_input = backward_model(backward_target, backward_style)

    print(f"input shape: {input.shape}")
    print(f"target shape: {target.shape}")
    print(f"backward_target shape: {backward_target.shape}")
    print(f"predicted_input shape: {predicted_input.shape}")

    plot_ps_trans_stoc(predicted_input, input)

    import pdb; pdb.set_trace()
