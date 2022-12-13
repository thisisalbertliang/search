import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

from args import parse_args
from data.utils import initialize_style_input_target

from map2map.map2map.models import StyledVNet, lag2eul
from map2map.map2map.utils import load_model_state_dict


def get_logger(args: argparse.Namespace):
    logger = SummaryWriter("runs")
    logger.add_text(
        tag="Arguments",
        text_string=str(vars(args)),
        global_step=0,
    )
    
    return logger


def get_model(args: argparse.Namespace, device: torch.device):
    model: StyledVNet = StyledVNet(
        style_size=args.style_size,
        in_chan=sum(args.in_chan),
        out_chan=sum(args.out_chan),
    )
    model.to(device)

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=True)
    
    print(
        f"Loaded model from {args.load_state}, which was trained for {state['epoch']} epochs.",
        flush=True
    )
    
    return model


def compute_loss(output: torch.Tensor, target: torch.Tensor, criterion):
    lag_out, lag_tgt = output, target
    eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])
    
    lag_loss = criterion(lag_out, lag_tgt)
    eul_loss = criterion(eul_out, eul_tgt)
    loss = lag_loss * eul_loss
    
    return loss


def search(args: argparse.Namespace):
    logger = get_logger(args)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    style, input, target = initialize_style_input_target(args, device=device)
    input.requires_grad = True # turns on gradient tracking

    model = get_model(args, device=device)
    optimizer = torch.optim.SGD(
        params=[input],
        lr=args.lr,
    )
    criterion = torch.nn.MSELoss()

    for epoch in tqdm(range(args.epochs)):
        output = model(input, style)

        loss = compute_loss(output, target, criterion)
        optimizer.zero_grad()
        torch.log(loss).backward() # actual loss is log(loss)
        optimizer.step()
        
        print(f"Epoch {epoch}: {loss.item()}")


if __name__ == "__main__":
    args = parse_args()

    search(args)
