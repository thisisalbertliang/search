#!/bin/bash

python search/main.py gradient_descent \
    --init-input-path "/user_data/ajliang/Linear/val/LH0045/4/dis.npy" \
    --style-path "/user_data/ajliang/Linear/val/LH0045/4/params.npy" \
    --target-output-path "/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy" \
    --load-forward-model-state "/home/ajliang/search/model_weights/paper_fwd_d2d_weights.pt" \
    --crop 32 \
    --lr 100 \
    --log-interval 25 \
    --save-interval 1000 \
    --experiment-name "GD_paper-fwd-model_random-init"
