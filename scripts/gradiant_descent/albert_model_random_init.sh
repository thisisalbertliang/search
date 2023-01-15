#!/bin/bash

python search/main.py gradient_descent \
    --init-input-path "/user_data/ajliang/Linear/val/LH0045/4/dis.npy" \
    --style-path "/user_data/ajliang/Linear/val/LH0045/4/params.npy" \
    --target-output-path "/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy" \
    --load-forward-model-state "/home/ajliang/search/model_weights/albert_d2d_forward.pt" \
    --crop 32 \
    --lr 100 \
    --log-interval 100 \
    --experiment-name "albert-fwd-model-and-random-init"
