#!/bin/bash

python search/main.py uncertain \
    --load-forward-model-state "/home/ajliang/search/model_weights/paper_fwd_d2d_weights.pt" \
    --crop 64 \
    --device-ordinal 1 \
    --sample-size 10 \
    --dropout-prob 0.1 \
    --dataset-limit 10 \
    --experiment-name "FINAL-FINAL-in-vs-out-dist-uncertainty"
