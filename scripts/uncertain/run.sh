#!/bin/bash

python search/main.py uncertain \
    --load-forward-model-state "/home/ajliang/search/search/map2map/checkpoints/fine-tune-paper-fwd-model-in-dist_2023-02-13-20-36-48/state_15.pt" \
    --crop 48 \
    --device-ordinal 0 \
    --sample-size 10 \
    --dropout-prob 0.3 \
    --dataset-limit 10 \
    --experiment-name "fine-tuned-in-vs-out-dist-uncertainty-GPU"
