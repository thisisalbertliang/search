#!/bin/bash

for dropout_prob in $(seq 0.1 0.1 0.9)
do
    python search/main.py uncertain \
        --dropout-prob $dropout_prob \
        --input-path /user_data/ajliang/Linear/val/LH0045/4/dis.npy \
        --load-model-state /home/ajliang/search/search/map2map/checkpoints/ALBERT-FWD-MODEL_2022-12-14-21-41-44/state_145.pt
done