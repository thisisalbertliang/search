#!/bin/bash

# loop over dropout probabilities
for dropout_prob in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python jit_violin.py --dropout-prob $dropout_prob
done
