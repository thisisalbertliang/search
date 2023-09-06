#!/bin/bash

# loop over dropout probabilities
for dropout_prob in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python jit_violin.py --dropout-prob $dropout_prob --experiment-name pseudo
done

# python jit_violin.py --dropout-prob 0.1 --model-state-dict /ocean/projects/cis230021p/lianga/search/search/map2map/checkpoints/train-BACKWARD_2023-05-07-01-00-01/state_17.pt

# python jit_violin.py --dropout-prob 0.3 --model-state-dict /ocean/projects/cis230021p/lianga/search/search/map2map/checkpoints/train-BACKWARD_2023-05-06-18-21-04/state_19.pt

# python jit_violin.py --dropout-prob 0.5 --model-state-dict /ocean/projects/cis230021p/lianga/search/search/map2map/checkpoints/train-BACKWARD_2023-05-06-18-21-02/state_17.pt
