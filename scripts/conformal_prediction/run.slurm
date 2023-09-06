#!/bin/bash

#SBATCH --job-name=conformal_prediction
#SBATCH --output=%x-%j.out
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32:1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

hostname; pwd; date

pushd /ocean/projects/cis230021p/lianga/search/search
python main.py conformal_prediction \
    --crop 48 \
    --experiment-name "ConformalPrediction_Eval"
popd
