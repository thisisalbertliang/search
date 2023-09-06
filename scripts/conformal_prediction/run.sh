#!/bin/bash


for calibration_simulation_index in 0 100 200 300 400 500 600 700 800 900
do
    for evaluation_simulation_index in 0 100 200 300 400 500 600 700 800 900
    do
        pushd /ocean/projects/cis230021p/lianga/search/search
        python main.py conformal_prediction \
            --crop 48 \
            --experiment-name "ConformalPrediction_CalibrationLH${calibration_simulation_index}_EvalLH${evaluation_simulation_index}" \
            --calibration-simulation-index ${calibration_simulation_index} \
            --evaluation-simulation-index ${evaluation_simulation_index}
        popd
    done
done



# pushd /ocean/projects/cis230021p/lianga/search/search
# python main.py conformal_prediction \
#     --crop 48 \
#     --experiment-name "ConformalPrediction_CalibrationLH300_EvalLH300" \
#     --calibration-simulation-index 300 \
#     --evaluation-simulation-index 300 \
#     --dataset-limit 100
# popd
