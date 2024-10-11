#!/bin/bash
conda deactivate
conda activate trajectory310

now=$(date +%Y_%m_%d_%H_%M_%S_%N)

python training_script_CNN_Sliding.py -r ACC_CNN_training > training_log_$dataset$now.txt