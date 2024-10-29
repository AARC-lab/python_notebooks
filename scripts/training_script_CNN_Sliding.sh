#!/bin/bash

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
min_epoch=1000
num_epoch=500000
learning_rate=0.0001

hz_values=(1 2 5 10 20 40 50 100 200)
base_id=100000
base_dataset="/home/trijya/VersionControl/strym_notebooks/StateData__"
base_result_folder="../"

for i in "${!hz_values[@]}"; do
  hz=${hz_values[$i]}
  dataset="${base_dataset}${hz}_Hz/"
  result_folder="${base_result_folder}${hz}HzFolderLog"
  ID=$((base_id * (i + 1)))

  for w in {3..15}; do
    for n in 8 16 32 64; do
      for j in {1..16}; do
        python training_script_CNN_Sliding.py \
          -d "$dataset" \
          -r "$result_folder" \
          -w "$w" \
          -n "$n" \
          -m "$num_epoch" \
          -l "$min_epoch" \
          -I "$ID" \
          -c "$learning_rate" \
          > "training_log_${now}_run_${j}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}_learning_rate_${learning_rate}.txt"
      done
      ((ID++))
    done
  done
done