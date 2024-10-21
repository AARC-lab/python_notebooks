
#!/bin/bash

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=200
dataset="your_dataset_name"  # replace with your dataset name

for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -r ACC_CNN_training -w $w -n $n -m 100000 -I $ID > training_log_${dataset}_${now}_w${w}_n${n}_run${i}.txt
        done
        ((ID++))  # increment ID by 1
    done
done