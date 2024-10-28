
#!/bin/bash

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=000001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__1_Hz/"  # replace with your dataset name
result_folder="../1HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=100001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__2_Hz/"  # replace with your dataset name
result_folder="../2HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=200001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__5_Hz/"  # replace with your dataset name
result_folder="../5HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=300001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__10_Hz/"  # replace with your dataset name
result_folder="../10HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=400001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__20_Hz/"  # replace with your dataset name
result_folder="../20HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=400001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__40_Hz/"  # replace with your dataset name
result_folder="../40HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=400001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__50_Hz/"  # replace with your dataset name
result_folder="../50HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=400001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__100_Hz/"  # replace with your dataset name
result_folder="../100HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done

now=$(date +%Y_%m_%d_%H_%M_%S_%N)
ID=400001
dataset="/home/trijya/VersionControl/strym_notebooks/StateData__200_Hz/"  # replace with your dataset name
result_folder="../200HzFolderLog"
min_epoch=1000
num_epoch=500000
for w in {3..15}; do
    for n in 8 16 32 64; do
        
        for i in {1..16}; do
            python training_script_CNN_Sliding.py -d $dataset -r $result_folder -w $w -n $n -m $num_epoch -l $min_epoch -I $ID > training_log_${now}_run_${i}_ID_${ID}_w_${w}_n_${n}_num_epoch_${num_epoch}_min_epoch_${min_epoch}.txt
        done
        ((ID++))  # increment ID by 1
    done
done