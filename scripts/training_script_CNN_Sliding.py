#!/usr/bin/env python
# coding: utf-8

# Author : Rahul Bhadani
# Initial Date: Oct 6, 2024
# About: Python Script for Complete Pipeline of training CNN with Sliding Window
# License: MIT License

#   Permission is hereby granted, free of charge, to any person obtaining
#   a copy of this software and associated documentation files
#   (the "Software"), to deal in the Software without restriction, including
#   without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject
#   to the following conditions:

#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.

#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
#   ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
#   TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#   PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
#   SHALL THE AUTHORS, COPYRIGHT HOLDERS OR ARIZONA BOARD OF REGENTS
#   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#   AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#   OR OTHER DEALINGS IN THE SOFTWARE.

__author__ = 'Rahul Bhadani'
__email__  = 'rahul.bhadani@uah.edu'

import datetime
import os
import random
import yaml


import sys, getopt
import time
from trajectorylib.utility.functions import dict_reverser
from trajectorylib.utility.functions import configure_logworker

import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from trajectorylib.ml.model import CNN
from trajectorylib.ml.data_process import SlidingWindowDataProcessor
from trajectorylib.ml.trainer import SlidingWindowTrainer

def main(argv):
    
    description = "CNN Training With Sldiing Window"
    
    # configuration for this training -- default values
    data_folder = '/home/trijya/Dataset/DrivingData/'
    window_size = 6
    train_test_split = 0.8
    result_folder = "acc_cnn_training"
    dropout_rate = 0.2
    n_filters = 32
    n_fc_unit = 64
    learning_rate = 0.001
    patience = 10
    min_lr=1e-6
    num_epochs= 40
    seed = 42
    ID = 100
    
    config = {}
    
    try:
        opts, args = getopt.getopt(argv,"hd:w:s:r:D:n:f:c:l:p:m:e:s:I:",["data_folder=","window_size=",
                "train_test_split=", "result_folder=", "dropout_rate=", "n_filters=", "n_fc_unit=", "learning_rate=", "patience=", "min_lr=", "num_epochs=", "seed=", "id="])
        if len(opts) == 0:
            print('Check options by typing:\n{} -h'.format(__file__))
            sys.exit()

    except getopt.GetoptError:
        print('Check options by typing:\n{} -h'.format(__file__))
        sys.exit(2)
        
    print("OPTS: {}".format(opts))
    for opt, arg in opts:
        if(opt == '-h'):
            print('\n{} [OPTIONS]'.format(__file__))
            print('\t -h, --help\t\t Get help')
            print('\t -d, --data_folder\t\t Enter address of the folder where data is located.')
            print('\t -w, --window_size\t\t Window Size for The Sliding Window')
            print('\t -s, --train_test_split\t\t Ratio of Training and Test Split')
            print('\t -r, --result_folder\t\t Folder where to store result and metrics')
            print('\t -D, --dropout_rate\t\t Dropout Rate for the Model')
            print('\t -n, --n_filters\t\t Number of Filters for Convolutional Layers')
            print('\t -f, --n_fc_unit\t\t Number of Units in Fully Connected Layers')
            print('\t -c, --learning_rate\t\t Learning Rate for the Model')
            print('\t -l, --patience\t\t\t Patience for Early Stopping')
            print('\t -p, --min_lr\t\t\t Minimum Learning Rate for the Model')
            print('\t -m, --num_epochs\t\t Number of Epochs to Train the Model')
            print('\t -e, --seed\t\t\t Random Seed for Reproducibility')
            print('\t -I, --id\t\t\t Unique Identified for this execution')
            sys.exit()
        elif opt in ("-d", "--data_folder"):
            data_folder = arg
            config['data_folder'] = data_folder
            print("Data Folder: {}".format(data_folder))

        elif opt in ("-w", "--window_size"):
            window_size = int(arg)
            config['window_size'] = window_size
            print("Window Size: {}".format(window_size))

        elif opt in ("-s", "--train_test_split"):
            train_test_split = float(arg)
            config['train_test_split'] = train_test_split
            print("Train Test Split: {}".format(train_test_split))

        elif opt in ("-r", "--result_folder"):
            result_folder = arg
            config['result_folder'] = result_folder
            print("Result Folder: {}".format(result_folder))

        elif opt in ("-D", "--dropout_rate"):
            dropout_rate = float(arg)
            config['dropout_rate'] = dropout_rate
            print("Dropout Rate: {}".format(dropout_rate))

        elif opt in ("-n", "--n_filters"):
            n_filters = int(arg)
            config['n_filters'] = n_filters
            print("Number of Filters: {}".format(n_filters))

        elif opt in ("-f", "--n_fc_unit"):
            n_fc_unit = int(arg)
            config['n_fc_unit'] = n_fc_unit
            print("Number of FC Units: {}".format(n_fc_unit))

        elif opt in ("-c", "--learning_rate"):
            learning_rate = float(arg)
            config['learning_rate'] = learning_rate
            print("Learning Rate: {}".format(learning_rate))

        elif opt in ("-l", "--patience"):
            patience = int(arg)
            config['patience'] = patience
            print("Patience: {}".format(patience))

        elif opt in ("-p", "--min_lr"):
            min_lr = float(arg)
            config['min_lr'] = min_lr
            print("Minimum Learning Rate: {}".format(min_lr))

        elif opt in ("-m", "--num_epochs"):
            num_epochs = int(arg)
            config['num_epochs'] = num_epochs
            print("Number of Epochs: {}".format(num_epochs))

        elif opt in ("-e", "--seed"):
            seed = int(arg)
            config['seed'] = seed
            print("Random Seed: {}".format(seed))
        
        elif opt in ("-I", "--id"):
            ID = int(arg)
            config['ID'] = ID
            print("Random Seed: {}".format(seed))

    random.seed(seed)
    
    
    dt_object = datetime.datetime.fromtimestamp(time.time())
    date_string = dt_object.strftime('%Y_%m_%d__%H_%M_%S_%f')
    
    date_string = "ID_{}_".format(ID) + date_string
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        
    result_directory = os.path.join(result_folder, date_string)
        
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    
    logfile = result_directory + "/" + description.strip().replace(" ","_") + "_sliding_window_trainer_log_" + date_string + ".log"

    configfile = result_directory + "/config_" + description.strip().replace(" ","_") + "_sliding_window_trainer_log_" + date_string + ".yaml"
    with open(configfile, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=4, sort_keys=True)
    
    print(logfile)
    # Instantiate the CNN model
    _LOGGER = configure_logworker(logfile = logfile, name = "root")
    
    # Check for GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get a list of all CSV files in the data folder with full paths
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    # Select 80% of the files randomly
    training_set = random.sample(csv_files, int(len(csv_files) * train_test_split))

    _LOGGER.info("Training set:")
    _LOGGER.info(training_set)

    _LOGGER.info("Number of files in training set:{}".format(len(training_set)))
    _LOGGER.info("Total number of CSV files:{}".format(len(csv_files)))
    
    # Get the remaining files for test set
    test_set = [f for f in csv_files if f not in training_set]
    _LOGGER.info("Test set: {}".format(test_set))
    
    dataprocessor = SlidingWindowDataProcessor(training_set, window_size=window_size, logfile=logfile)
    dataprocessor.process_file_list()
    
    # Create an instance of the CNN model
    model = CNN(window_size=window_size, dropout_rate=dropout_rate, n_features=dataprocessor.all_X[0].shape[1], n_filters=n_filters, n_fc_unit = n_fc_unit, logfile=logfile)
    
    trainer = SlidingWindowTrainer(model = model, device = device, data_processor=dataprocessor, logfile=logfile, result_directory = result_directory)
    
    # Define Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #Define Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor=0.1, patience=patience, min_lr=min_lr)
    trainer.train(criterion=criterion, optimizer=optimizer, scheduler=scheduler, num_epochs=num_epochs)
    _LOGGER.info("Training finished")
    
    
    
    _LOGGER.info("Testing phase started.")
    testdata_processor = SlidingWindowDataProcessor(test_set, window_size=window_size, logfile=logfile)
    testdata_processor.process_file_list()
    trainer.predict(testdata_processor)
    
    onnx_accuracy = trainer.predict_onnx(testdata_processor)
    
    # plot the metrics
    metric_df = pd.read_csv(trainer.metricfile)
    
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 24  # Double the default size


    # Set the figure size and resolution
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), dpi=600)

    # Set LaTeX interpreter and font size
    
    # Plot the training loss and validation loss
    axs[0].plot(metric_df['Epoch'], metric_df['Train Loss'], label='Training Loss', color='#2d6a4f', linewidth=2)
    axs[0].plot(metric_df['Epoch'], metric_df['Validation Loss'], label='Validation Loss', color='#a4161a', linewidth=2)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Training and Validation Loss')
    axs[0].legend(loc='upper right', framealpha=0.5)

    # Plot the training accuracy and validation accuracy
    axs[1].plot(metric_df['Epoch'], metric_df['Train Accuracy'], label='Training Accuracy', color='#2d6a4f', linewidth=2)
    axs[1].plot(metric_df['Epoch'], metric_df['Validation Accuracy'], label='Validation Accuracy', color='#a4161a', linewidth=2)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].legend(loc='upper right', framealpha=0.5)

    # Add grid with light gray color
    
    

    for ax in axs:
        # Turn on the minor grid
        ax.minorticks_on()
        ax.grid(which='major', linestyle='--', alpha=0.2, color='#cccccc')
        ax.grid(which='minor', linestyle='--', alpha=0.2, color='#cccccc')

    # Layout so plots do not overlap
    fig.tight_layout()

    # Save the plot to a file
    plt.savefig(result_directory + '/training_metrics_plot.png', bbox_inches='tight')
    plt.savefig(result_directory + '/training_metrics_plot.pdf', bbox_inches='tight')
    
    
    
if __name__ == "__main__":
   main(sys.argv[1:])
