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

import os
import random

import sys, getopt
from functions import dict_reverser

import pandas as pd
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from trajectorylib.ml.model import CNN
from trajectorylib.ml.data_process import SlidingWindowDataProcessor
from trajectorylib.ml.trainer import SlidingWindowTrainer

def main(argv):
    data_folder = '/home/trijya/Dataset/DrivingData/'
    window_size = 6
    train_test_split = 0.80
    result_folder ='./'
    
    try:
        opts, args = getopt.getopt(argv,"hd:w:s:r:",["data_folder=","window_size=","train_test_split=", "result_folder="])
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
            print('\t -d, --data_folder\t\t\t Enter address of the folder where data is located.')
            print('\t -w, --window_size\t Window Size for The Sliding Window')
            print('\t -s, --train_test_split\t\t Ratio of Training and Test Split')
            print('\t -r, --result_folder\t\t Folder where to store result and metrics')
            sys.exit()


if __name__ == "__main__":
   main(sys.argv[1:])
