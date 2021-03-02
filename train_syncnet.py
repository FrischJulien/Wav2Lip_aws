#!/usr/bin/env python

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import json
import pickle
import sys
import traceback

import pandas as pd

from sklearn import tree

# These are the paths to where SageMaker mounts interesting things in your container.

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
output_path = os.path.join(prefix, 'output')
#model_path = os.path.join(prefix, 'model')
model_path="/workspace/Wav2Lip_aws/checkpoints"
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

syncnet_checkpoint_path="/workspace/Wav2Lip_aws/checkpoints/syncnet.path"

# This algorithm has a two channels of input data called 'trainingand' and 'syncnet_checkpoint'. Since we run in
# File mode, the input files are copied to the directory specified here.
training_path = os.path.join(input_path, 'training')
syncnet_checkpoint_path=os.path.join(input_path, 'syncnet_checkpoint/checkpoint_latest.pth')


# The function to execute the training.
def train():
    print('Starting the training.')
    try:
        os.system("python3 wav2lip_train.py --data_root {} --checkpoint_dir {} --syncnet_checkpoint_path {}".format(training_path,model_path,syncnet_checkpoint_path))
        print('Training complete.')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(os.path.join(output_path, 'failure'), 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        print('Exception during training: ' + str(e) + '\n' + trc, file=sys.stderr)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)

if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)