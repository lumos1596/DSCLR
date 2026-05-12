#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate dsclr
cd /home/luwa/Documents/DSCLR
python model/prepare_train_data.py
