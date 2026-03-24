#!/bin/bash

cd ../src

python fnn.py \
    --fnn_class medium \
    --activation relu \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size 32 \
    --optimizer sgd \
    --epochs 1000 \
    --info depth