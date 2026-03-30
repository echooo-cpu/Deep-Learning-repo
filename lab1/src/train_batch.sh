#!/bin/bash

#python fnn.py --fnn_class large --activation relu --hidden_dims 128 64 32 --lr 0.001 --batch_size 32 --optimizer sgd --epochs 200 --patience 5 --info depth
cd ../src
echo "开始执行批量实验..."

EPOCHS=1000
OPTIMIZER="sgd"
BATCH_SIZES=32

python fnn.py \
    --fnn_class small \
    --activation relu \
    --hidden_dims 64 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info depth &
wait

python fnn.py \
    --fnn_class medium \
    --activation relu \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info depth &

wait

python fnn.py \
    --fnn_class large \
    --activation relu \
    --hidden_dims 128 64 32 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info depth &
wait

#==================================================

python fnn.py \
    --fnn_class medium \
    --activation relu \
    --hidden_dims 64 32 \
    --lr 0.1 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info lr &
wait

python fnn.py \
    --fnn_class medium \
    --activation relu \
    --hidden_dims 64 32 \
    --lr 0.01 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info lr &
wait

python fnn.py \
    --fnn_class medium \
    --activation relu \
    --hidden_dims 64 32 \
    --lr 0.0001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info lr &
wait

#==================================================

python fnn.py \
    --fnn_class medium \
    --activation tanh \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info act &
wait

python fnn.py \
    --fnn_class medium \
    --activation sigmoid \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info act &
wait    

python fnn.py \
    --fnn_class medium \
    --activation leaky_relu \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info act &
wait

python fnn.py \
    --fnn_class medium \
    --activation swish \
    --hidden_dims 64 32 \
    --lr 0.001 \
    --batch_size 32 \
    --batch_size ${BATCH_SIZES} \
    --optimizer ${OPTIMIZER} \
    --epochs ${EPOCHS} \
    --info act &
wait




echo "所有批量实验已全部完成！"