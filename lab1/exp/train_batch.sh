#!/bin/bash

# ==========================================
# FNN 批量实验脚本
# ==========================================

# 1. 定义你要遍历的超参数列表
# FNN_CLASSES=("small" "medium")
# ACTIVATIONS=("relu" "swish" "leaky_relu")
# LEARNING_RATES=(0.001 0.005)
# BATCH_SIZES=(32 64)
# OPTIMIZER="adam" # 优化器固定为 adam 作为示例


# if [ "$fnn_class" == "small" ]; then
#     HIDDEN_DIMS="64"
# elif [ "$fnn_class" == "medium" ]; then
#     HIDDEN_DIMS="64 32"
# else
#     HIDDEN_DIMS="128 64 32"
# fi
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