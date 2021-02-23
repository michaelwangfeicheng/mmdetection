#!/usr/bin/env bash

CONFIG=$1
NPROC_PER_NODE=1
GPUS=2
PORT=${PORT:-29500}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

echo GPUS=$GPUS
echo PORT=$PORT
echo PYTHONPATH=$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
