set -x
CONFIGS=${@:1}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS=${CUDA_VISIBLE_DEVICES//,/}
GPUS=${#GPUS}
PORT=${PORT:-$RANDOM}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

torchrun --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --nproc_per_node=${GPUS} \
         --master_addr=$MASTER_ADDR \
         --master_port=$PORT \
         tools/trainval.py --launcher pytorch ${CONFIGS} \
