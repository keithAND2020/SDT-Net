set -x

CONFIG=$1
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-6}
PY_ARGS=${@:2}
# NODES=${NODES:-1}
NODES=${NODES:-2}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=$RANDOM srun -A ailab \
    -p vip_gpu_ailab_low \
    -x g0824  \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --nodes=${NODES} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --qos=gpugpu \
    python -u tools/trainval.py ${CONFIG}   --launcher="slurm" ${PY_ARGS}
    # GPUS=4 GPUS_PER_NODE=2 bash tools/slurm_trainval_beijing.sh configs/models/cornet2.py --log_dir /ailab/user/zhuangguohang/ai4stronomy/zhuangguohang/ai4astronomy/sd_docs/PJ-Astronomy-spacedebris/log/test
    # GPUS=1 GPUS_PER_NODE=1 bash tools/slurm_trainval_beijing.sh configs/models/cornet2.py --log_dir /ailab/user/zhuangguohang/ai4stronomy/zhuangguohang/ai4astronomy/sd_docs/PJ-Astronomy-spacedebris/log/corner_ab/ -e --resume /ailab/user/zhuangguohang/ai4stronomy/zhuangguohang/ai4astronomy/sd_docs/PJ-Astronomy-spacedebris/log/corner_ab/model_best.pth
    # --qos=gpugpu \