set -x

CONFIG=$1
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
QUOTATYPE=${QUOTATYPE:-'reserved'}
PY_ARGS=${@:2}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=$RANDOM srun -p ai4science \
    -x g0824 \
    --job-name=${JOB_NAME} \
    --quotatype=${QUOTATYPE} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/trainval.py ${CONFIG} --launcher="slurm" ${PY_ARGS}