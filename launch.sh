#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Options (can appear in any order after mode and model_size):
#   --lr-schedule <cosine|WSD|constant>   LR schedule (default: cosine for train, constant for throughput)
#   --wsd-decay-pct <int>                 % of training steps used for WSD decay phase (default: 30)
#   --gbs <N>                             Override global batch size (default: 256)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 2000 1
#            ./launch.sh train 760m 2000 1 --lr-schedule cosine
#            ./launch.sh train 760m 2000 1 --lr-schedule WSD --wsd-decay-pct 30
#            ./launch.sh train 760m 2000 1 --lr-schedule WSD --wsd-decay-pct 20

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [options]}
shift 2

################ Parse remaining args ################
LR_SCHEDULE=""        # empty = use mode default (cosine for train, constant for throughput)
WSD_DECAY_PCT=30      # % of total steps used for the WSD decay phase
GBS_OVERRIDE=""
MBS_OVERRIDE=""
DRY_RUN=false

_POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --lr-schedule)    LR_SCHEDULE="${2:?--lr-schedule requires cosine|WSD|constant}"; shift 2;;
        --wsd-decay-pct)  WSD_DECAY_PCT="${2:?--wsd-decay-pct requires an integer}"; shift 2;;
        --gbs)            GBS_OVERRIDE="${2:?--gbs requires N}"; shift 2;;
        --mbs)            MBS_OVERRIDE="${2:?--mbs requires N}"; shift 2;;
        --dry-run)        DRY_RUN=true; shift;;
        -*)               echo "Unknown option: $1"; exit 1;;
        *)                _POSITIONAL+=("$1"); shift;;
    esac
done

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${_POSITIONAL[0]:-50}
        NODES=${_POSITIONAL[1]:-4}
        TIME=00:20:00   # default; overridden per model size below after MODEL_SIZE case
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        [[ -z "$LR_SCHEDULE" ]] && LR_SCHEDULE="constant"
        CHECKPOINT_EXTRA=""   # no checkpointing for 50-step throughput runs
        ;;
    train|batchsize)
        TRAINING_STEPS=${_POSITIONAL[0]:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${_POSITIONAL[1]:-4}
        # 1h per job slice: shorter jobs get higher SLURM priority.
        # Jobs checkpoint every 100 steps and resume automatically on resubmit.
        TIME=01:00:00
        EVAL_INTERVAL=50
        EVAL_ITERS=10
        LR_WARMUP_ITERS=100
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        # Checkpoint every 50 steps so a failed job can resume from last save
        CHECKPOINT_EXTRA="
    --save \$CHECKPOINT_DIR
    --save-interval 50
    --load \$CHECKPOINT_DIR"
        WANDB=true
        [[ -z "$LR_SCHEDULE" ]] && LR_SCHEDULE="cosine"
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train, batchsize"
        exit 1
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

GBS=${GBS_OVERRIDE:-256}
[[ -n "$MBS_OVERRIDE" ]] && MBS="$MBS_OVERRIDE"
SEQ_LEN=4096

# For throughput mode, set time limits per model size.
# Estimates: startup+JIT ~10 min, then 50 steps of training.
# 125m/350m/760m: ~5-11 min training → 30 min total  (+15 min buffer)
# 1.5b:           ~20 min training  → 45 min total  (+15 min buffer)
# 3b:             ~39 min training  → 70 min total  (+20 min buffer)
# 8b:             ~60 min training  → 90 min total  (+20 min buffer)
# Multi-node jobs finish faster per wall-clock but keep same limit for safety.
if [[ "$MODE" == "throughput" ]]; then
    case $MODEL_SIZE in
        125m|350m|760m) TIME=00:30:00 ;;
        1.5b)           TIME=00:45:00 ;;
        3b)             TIME=01:10:00 ;;
        8b)             TIME=01:30:00 ;;
    esac
fi
if [[ "$LR_SCHEDULE" == "WSD" ]]; then
    LR_TAG="WSD${WSD_DECAY_PCT}"
else
    LR_TAG="${LR_SCHEDULE}"
fi
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-${LR_TAG}-gbs${GBS}-mbs${MBS}"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: \$(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${JOB_NAME}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CHECKPOINT_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/checkpoints/\$EXP_NAME
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

# Strip user conda/miniconda from PATH so the container's Python and torchrun
# are used instead of the submitting shell's conda environment.
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v -E '(conda|miniconda)' | tr '\n' ':' | sed 's/:$//')
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR $CHECKPOINT_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --min-lr 3e-5
    --lr-warmup-iters ${LR_WARMUP_ITERS}
    --lr-decay-style ${LR_SCHEDULE}
$(if [[ "${LR_SCHEDULE}" == "cosine" ]]; then
    echo "    --lr-decay-iters ${TRAINING_STEPS}"
elif [[ "${LR_SCHEDULE}" == "WSD" ]]; then
    WSD_DECAY_ITERS=$(( TRAINING_STEPS * WSD_DECAY_PCT / 100 ))
    echo "    --lr-decay-iters ${TRAINING_STEPS}"
    echo "    --lr-wsd-decay-iters ${WSD_DECAY_ITERS}"
    echo "    --lr-wsd-decay-style cosine"
fi)
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
${CHECKPOINT_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN — not submitted. To submit:"
    echo "  sbatch $SCRIPT"
else
    sbatch "$SCRIPT"
fi
