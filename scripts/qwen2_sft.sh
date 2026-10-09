set -x

MODEL_PATH=/your_path/exp/qwen_sft_sparse_exp/$1
LR=$2
BSZ=$3
PRETRAIN_PATH=/your_path/checkpoints/$4
LOCAL_BSZ=$5
SPARSE_RATIO=$6
extra_args=$7

DATA_PATH=/your_path/data/OpenOrca-hf,/your_path/data/MetaMathQA/
# DATA_PATH=/your_path/data/MetaMathQA/
# DATA_PATH=/your_path/data/OpenMathInstruct

mkdir -p $MODEL_PATH

deepspeed examples/train_sft.py \
    --max_len 4096 \
    --dataset $DATA_PATH \
    --dataset_probs 0.71,0.29 \
    --max_samples 1500000 \
    --pretrain $PRETRAIN_PATH \
    --save_path $MODEL_PATH \
    --save_steps 1000000 \
    --logging_steps 100 \
    --eval_steps -1 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate $LR \
    --gradient_checkpointing \
    --train_batch_size $BSZ \
    --micro_train_batch_size $LOCAL_BSZ \
    --sparse_ratio $SPARSE_RATIO \
    --lr_scheduler cosine $extra_args