#!/bin/sh
WorkLOC=/home/vincent/AccAlign #yours

# ======== train with Align6 ==========
# TRAIN_FILE_SRC=/home/vincent/alignment_datasets/ALIGN6/train_data/train.src
# TRAIN_FILE_TGT=/home/vincent/alignment_datasets/ALIGN6/train_data/train.tgt
# TRAIN_FILE_ALIGN=/home/vincent/alignment_datasets/ALIGN6/train_data/train.talp

# EVAL_FILE_SRC=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.src
# EVAL_FILE_TGT=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.tgt
# Eval_gold_file=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.talp
# # =====================================

# ======== train with Align6+JAEN ==========
TRAIN_FILE_SRC=/home/vincent/alignment_datasets/ALIGN6/train_data/train_jaen.src
TRAIN_FILE_TGT=/home/vincent/alignment_datasets/ALIGN6/train_data/train_jaen.tgt
TRAIN_FILE_ALIGN=/home/vincent/alignment_datasets/ALIGN6/train_data/train_jaen.talp

EVAL_FILE_SRC=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.src
EVAL_FILE_TGT=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.tgt
Eval_gold_file=/home/vincent/alignment_datasets/ALIGN6/dev_data/dev.talp
# =====================================

# ======= train with gold target ======
#TRAIN_FILE_SRC=/home/vincent/alignment_datasets/kftt-alignments/data/japanese-dev.txt
#TRAIN_FILE_TGT=/home/vincent/alignment_datasets/kftt-alignments/data/english-dev.txt
#TRAIN_FILE_ALIGN=/home/vincent/alignment_datasets/kftt-alignments/data/align-dev.txt

# train bilingual
# TRAIN_FILE_SRC=/data/datasets/KFTT/kftt-data-1.0/data/tok/kyoto-train.cln.ja
# TRAIN_FILE_TGT=/data/datasets/KFTT/kftt-data-1.0/data/tok/kyoto-train.cln.en

# EVAL_FILE_SRC=/home/vincent/alignment_datasets/kftt-alignments/data/japanese-dev.txt
# EVAL_FILE_TGT=/home/vincent/alignment_datasets/kftt-alignments/data/english-dev.txt
# Eval_gold_file=/home/vincent/alignment_datasets/kftt-alignments/data/align-dev.txt
# ===================================== 

OUTPUT_DIR_ADAPTER=$WorkLOC/adapter_output/try
OUTPUT_DIR=$WorkLOC/model_output
Model=/home/vincent/.cache/torch/sentence_transformers/sentence-transformers_LaBSE
ADAPTER=$WorkLOC/adapter_output/try/checkpoint-1200
EVAL_RES=$WorkLOC/eval_result

# if output_dir_adapter is not empty, resume training (WARNING! will overwrite the saved checkpoint files. TODO: fix)
# if continue training, then define --adapter_path 
CUDA_VISIBLE_DEVICES=0 python $WorkLOC/train_alignment_adapter.py \
    --output_dir_adapter $OUTPUT_DIR_ADAPTER \
    --eval_res_dir $EVAL_RES \
    --model_name_or_path $Model \
    --adapter_path $ADAPTER \
    --extraction 'softmax' \
    --train_so \
    --do_train \
    --do_eval \
    --train_data_file_src $TRAIN_FILE_SRC \
    --train_data_file_tgt $TRAIN_FILE_TGT \
    --train_gold_file $TRAIN_FILE_ALIGN \
    --eval_data_file_src $EVAL_FILE_SRC \
    --eval_data_file_tgt $EVAL_FILE_TGT \
    --per_gpu_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --save_steps 100  \
    --max_steps 1200 \
    --align_layer 6 \
    --logging_steps 50 \
    --eval_gold_file $Eval_gold_file \
    --softmax_threshold 0.1 \


exit

