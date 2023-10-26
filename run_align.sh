#!/bin/sh


WorkLOC=/home/vincent/AccAlign #yours

SRC=/home/vincent/alignment_datasets/kftt-alignments/data/japanese-test.txt
TGT=/home/vincent/alignment_datasets/kftt-alignments/data/english-test.txt

OUTPUT_DIR=$WorkLOC/infer_output
ADAPTER=$WorkLOC/adapter_output/try/checkpoint-1200
Model=/home/vincent/.cache/torch/sentence_transformers/sentence-transformers_LaBSE


# for replicating non-finetune, remove the --adapter_path
python $WorkLOC/train_alignment_adapter.py \
    --infer_path $OUTPUT_DIR \
    --adapter_path $ADAPTER \
    --model_name_or_path $Model \
    --extraction 'softmax' \
    --infer_data_file_src $SRC \
    --infer_data_file_tgt $TGT \
    --per_gpu_train_batch_size 40 \
    --gradient_accumulation_steps 1 \
    --align_layer 6 \
    --softmax_threshold 0.1 \
    --do_test \

exit


