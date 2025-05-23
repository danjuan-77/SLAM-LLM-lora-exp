#!/bin/bash

DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/llama_qa/gpu4-btz1-lr1e-4-interleave_text12_audio36-Qwen2.5-3b-instruct-gradient_accumulation2-lora-audio_embed_only-lora_rank1024-alpha2048
FORMAT=tsv    # tsv jsonl

PRED_FILE="$DECODE_DIR/pred_text"
PRED_FILE="$DECODE_DIR/pred_audio_asr_text"
GT_FILE="$DECODE_DIR/gt_text"

if [ "$FORMAT" == "jsonl" ]; then
    PRED_FILE="$DECODE_DIR/test.jsonl"
fi

# -m debugpy --listen 5678 --wait-for-client
python ./examples/s2s/evaluation/close_qa.py \
    --pred "$PRED_FILE" \
    --gt "$GT_FILE" \
    --exist \
    --format "$FORMAT" \
    --show-mixmatch

# bash ./examples/s2s/scripts/eval/close_qa.sh