#!/bin/bash

DECODE_DIR=/home/wenxi/mydisk/exp/standard_qa_eval/trivia_qa/qwen2.5-7b-instruct
FORMAT=jsonl    # tsv jsonl

PRED_FILE="$DECODE_DIR/pred_text"
# PRED_FILE="$DECODE_DIR/pred_audio_asr_text"
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
    # --show-mixmatch

# bash ./examples/s2s/scripts/eval/close_qa.sh