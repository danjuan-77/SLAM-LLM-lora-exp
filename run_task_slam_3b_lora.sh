#!/bin/bash

bash examples/s2s/scripts/finetune/slam-omni3b/finetune_s2s_interleave_lora_ultravoice_sft_gqa_emotion_lr5.sh

# nohup bash run_task_slam_3b_lora.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_3b_lora_$(date +%Y%m%d%H%M%S).log 2>&1 &