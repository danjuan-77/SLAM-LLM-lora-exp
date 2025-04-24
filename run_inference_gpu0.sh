#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_volume.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_speed.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_emotion.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_desc.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_lang.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_orig_accent.sh

# nohup bash run_inference_gpu0.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_inference_gpu0_$(date +%Y%m%d%H%M%S).log 2>&1 &