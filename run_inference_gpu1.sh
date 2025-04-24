#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_volume.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_speed.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_emotion.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_desc.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_lang.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_accent.sh

# nohup bash run_inference_gpu1.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_inference_gpu1_$(date +%Y%m%d%H%M%S).log 2>&1 &
