#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_volume.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_speed.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_emotion.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_desc.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_lang.sh
bash ./examples/s2s/scripts/inference/slam-omni0.5b/inference_s2s_batch_wenxi_accent.sh

# nohup bash run_inference_gpu2.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_inference_gpu2_$(date +%Y%m%d%H%M%S).log 2>&1 &
