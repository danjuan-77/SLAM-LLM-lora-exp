#!/bin/bash

bash examples/s2s/scripts/inference/local4090/inference_s2s_batch_accent.sh

bash examples/s2s/scripts/inference/local4090/inference_s2s_batch_emotion.sh

bash examples/s2s/scripts/inference/local4090/inference_s2s_batch_speed.sh

bash examples/s2s/scripts/inference/local4090/inference_s2s_batch_style.sh

bash examples/s2s/scripts/inference/local4090/inference_s2s_batch_volume.sh


# nohup bash run_task.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_$(date +%Y%m%d%H%M%S).log 2>&1 &