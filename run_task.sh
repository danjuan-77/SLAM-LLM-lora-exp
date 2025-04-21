#!/bin/bash


bash examples/s2s/scripts/finetune/finetune_s2s_interleave_lora_ultravoice_sft.sh # lr 0.4

bash examples/s2s/scripts/finetune/finetune_s2s_interleave_lora_ultravoice_sft_lr5.sh # lr 0.5

bash examples/s2s/scripts/finetune/finetune_s2s_interleave_lora_ultravoice_sft_lr3.sh # lr 0.3


# nohup bash run_task.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_$(date +%Y%m%d%H%M%S).log 2>&1 &