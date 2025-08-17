#!/bin/bash
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_speed_lr5.sh
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_speed_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_volume_lr5.sh
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_volume_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_emotion_lr5.sh
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_emotion_lr5.sh


# nohup bash run_task_node1.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_node1_$(date +%Y%m%d%H%M%S).log 2>&1 &