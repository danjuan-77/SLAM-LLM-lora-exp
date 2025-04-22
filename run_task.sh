#!/bin/bash
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_wenxi_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_wenxi_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_emotion_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_emotion_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_volume_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_volume_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_speed_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_speed_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_desc_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_desc_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_accent_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_accent_lr5.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_lang_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_lang_lr5.sh

# nohup bash run_task.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_$(date +%Y%m%d%H%M%S).log 2>&1 &