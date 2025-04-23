#!/bin/bash
bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_desc_lr5.sh

# bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_accent_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_accent_lr5.sh

# bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_lang_lr4.sh

bash examples/s2s/scripts/finetune/slam-omni0.5b/finetune_s2s_group_ultravoice_sft_gqa_lang_lr5.sh

# nohup bash run_task_node2.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_sft_node2_$(date +%Y%m%d%H%M%S).log 2>&1 &