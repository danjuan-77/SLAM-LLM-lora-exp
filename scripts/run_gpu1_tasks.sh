#!/bin/bash

# GPU 1 任务列表 (共 15 个任务)

# # 任务: s2s_epoch_1_step_1000 - volume
# echo "开始运行任务: s2s_epoch_1_step_1000 - volume"
# bash scripts/inference_s2s_epoch_1_step_1000_volume.sh
# echo "任务完成: s2s_epoch_1_step_1000 - volume"

# # 任务: s2s_epoch_1_step_1000 - accent
# echo "开始运行任务: s2s_epoch_1_step_1000 - accent"
# bash scripts/inference_s2s_epoch_1_step_1000_accent.sh
# echo "任务完成: s2s_epoch_1_step_1000 - accent"

# # 任务: s2s_epoch_1_step_5000 - emotion
# echo "开始运行任务: s2s_epoch_1_step_5000 - emotion"
# bash scripts/inference_s2s_epoch_1_step_5000_emotion.sh
# echo "任务完成: s2s_epoch_1_step_5000 - emotion"

# 任务: s2s_epoch_1_step_10000 - volume
echo "开始运行任务: s2s_epoch_1_step_10000 - volume"
bash scripts/inference_s2s_epoch_1_step_10000_volume.sh
echo "任务完成: s2s_epoch_1_step_10000 - volume"

# 任务: s2s_epoch_1_step_10000 - accent
echo "开始运行任务: s2s_epoch_1_step_10000 - accent"
bash scripts/inference_s2s_epoch_1_step_10000_accent.sh
echo "任务完成: s2s_epoch_1_step_10000 - accent"

# 任务: s2s_epoch_1_step_12000 - emotion
echo "开始运行任务: s2s_epoch_1_step_12000 - emotion"
bash scripts/inference_s2s_epoch_1_step_12000_emotion.sh
echo "任务完成: s2s_epoch_1_step_12000 - emotion"

# 任务: s2s_epoch_1_step_15000 - volume
echo "开始运行任务: s2s_epoch_1_step_15000 - volume"
bash scripts/inference_s2s_epoch_1_step_15000_volume.sh
echo "任务完成: s2s_epoch_1_step_15000 - volume"

# 任务: s2s_epoch_1_step_15000 - accent
echo "开始运行任务: s2s_epoch_1_step_15000 - accent"
bash scripts/inference_s2s_epoch_1_step_15000_accent.sh
echo "任务完成: s2s_epoch_1_step_15000 - accent"

# 任务: s2s_epoch_1_step_20000 - emotion
echo "开始运行任务: s2s_epoch_1_step_20000 - emotion"
bash scripts/inference_s2s_epoch_1_step_20000_emotion.sh
echo "任务完成: s2s_epoch_1_step_20000 - emotion"

# 任务: s2s_epoch_2_step_10096 - volume
echo "开始运行任务: s2s_epoch_2_step_10096 - volume"
bash scripts/inference_s2s_epoch_2_step_10096_volume.sh
echo "任务完成: s2s_epoch_2_step_10096 - volume"

# 任务: s2s_epoch_2_step_10096 - accent
echo "开始运行任务: s2s_epoch_2_step_10096 - accent"
bash scripts/inference_s2s_epoch_2_step_10096_accent.sh
echo "任务完成: s2s_epoch_2_step_10096 - accent"

# 任务: s2s_epoch_2_step_15096 - emotion
echo "开始运行任务: s2s_epoch_2_step_15096 - emotion"
bash scripts/inference_s2s_epoch_2_step_15096_emotion.sh
echo "任务完成: s2s_epoch_2_step_15096 - emotion"

# 任务: s2s_epoch_2_step_20096 - volume
echo "开始运行任务: s2s_epoch_2_step_20096 - volume"
bash scripts/inference_s2s_epoch_2_step_20096_volume.sh
echo "任务完成: s2s_epoch_2_step_20096 - volume"

# 任务: s2s_epoch_2_step_20096 - accent
echo "开始运行任务: s2s_epoch_2_step_20096 - accent"
bash scripts/inference_s2s_epoch_2_step_20096_accent.sh
echo "任务完成: s2s_epoch_2_step_20096 - accent"


