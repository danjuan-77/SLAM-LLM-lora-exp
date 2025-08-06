#!/bin/bash

# 启动所有GPU任务

# 启动GPU 0任务
nohup bash scripts/run_gpu0_tasks.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu0_tasks_$(date +%Y%m%d%H%M%S).log 2>&1 &
echo "GPU 0 任务已启动，PID: $!"

# 启动GPU 1任务
nohup bash scripts/run_gpu1_tasks.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu1_tasks_$(date +%Y%m%d%H%M%S).log 2>&1 &
echo "GPU 1 任务已启动，PID: $!"

# 启动GPU 2任务
nohup bash scripts/run_gpu2_tasks.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu2_tasks_$(date +%Y%m%d%H%M%S).log 2>&1 &
echo "GPU 2 任务已启动，PID: $!"

# 启动GPU 3任务
nohup bash scripts/run_gpu3_tasks.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu3_tasks_$(date +%Y%m%d%H%M%S).log 2>&1 &
echo "GPU 3 任务已启动，PID: $!"

echo "所有GPU任务已启动"
echo "使用 'ps aux | grep inference' 查看运行状态"
