#!/bin/bash

# 任务监控脚本 - 实时查看所有GPU任务状态

echo "================================="
echo "SLAM-OMNI 推理任务监控面板"
echo "================================="
echo "时间: $(date)"
echo ""

# 检查运行中的推理任务
echo "📊 运行中的推理任务:"
echo "---------------------------------"
running_tasks=$(ps aux | grep "inference_s2s" | grep -v grep | wc -l)
if [ $running_tasks -gt 0 ]; then
    ps aux | grep "inference_s2s" | grep -v grep | awk '{print "PID: " $2 " | GPU: " $15 " | 任务: " $11}' | head -10
    if [ $running_tasks -gt 10 ]; then
        echo "... 还有 $((running_tasks - 10)) 个任务在运行"
    fi
else
    echo "❌ 当前没有推理任务在运行"
fi
echo ""

# 检查各GPU使用情况
echo "🖥️ GPU使用情况:"
echo "---------------------------------"
for gpu_id in {0..3}; do
    gpu_tasks=$(ps aux | grep "CUDA_VISIBLE_DEVICES=$gpu_id" | grep "inference_s2s" | grep -v grep | wc -l)
    echo "GPU $gpu_id: $gpu_tasks 个任务运行中"
done
echo ""

# 检查日志目录
echo "📝 最新日志文件:"
echo "---------------------------------"
log_dir="/share/nlp/tuwenming/projects/UltraVoice_dev/logs"
if [ -d "$log_dir" ]; then
    find "$log_dir" -name "*.log" -type f -mmin -60 | head -5 | while read log_file; do
        echo "📄 $(basename "$log_file") ($(date -r "$log_file" '+%H:%M'))"
    done
else
    echo "❌ 日志目录不存在: $log_dir"
fi
echo ""

# 任务完成统计
echo "📈 任务进度统计:"
echo "---------------------------------"
total_scripts=60
completed_count=0

# 检查所有checkpoint和dataset组合
checkpoints=("s2s_epoch_1_step_1000" "s2s_epoch_1_step_5000" "s2s_epoch_1_step_10000" "s2s_epoch_1_step_12000" "s2s_epoch_1_step_15000" "s2s_epoch_1_step_20000" "s2s_epoch_2_step_10096" "s2s_epoch_2_step_15096" "s2s_epoch_2_step_20096")
datasets=("speed" "volume" "language" "emotion" "description" "accent")

for checkpoint in "${checkpoints[@]}"; do
    for dataset in "${datasets[@]}"; do
        output_dir="/mnt/buffer/tuwenming/checkpoints/slam-omni/gpu4-btz1-lr1e-5-warmup_steps5000-SLAM-Omni-fine-tuning-dataset-ultravoice100k-train_all/${checkpoint}/s2s_decode_trp1.2_arp3_seed777_greedy_dataset_${dataset}"
        if [ -d "$output_dir" ] && [ -f "$output_dir/exp.log" ]; then
            # 检查是否完成（简单检查日志文件是否存在且不为空）
            if [ -s "$output_dir/exp.log" ]; then
                ((completed_count++))
            fi
        fi
    done
done

echo "总任务数: $total_scripts"
echo "已完成: $completed_count"
echo "剩余: $((total_scripts - completed_count))"
echo "进度: $(( completed_count * 100 / total_scripts ))%"
echo ""

# 提供操作建议
echo "🛠️ 操作命令:"
echo "---------------------------------"
echo "启动所有任务: bash scripts/run_all_inference.sh"
echo "启动GPU0任务: bash scripts/run_gpu0_tasks.sh"
echo "查看特定GPU任务: ps aux | grep 'CUDA_VISIBLE_DEVICES=0'"
echo "杀死所有推理任务: pkill -f 'inference_s2s'"
echo "实时监控: watch -n 30 'bash scripts/monitor_tasks.sh'"
echo ""

echo "================================="