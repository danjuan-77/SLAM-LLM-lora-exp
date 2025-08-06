# SLAM-OMNI 推理任务管理系统

本系统为10个checkpoint × 6个数据集 = 60个推理任务提供自动化管理，合理分配到4个GPU上。

## 📁 文件结构

```
scripts/
├── run_all_inference.sh          # 🚀 启动所有GPU任务的主脚本
├── run_gpu{0-3}_tasks.sh         # 🎯 各GPU的任务队列（每个15个任务）
├── inference_*.sh                # 📜 60个具体的推理脚本
├── monitor_tasks.sh              # 📊 任务监控脚本
└── README.md                     # 📖 本说明文档
```

## 🎯 Checkpoint列表
- s2s_epoch_1_step_1000
- s2s_epoch_1_step_5000  
- s2s_epoch_1_step_10000
- s2s_epoch_1_step_12000
- s2s_epoch_1_step_15000
- s2s_epoch_1_step_20000
- s2s_epoch_2_step_10096
- s2s_epoch_2_step_15096
- s2s_epoch_2_step_20096
- s2s_epoch_2_step_25096

## 📊 数据集列表
- speed (语速测试)
- volume (音量测试)  
- language (多语言测试)
- emotion (情感测试)
- description (描述测试)
- accent (口音测试)

## 🚀 使用方式

### 方式1: 启动所有任务（推荐）
```bash
bash scripts/run_all_inference.sh
```

### 方式2: 分GPU运行
```bash
# 启动GPU0的15个任务
bash scripts/run_gpu0_tasks.sh

# 启动GPU1的15个任务  
bash scripts/run_gpu1_tasks.sh

# 启动GPU2的15个任务
bash scripts/run_gpu2_tasks.sh

# 启动GPU3的15个任务
bash scripts/run_gpu3_tasks.sh
```

### 方式3: 运行单个任务
```bash
# 运行特定的推理任务
bash scripts/inference_s2s_epoch_1_step_1000_speed.sh
```

## 📊 监控和管理

### 实时监控
```bash
# 查看任务状态面板
bash scripts/monitor_tasks.sh

# 实时监控（每30秒刷新）
watch -n 30 'bash scripts/monitor_tasks.sh'
```

### 查看运行状态
```bash
# 查看所有运行中的推理任务
ps aux | grep inference_s2s

# 查看特定GPU的任务
ps aux | grep "CUDA_VISIBLE_DEVICES=0"

# 查看GPU使用情况
nvidia-smi
```

### 日志管理
```bash
# 查看最新日志
ls -lt /share/nlp/tuwenming/projects/UltraVoice_dev/logs/

# 查看特定任务日志
tail -f /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu0_tasks_*.log
```

## 🛑 任务控制

### 停止任务
```bash
# 停止所有推理任务
pkill -f inference_s2s

# 停止特定GPU的任务
pkill -f "CUDA_VISIBLE_DEVICES=0.*inference_s2s"
```

### 重启任务
```bash
# 先停止所有任务
pkill -f inference_s2s

# 等待几秒钟
sleep 5

# 重新启动
bash scripts/run_all_inference.sh
```

## 📈 任务分配策略

- **总任务数**: 60个 (10 checkpoints × 6 datasets)
- **GPU分配**: 每个GPU分配15个任务
- **执行方式**: 每个GPU串行执行任务（确保最佳性能）
- **日志管理**: 自动生成带时间戳的日志文件

## 🎯 GPU任务分配

| GPU | 任务数 | 主要负责 |
|-----|--------|----------|
| GPU0 | 15个   | 平均分配各种组合 |
| GPU1 | 15个   | 平均分配各种组合 |
| GPU2 | 15个   | 平均分配各种组合 |
| GPU3 | 15个   | 平均分配各种组合 |

## 📋 注意事项

1. **资源确保**: 运行前确保模型和数据路径正确
2. **磁盘空间**: 确保有足够空间存储推理结果和日志
3. **内存监控**: 监控GPU内存使用，避免OOM
4. **任务队列**: 每个GPU的任务是串行执行的，确保稳定性
5. **日志清理**: 定期清理旧的日志文件

## 🔍 故障排除

1. **任务卡住**: 检查GPU内存和进程状态
2. **日志错误**: 查看具体的错误日志文件
3. **路径问题**: 确认模型和数据路径是否存在
4. **权限问题**: 确保脚本有执行权限

## 📞 支持

如有问题，请检查：
1. 监控脚本输出: `bash scripts/monitor_tasks.sh`
2. 系统资源: `nvidia-smi`, `htop`
3. 错误日志: 查看对应的log文件