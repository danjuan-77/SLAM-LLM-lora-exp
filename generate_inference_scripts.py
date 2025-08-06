#!/usr/bin/env python3
"""
脚本生成器：为所有checkpoint和数据集组合生成推理脚本
生成60个任务（10个checkpoint × 6个数据集），合理分配到4个GPU上
"""

import os
from pathlib import Path

# 定义checkpoint列表
checkpoints = [
    "s2s_epoch_1_step_1000",
    "s2s_epoch_1_step_5000", 
    "s2s_epoch_1_step_10000",
    "s2s_epoch_1_step_12000",
    "s2s_epoch_1_step_15000",
    "s2s_epoch_1_step_20000",
    "s2s_epoch_2_step_10096",
    "s2s_epoch_2_step_15096", 
    "s2s_epoch_2_step_20096",
    "s2s_epoch_2_step_25096"
]

# 定义数据集列表
datasets = ["speed", "volume", "language", "emotion", "description", "accent"]

# GPU分配：4个GPU，每个GPU 15个任务
gpu_count = 4
total_tasks = len(checkpoints) * len(datasets)
tasks_per_gpu = total_tasks // gpu_count

print(f"总任务数: {total_tasks}")
print(f"每个GPU任务数: {tasks_per_gpu}")

def generate_inference_script(checkpoint, dataset, gpu_id):
    """生成单个推理脚本"""
    
    script_template = f"""#!/bin/bash
export CUDA_VISIBLE_DEVICES={gpu_id}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=2
export CUDA_LAUNCH_BLOCKING=1


code_dir=examples/s2s

whisper_size=small                  # tiny base small medium large-v3
speech_encoder_path="/share/nlp/tuwenming/models/openai/whisper/${{whisper_size}}.pt"   # replace this with your own whisper model path (different whisper size)
llm_path="/share/nlp/tuwenming/models/Qwen/Qwen2-0.5B"
codec_decoder_path="/share/nlp/tuwenming/models/CosyVoice/CosyVoice-300M-SFT" # replace this with your own CosyVoice model path

encoder_dim=768                     # 384 512 768 896 1024 1280 
mel_size=80                         # 80 128 (128 for whisper-large only, 80 for others)
llm_dim=896                         # 896 1536 2048 3584  -> 0.5B 1.5B 3B 7B

task_type=s2s
split_size=0.2

# vocabulary settings
code_layer=3                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers 
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice                 # CosyVoice or SNAC
codec_decoder_type=CosyVoice
num_latency_tokens=0                # number of latency tokens (same as the number in training)
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

ckpt_path=/mnt/buffer/tuwenming/checkpoints/slam-omni/gpu4-btz1-lr1e-5-warmup_steps5000-SLAM-Omni-fine-tuning-dataset-ultravoice100k-train_all/{checkpoint}

# huggingface dataset
dataset_name={dataset}
manifest_format=parquet
val_data_path="/share/nlp/tuwenming/projects/UltraVoice_dev/data/slam_omni_parquet/test/${{dataset_name}}"
load_from_cache_file=true
dataset_sample_seed=777

# model settings
group_decode=true
group_decode_adapter_type=linear

# decode config
text_repetition_penalty=1.2
audio_repetition_penalty=3        # default 1.0, set to 1.2 for reduce silence
max_new_tokens=3000                 # 500 for SNAC, 3000 for CosyVoice-single
do_sample=false
top_p=1.0
top_k=0
temperature=1.0
decode_text_only=false

output_text_only=false
speech_sample_rate=22050            # 22050 for CosyVoice, 24000 for SNAC
inference_online=false
audio_prompt_path=/share/nlp/tuwenming/projects/UltraVoice_dev/data/spk_voice/alloy.wav      # replace this with your own audio prompt path or our provided audio prompt path

decode_log=$ckpt_path/s2s_decode_${{split}}_trp${{text_repetition_penalty}}_arp${{audio_repetition_penalty}}_seed${{dataset_sample_seed}}_greedy_dataset_${{dataset_name}}
if [ "$do_sample" = true ] ; then
    decode_log=$ckpt_path/s2s_decode_${{split}}_trp${{text_repetition_penalty}}_arp${{audio_repetition_penalty}}_seed${{dataset_sample_seed}}_sampling_topk${{top_k}}_topp${{top_p}}_temp${{temperature}}_dataset_${{dataset_name}}
fi

if [ "$decode_text_only" = true ] ; then
    decode_log=$decode_log"_text_only"
fi

# -m debugpy --listen 5678 --wait-for-client
python $code_dir/inference_s2s.py \\
        --config-path "conf" \\
        --config-name "prompt.yaml" \\
        hydra.run.dir=$ckpt_path \\
        ++model_config.llm_name=qwen2-0.5b \\
        ++model_config.llm_path=$llm_path \\
        ++model_config.llm_dim=$llm_dim \\
        ++model_config.encoder_name=whisper \\
        ++model_config.encoder_projector_ds_rate=5 \\
        ++model_config.encoder_path=$speech_encoder_path \\
        ++model_config.encoder_dim=$encoder_dim \\
        ++model_config.encoder_projector=linear \\
        ++model_config.codec_decoder_path=$codec_decoder_path \\
        ++model_config.codec_decode=true \\
        ++model_config.vocab_config.code_layer=$code_layer \\
        ++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \\
        ++model_config.vocab_config.total_vocabsize=$total_vocabsize \\
        ++model_config.code_type=$code_type \\
        ++model_config.codec_decoder_type=$codec_decoder_type \\
        ++model_config.group_decode=$group_decode \\
        ++model_config.group_decode_adapter_type=$group_decode_adapter_type \\
        ++dataset_config.dataset=speech_dataset_s2s \\
        ++dataset_config.val_data_path=$val_data_path \\
        ++dataset_config.train_data_path=$val_data_path \\
        ++dataset_config.input_type=mel \\
        ++dataset_config.mel_size=$mel_size \\
        ++dataset_config.inference_mode=true \\
        ++dataset_config.manifest_format=$manifest_format \\
        ++dataset_config.split_size=$split_size \\
        ++dataset_config.load_from_cache_file=$load_from_cache_file \\
        ++dataset_config.task_type=$task_type \\
        ++dataset_config.seed=$dataset_sample_seed \\
        ++dataset_config.vocab_config.code_layer=$code_layer \\
        ++dataset_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \\
        ++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \\
        ++dataset_config.code_type=$code_type \\
        ++dataset_config.num_latency_tokens=$num_latency_tokens \\
        ++dataset_config.do_layershift=$do_layershift \\
        ++train_config.model_name=s2s \\
        ++train_config.freeze_encoder=true \\
        ++train_config.freeze_llm=true \\
        ++train_config.freeze_encoder_projector=true \\
        ++train_config.freeze_group_decode_adapter=true \\
        ++train_config.batching_strategy=custom \\
        ++train_config.num_epochs=1 \\
        ++train_config.val_batch_size=1 \\
        ++train_config.num_workers_dataloader=2 \\
        ++train_config.task_type=$task_type \\
        ++decode_config.text_repetition_penalty=$text_repetition_penalty \\
        ++decode_config.audio_repetition_penalty=$audio_repetition_penalty \\
        ++decode_config.max_new_tokens=$max_new_tokens \\
        ++decode_config.task_type=$task_type \\
        ++decode_config.do_sample=$do_sample \\
        ++decode_config.top_p=$top_p \\
        ++decode_config.top_k=$top_k \\
        ++decode_config.temperature=$temperature \\
        ++decode_config.decode_text_only=$decode_text_only \\
        ++decode_config.do_layershift=$do_layershift \\
        ++decode_log=$decode_log \\
        ++log_config.log_file=$decode_log/exp.log \\
        ++decode_config.num_latency_tokens=$num_latency_tokens \\
        ++ckpt_path=$ckpt_path/model.pt \\
        ++output_text_only=$output_text_only \\
        ++inference_online=$inference_online \\
        ++speech_sample_rate=$speech_sample_rate \\
        ++audio_prompt_path=$audio_prompt_path

# nohup bash ./scripts/inference_{checkpoint}_{dataset}.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/run_task_slamomni_inference_{checkpoint}_{dataset}_gpu{gpu_id}_$(date +%Y%m%d%H%M%S).log 2>&1 &
"""
    
    return script_template

def main():
    """主函数：生成所有脚本"""
    
    # 创建脚本目录
    script_dir = Path("scripts")
    script_dir.mkdir(exist_ok=True)
    
    # 生成任务分配
    task_assignments = {i: [] for i in range(gpu_count)}
    task_counter = 0
    
    # 为每个checkpoint和dataset组合生成脚本
    for checkpoint in checkpoints:
        for dataset in datasets:
            gpu_id = task_counter % gpu_count
            task_assignments[gpu_id].append((checkpoint, dataset))
            
            # 生成脚本文件
            script_name = f"inference_{checkpoint}_{dataset}.sh"
            script_path = script_dir / script_name
            
            script_content = generate_inference_script(checkpoint, dataset, gpu_id)
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            # 设置执行权限
            os.chmod(script_path, 0o755)
            
            print(f"生成脚本: {script_name} (GPU {gpu_id})")
            task_counter += 1
    
    # 生成GPU任务运行脚本
    for gpu_id in range(gpu_count):
        gpu_script_name = f"run_gpu{gpu_id}_tasks.sh"
        gpu_script_path = script_dir / gpu_script_name
        
        with open(gpu_script_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"# GPU {gpu_id} 任务列表 (共 {len(task_assignments[gpu_id])} 个任务)\n\n")
            
            for checkpoint, dataset in task_assignments[gpu_id]:
                script_name = f"inference_{checkpoint}_{dataset}.sh"
                f.write(f"# 任务: {checkpoint} - {dataset}\n")
                f.write(f"echo \"开始运行任务: {checkpoint} - {dataset}\"\n")
                f.write(f"bash scripts/{script_name}\n")
                f.write(f"echo \"任务完成: {checkpoint} - {dataset}\"\n\n")
        
        os.chmod(gpu_script_path, 0o755)
        print(f"生成GPU任务脚本: {gpu_script_name}")
    
    # 生成总的运行脚本
    master_script_path = script_dir / "run_all_inference.sh"
    with open(master_script_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        f.write("# 启动所有GPU任务\n\n")
        
        for gpu_id in range(gpu_count):
            f.write(f"# 启动GPU {gpu_id}任务\n")
            f.write(f"nohup bash scripts/run_gpu{gpu_id}_tasks.sh > /share/nlp/tuwenming/projects/UltraVoice_dev/logs/gpu{gpu_id}_tasks_$(date +%Y%m%d%H%M%S).log 2>&1 &\n")
            f.write(f"echo \"GPU {gpu_id} 任务已启动，PID: $!\"\n\n")
        
        f.write("echo \"所有GPU任务已启动\"\n")
        f.write("echo \"使用 'ps aux | grep inference' 查看运行状态\"\n")
    
    os.chmod(master_script_path, 0o755)
    print(f"生成主运行脚本: run_all_inference.sh")
    
    # 打印任务分配总结
    print("\n=== 任务分配总结 ===")
    for gpu_id in range(gpu_count):
        print(f"GPU {gpu_id}: {len(task_assignments[gpu_id])} 个任务")
        for checkpoint, dataset in task_assignments[gpu_id]:
            print(f"  - {checkpoint} - {dataset}")
    
    print(f"\n总共生成了 {total_tasks} 个推理脚本")
    print("运行方式:")
    print("1. 运行所有任务: bash scripts/run_all_inference.sh")
    print("2. 运行单个GPU任务: bash scripts/run_gpu0_tasks.sh")
    print("3. 运行单个脚本: bash scripts/inference_s2s_epoch_1_step_1000_speed.sh")

if __name__ == "__main__":
    main()