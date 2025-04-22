#!/bin/bash
export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false

code_dir=examples/s2s
num_gpus_per_node=$(( $(echo ${CUDA_VISIBLE_DEVICES} | tr -cd ',' | wc -c) + 1 ))
num_nodes=1
num_gpus=$(( num_gpus_per_node * num_nodes ))

whisper_size=small                  # tiny base small medium large-v3
speech_encoder_path="/share/nlp/tuwenming/models/openai/whisper/${whisper_size}.pt"   # different whisper size
llm_path="/share/nlp/tuwenming/models/Qwen/Qwen2.5-3B/"  # Qwen/Qwen2-0.5B, you can choose other Qwen models (Qwen2 or Qwen2.5)
llm_name=Qwen2.5-3B

encoder_dim=768                     # 384 512 768 1024 1280
mel_size=80                         # 80 128 ( only whisper-large-v3 supports 128 )
llm_dim=2048                         # 896 1536 2048 3584  -> 0.5B 1.5B 3B 7B

# vocabulary settings
code_layer=0                        # 1 single semantic code layer   2 3 4 5 6 7 8 group semantic code layers  0 for interleaved paradigm
total_audio_vocabsize=4160          # the vocab size of the codec token
llm_vocabsize=152000                # the vocab size of the LLM model (Qwen2 here)
total_vocabsize=$((total_audio_vocabsize + llm_vocabsize))

# code settings
code_type=CosyVoice                 # CosyVoice or SNAC
do_layershift=false                 # if false, tokens in each layers use the same codebook, otherwise, use different codebooks

# dataset settings
manifest_format=parquet             # parquet or jsonl
train_data_path=/share/nlp/tuwenming/datasets/ultravoice160k/gqa_language # train data & validation data
val_data_path=/share/nlp/tuwenming/datasets/ultravoice160k/gqa_language
load_from_cache_file=true           # set to true if you have already generated the cache file, otherwise set to false

# training settings
modeling_paradigm=interleaved
interleaved_text_token_num=12
interleaved_audio_token_num=36
batch_size_training=1
use_fp16=true
freeze_llm=true
num_epochs=10
lr=1e-4
task_type=s2s
warmup_steps=5500
total_steps=55000
gradient_accumulation_steps=1
train_audio_embed_only=true

# PEFT settings
use_peft=true
lora_r=384
lora_alpha=$((lora_r * 2))

# validation settings
validation_interval=1000
split_size=0.01

exp_name="gpu${num_gpus}-btz${batch_size_training}-lr${lr}-warmup_steps${warmup_steps}-interleave_text${interleaved_text_token_num}_audio${interleaved_audio_token_num}-Qwen2.5-3b-gradient_accumulation${gradient_accumulation_steps}-lora-audio_embed_only-lora_rank${lora_r}-alpha${lora_alpha}"
# exp_name="debug"
wandb_entity_name=kevin-tutu
# wandb_project_name=slam-omni-3b-lora-finetune
wandb_project_name=test


home_dir=/mnt/buffer/tuwenming/checkpoints/slam-omni-3b
output_dir=$home_dir/$exp_name
ckpt_path=/mnt/buffer/tuwenming/checkpoints/slam-omni-3b/wenxi/gpu4-btz1-lr1e-4-interleave_text12_audio36-Qwen2.5-3b-gradient_accumulation2-lora-audio_embed_only-lora_rank384-alpha768-s2s_epoch_3_step_68390  # this line is for resuming training

if [ "$exp_name" = "debug" ]; then
    use_wandb=false
else
    use_wandb=true
fi
wandb_exp_name=$exp_name
# use_wandb=false

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=$llm_name \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=linear \
++model_config.vocab_config.code_layer=$code_layer \
++model_config.vocab_config.total_audio_vocabsize=$total_audio_vocabsize \
++model_config.vocab_config.total_vocabsize=$total_vocabsize \
++model_config.code_type=$code_type \
++dataset_config.dataset=speech_dataset_s2s \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=$mel_size \
++dataset_config.seed=42 \
++dataset_config.manifest_format=$manifest_format \
++dataset_config.split_size=$split_size \
++dataset_config.load_from_cache_file=$load_from_cache_file \
++dataset_config.task_type=$task_type \
++dataset_config.vocab_config.code_layer=$code_layer \
++dataset_config.vocab_config.total_vocabsize=$total_vocabsize \
++dataset_config.code_type=$code_type \
++dataset_config.do_layershift=$do_layershift \
++dataset_config.modeling_paradigm=$modeling_paradigm \
++dataset_config.interleaved_text_token_num=$interleaved_text_token_num \
++dataset_config.interleaved_audio_token_num=$interleaved_audio_token_num \
++train_config.model_name=s2s \
++train_config.num_epochs=$num_epochs \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=$freeze_llm \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=$warmup_steps \
++train_config.total_steps=$total_steps \
++train_config.lr=$lr \
++train_config.validation_interval=$validation_interval \
++train_config.batch_size_training=$batch_size_training \
++train_config.val_batch_size=$batch_size_training \
++train_config.num_workers_dataloader=0 \
++train_config.output_dir=$output_dir \
++train_config.use_fp16=$use_fp16 \
++train_config.task_type=$task_type \
++train_config.use_peft=$use_peft \
++train_config.modeling_paradigm=$modeling_paradigm \
++train_config.interleaved_text_token_num=$interleaved_text_token_num \
++train_config.interleaved_audio_token_num=$interleaved_audio_token_num \
++train_config.gradient_accumulation_steps=$gradient_accumulation_steps \
++train_config.train_audio_embed_only=$train_audio_embed_only \
++train_config.peft_config.lora_alpha=$lora_alpha \
++train_config.peft_config.r=$lora_r \
++metric=acc \
++log_config.use_wandb=$use_wandb \
++log_config.wandb_entity_name=$wandb_entity_name \
++log_config.wandb_project_name=$wandb_project_name \
++log_config.wandb_exp_name=$wandb_exp_name \
++log_config.wandb_dir=$output_dir \
++log_config.log_file=$output_dir/exp.log \
++log_config.log_interval=100 \
++ckpt_path=$ckpt_path/model.pt \
"
# ↑ this line is for resuming training


if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    if [ "$exp_name" = "debug" ]; then
        python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            $hydra_args
    else
        python $code_dir/finetune_s2s.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            $hydra_args
    fi
else
    torchrun \
        --nnodes $num_nodes \
        --nproc_per_node $num_gpus_per_node \
        --master_port=1234 \
        $code_dir/finetune_s2s.py \
        --config-path "conf" \
        --config-name "prompt.yaml" \
        ++train_config.enable_ddp=true \
        ++train_config.enable_fsdp=false \
        $hydra_args
fi

# for multi-machine training, you should add the following line to the torchrun command
# --node_rank=$node_rank \
# --master_addr=$master_addr \

# bash examples/s2s/scripts/finetune/finetune_s2s_interleave_lora_ultravoice_sft_gqa_lang_lr4.sh