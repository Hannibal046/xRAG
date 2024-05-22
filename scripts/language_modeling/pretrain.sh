## mistral-7b + SFR
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 29666 \
    -m \
    src.language_modeling.train \
    --config config/language_modeling/pretrain.yaml \

## mistral-moe + SFR
accelerate launch \
    --config_file accelerate_fsdp.config \
    -m src.language_modeling.train \
        --config config/language_modeling/pretrain.yaml \
        --chat_format mixtral --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
        --exp_name fsdp_mixtral_moe --per_device_train_batch_size 4 --gradient_accumulation_steps 12
