## mistral-7b + sfr
accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 8 \
    --main_process_port 29666 \
    -m src.language_modeling.train \
        --config config/language_modeling/finetune.yaml \
        --chat_format mistral --model_name_or_path pretrained_model/sfr-mistral-7b \
        --train_file data/instruction_tuning/processed/ablation_data.jsonl
    


## mixtral-moe + sfr
accelerate launch \
    --config_file accelerate_fsdp.config \
    -m src.language_modeling.train \
        --config config/language_modeling/finetune.yaml \
        --chat_format mixtral --model_name_or_path wandb/run-20240310_094951-li520mhm/files/checkpoint/last \
        --exp_name mixtral_moe \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 8