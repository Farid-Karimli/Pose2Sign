CUDA_VISIBLE_DEVICES=1 accelerate  launch --num_processes 1 train.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="checkpoints" \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=16 \
    --num_train_epochs=50 \
    --checkpointing_steps=50 --checkpoints_total_limit=10 \
    --learning_rate=5e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --validation_steps=200 \
    --num_workers=4 \
    --enable_xformers_memory_efficient_attention \
    --train_data_path="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/training" \
    --logging_dir my_tensorboard_logs \
    --report_to tensorboard \

#--pretrain_unet="/restricted/projectnb/cs599dg/Pose2Sign/VDM_EVFI/scripts/checkpoints/checkpoint-650" \
#    --controlnet_model_name_or_path="/restricted/projectnb/cs599dg/Pose2Sign/VDM_EVFI/scripts/checkpoints/checkpoint-650" \