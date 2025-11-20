
HF_HOME=/projectnb/herbdl/huggingface CUDA_VISIBLE_DEVICES=0 accelerate launch valid.py \
    --pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid" \
    --output_dir="/restricted/projectnb/cs599dg/faridkar/Pose2Sign/VDM_EVFI/zero_shot" \
    --test_data_path="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/1000_10030_videos/asl" \
    --event_scale=32 \
    --controlnet_model_name_or_path="/projectnb/herbdl/huggingface/hub/models--jxAIbot--VDM_EVFI_VIDEO/snapshots/ff7d93a7c34f11f0f87e9eec891e93e560e4a493/controlnet" \
    --eval_folder_start=0 \
    --eval_folder_end=3 \
    --num_frames=14 \
    --width=224 \
    --height=224 \
    --rescale_factor=2 \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=4 \
    --num_train_epochs=50 \
    --checkpointing_steps=20 \
    --checkpoints_total_limit=1 \
    --learning_rate=3e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --validation_steps=20 \
    --num_workers=8 \
    --num_inference_steps=25 \
    --decode_chunk_size=2 \
    --enable_xformers_memory_efficient_attention \
    --mixed_precision="fp16" \
    --overlapping_ratio=0.1 \
    --t0=0 \
    --M=2 \
    --s_churn=0.5 \
    