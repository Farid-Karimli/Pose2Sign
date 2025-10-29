accelerate  launch --config_file ./deepspeed.yaml train_svd.py \
 --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
 --output_dir="./outputs" \
 --dataset_type="asl" \
 --meta_info_path="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/training" \
 --validation_image_folder="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames" \
 --validation_control_folder="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/pose" \
 --validation_image=$PATH_TO_THE_REFERENCE_IMAGE_FILE_FOR_EVALUATION \
 --num_validation_images=5 \
 --width=512 \
 --height=512 \
 --lr_warmup_steps 500 \
 --sample_n_frames 14 \
 --interval_frame 3 \
 --learning_rate=1e-5 \
 --per_gpu_batch_size=1 \
 --num_train_epochs=100 \
 --mixed_precision="bf16" \
 --gradient_accumulation_steps=1 \
 --checkpointing_steps=2000 \
 --validation_steps=2000 \
 --gradient_checkpointing \
 --checkpoints_total_limit 4 \
 --resume_from_checkpoint "latest"

# # For Resume
#  --controlnet_model_name_or_path $PATH_TO_THE_CONTROLNEXT_WEIGHT
#  --unet_model_name_or_path $PATH_TO_THE_UNET_WEIGHT


