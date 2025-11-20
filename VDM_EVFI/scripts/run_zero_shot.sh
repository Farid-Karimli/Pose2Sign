HF_HOME=/scratch/onur151/transformers_cache python inference.py \
  --base_model_path "stabilityai/stable-video-diffusion-img2vid" \
  --checkpoint_path /projectnb/herbdl/huggingface/hub/models--jxAIbot--VDM_EVFI_VIDEO/snapshots/ff7d93a7c34f11f0f87e9eec891e93e560e4a493 \
  --data_path /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation \
  --output_path /restricted/projectnb/cs599dg/faridkar/Pose2Sign/VDM_EVFI/zero_shot \
  --data_item_start 0 \
  --data_item_end 3 \
  --width 256 \
  --height 256 \