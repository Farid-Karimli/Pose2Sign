HF_HOME=/projectnb/herbdl/huggingface CUDA_VISIBLE_DEVICES=2,3 python run_controlnext.py \
  --pretrained_model_name_or_path stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --output_dir outputs \
  --max_frame_num 240 \
  --guidance_scale 6 \
  --batch_frames 24 \
  --sample_stride 1 \
  --overlap 6 \
  --height 256 \
  --width 256 \
  --controlnext_path ./pretrained/controlnet.bin \
  --unet_path ./pretrained/unet.bin \
  --validation_control_video_path /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/asl/81602328051681-LICENSE.mp4 \
  --ref_image_path /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames/81602328051681-LICENSE.png

