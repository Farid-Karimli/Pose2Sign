#!/bin/bash

# Example usage scripts for inference_from_checkpoint.py
# This script shows different ways to run inference with trained checkpoints

# ============================================================
# Example 1: Inference with best model (auto-find)
# ============================================================
python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint best \
  --output_dir output/train \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/best_model_results \
  --num_inference_steps 25 \
  --guidance_scale 6.0 \
  --seed 43


# ============================================================
# Example 2: Inference with latest checkpoint (auto-find)
# ============================================================
python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint latest \
  --output_dir output/train \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/latest_checkpoint_results \
  --num_inference_steps 25


# ============================================================
# Example 3: Inference with specific checkpoint
# ============================================================
python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint output/train/checkpoint-5000 \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/checkpoint_5000_results


# ============================================================
# Example 4: Inference on ASL Citizen validation data
# ============================================================
VALIDATION_DIR="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation"

python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint best \
  --output_dir output/train \
  --control_video "${VALIDATION_DIR}/pose/01581845388185399-GLASS 2.mp4" \
  --ref_image "${VALIDATION_DIR}/ref_frames/01581845388185399-GLASS 2.png" \
  --prompt "A person signing a word in American Sign Language. High quality, masterpiece, best quality, high resolution." \
  --negative_prompt "Twisted body, limb deformities" \
  --save_path samples/asl_validation_results \
  --video_length 49 \
  --height 224 \
  --width 224 \
  --fps 7 \
  --num_inference_steps 25 \
  --guidance_scale 6.0 \
  --seed 43


# ============================================================
# Example 5: Faster inference with TeaCache
# ============================================================
python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint best \
  --output_dir output/train \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/teacache_results \
  --enable_teacache \
  --teacache_threshold 0.15 \
  --num_inference_steps 25


# ============================================================
# Example 6: Memory-efficient inference (sequential CPU offload)
# ============================================================
python scripts/inference_from_checkpoint.py \
  --config_path config/wan2.1/wan_civitai.yaml \
  --model_name ./ckpts \
  --checkpoint best \
  --output_dir output/train \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/memory_efficient_results \
  --gpu_memory_mode sequential_cpu_offload \
  --mixed_precision fp16


# ============================================================
# Example 7: Batch inference on multiple videos
# ============================================================
# Loop through all pose videos in a directory
POSE_DIR="/path/to/pose_videos"
REF_DIR="/path/to/reference_images"
OUTPUT_DIR="samples/batch_results"

for pose_video in "${POSE_DIR}"/*.mp4; do
  video_name=$(basename "$pose_video" .mp4)
  ref_image="${REF_DIR}/${video_name}.png"

  if [ -f "$ref_image" ]; then
    echo "Processing: $video_name"
    python scripts/inference_from_checkpoint.py \
      --config_path config/wan2.1/wan_civitai.yaml \
      --model_name ./ckpts \
      --checkpoint best \
      --output_dir output/train \
      --control_video "$pose_video" \
      --ref_image "$ref_image" \
      --save_path "$OUTPUT_DIR" \
      --num_inference_steps 25
  else
    echo "Warning: Reference image not found for $video_name"
  fi
done


# ============================================================
# Notes:
# ============================================================
#
# Checkpoint Selection:
# - Use --checkpoint best to load the best model (lowest validation loss)
# - Use --checkpoint latest to load the most recent checkpoint
# - Use --checkpoint /path/to/checkpoint to load a specific checkpoint
#
# GPU Memory Modes:
# - model_full_load: Load entire model on GPU (fastest, needs most memory)
# - model_cpu_offload: Offload model to CPU when not in use (balanced)
# - sequential_cpu_offload: Offload each layer after use (slowest, least memory)
# - model_cpu_offload_and_qfloat8: CPU offload + float8 quantization (very low memory)
#
# Video Parameters:
# - Adjust --height and --width to match your training resolution
# - Adjust --fps to match your training FPS
# - --video_length should match your training video length
#
# Generation Quality:
# - Increase --num_inference_steps for better quality (slower)
# - Adjust --guidance_scale (3-10, higher = more prompt adherence)
# - Change --seed for different random variations
