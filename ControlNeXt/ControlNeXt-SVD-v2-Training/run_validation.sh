#!/bin/bash
#
# Script to run validation on the old checkpoint with pose metrics
#

# Set the checkpoint path (use the converted output_dir)
CHECKPOINT_PATH="/restricted/projectnb/cs599dg/faridkar/Pose2Sign/ControlNeXt/ControlNeXt-SVD-v2-Training/outputs/checkpoint-last/output_dir"

# Set the pretrained model path
PRETRAINED_MODEL="stabilityai/stable-video-diffusion-img2vid-xt"

# Set validation data path
VALIDATION_BASE="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation"

# Set output directory (will create validation_output inside checkpoint-last)
OUTPUT_DIR="${CHECKPOINT_PATH}/validation_output"

# Run validation
python validate_checkpoint.py \
    --pretrained_model_name_or_path ${PRETRAINED_MODEL} \
    --checkpoint_path ${CHECKPOINT_PATH} \
    --validation_base ${VALIDATION_BASE} \
    --output_dir ${OUTPUT_DIR} \
    --num_validation_images 5 \
    --width 512 \
    --height 512 \
    --sample_n_frames 14 \
    --mixed_precision fp16

echo "Validation complete! Results saved to: ${OUTPUT_DIR}"
