#!/bin/bash
# Example training commands with different freezing strategies for HyperMotion

# Data paths
DATA_FOLDER="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/training/"
OUTPUT_DIR="./output"

# ============================================================================
# Example 1: Full Fine-tuning (no freezing)
# ============================================================================
# Train all parameters in the transformer
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/full_finetune \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 2: Freeze Early Layers (Transfer Learning)
# ============================================================================
# Freeze the first 16 transformer blocks, train the rest
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/freeze_early \
  --freeze_transformer_layers "0-15" \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 3: Train Only Self-Attention
# ============================================================================
# Freeze cross-attention and FFN, train only self-attention
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/train_self_attn \
  --freeze_cross_attention \
  --freeze_ffn \
  --freeze_patch_embedding \
  --freeze_text_embedding \
  --freeze_time_embedding \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 4: Train Only Cross-Attention
# ============================================================================
# Freeze self-attention and FFN, train only cross-attention
# Useful for adapting to new text prompts/domains
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/train_cross_attn \
  --freeze_self_attention \
  --freeze_ffn \
  --freeze_patch_embedding \
  --freeze_text_embedding \
  --freeze_time_embedding \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 5: Train Only Motion Scale Parameters (Minimal Fine-tuning)
# ============================================================================
# Train ONLY motion_scale and space_scale_factor parameters
# This is the most parameter-efficient approach - only ~32-64 parameters!
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/motion_scale_only \
  --train_motion_scale_only \
  --batch_size 2 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_epochs 50

# ============================================================================
# Example 6: Freeze Embeddings, Train Transformer Blocks
# ============================================================================
# Keep input/text/time embeddings frozen, train transformer blocks
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/freeze_embeddings \
  --freeze_patch_embedding \
  --freeze_text_embedding \
  --freeze_time_embedding \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 7: Freeze Self-Attention, Train Cross-Attention and FFN
# ============================================================================
# For domain adaptation when spatial-temporal understanding is good
# but need to adapt to different conditioning
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/freeze_self_attn \
  --freeze_self_attention \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 8: Custom Layer Freezing
# ============================================================================
# Freeze specific layers: 0,1,2,3 and 28,29,30,31 (first 4 and last 4)
python scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/custom_layers \
  --freeze_transformer_layers "0,1,2,3,28,29,30,31" \
  --batch_size 1 \
  --num_epochs 100

# ============================================================================
# Example 9: Multi-GPU Training with Freezing
# ============================================================================
# Use 4 GPUs with frozen early layers
torchrun --nproc_per_node=4 scripts/train.py \
  --data_folder $DATA_FOLDER \
  --output_dir ${OUTPUT_DIR}/multi_gpu_freeze \
  --freeze_transformer_layers "0-15" \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_epochs 100

# ============================================================================
# Parameter Count Comparison
# ============================================================================
# Full model: ~2B parameters
# Freeze first half (0-15): ~1B trainable parameters
# Train only cross-attention: ~300M parameters
# Train only motion_scale: ~32 parameters (extremely efficient!)

# ============================================================================
# Recommended Strategies by Use Case
# ============================================================================

# 1. Limited compute/memory: --train_motion_scale_only
# 2. Domain adaptation: --freeze_self_attention
# 3. New conditioning type: --freeze_self_attention --freeze_ffn
# 4. Fine-tune on small dataset: --freeze_transformer_layers "0-20"
# 5. Full training on large dataset: no freezing flags
