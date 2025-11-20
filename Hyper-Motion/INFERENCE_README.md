# Inference from Training Checkpoints

This guide explains how to use `scripts/inference_from_checkpoint.py` to generate ASL videos from trained model checkpoints.

## Quick Start

```bash
# Inference with best model
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video /path/to/pose_video.mp4 \
  --ref_image /path/to/reference_image.png \
  --save_path samples/results
```

## Features

✅ **Automatic Checkpoint Loading**
- `--checkpoint best`: Loads the model with lowest validation loss
- `--checkpoint latest`: Loads the most recent checkpoint
- `--checkpoint /path/to/checkpoint`: Loads a specific checkpoint

✅ **Memory Management**
- Multiple GPU memory modes for different hardware
- Supports float16, bfloat16, and float32 precision
- Optional TeaCache for faster inference

✅ **Flexible Configuration**
- All generation parameters configurable via CLI
- Compatible with all training configurations
- Loads base model components (VAE, text encoder, CLIP) from pretrained weights

## Command Line Arguments

### Required Arguments

```bash
--control_video PATH       # Path to pose control video
--ref_image PATH          # Path to reference image
```

### Model Loading

```bash
--config_path PATH        # Model config (default: config/wan2.1/wan_civitai.yaml)
--model_name PATH         # Base model directory (default: ./ckpts)
--checkpoint TYPE/PATH    # Checkpoint to load (default: latest)
--output_dir PATH         # Training output dir (default: output/train)
```

**Checkpoint Options:**
- `best` - Auto-find best model in `{output_dir}/best_model/`
- `latest` - Auto-find latest checkpoint in `{output_dir}/checkpoint-*/`
- `/path/to/checkpoint` - Load specific checkpoint directory
- `/path/to/checkpoint.pt` - Load specific checkpoint file

### Generation Parameters

```bash
--prompt TEXT                    # Text prompt (default: ASL signing prompt)
--negative_prompt TEXT           # Negative prompt (default: deformities)
--guidance_scale FLOAT           # CFG scale (default: 6.0)
--num_inference_steps INT        # Denoising steps (default: 25)
--seed INT                       # Random seed (default: 43)
```

### Video Parameters

```bash
--video_length INT        # Number of frames (default: 49)
--height INT             # Video height (default: 576)
--width INT              # Video width (default: 1024)
--fps INT                # Output FPS (default: 16)
```

### GPU & Performance

```bash
--gpu_memory_mode MODE    # Memory management (default: model_cpu_offload)
--mixed_precision TYPE    # Precision type (default: bf16)
--enable_teacache        # Enable TeaCache acceleration
--teacache_threshold F    # TeaCache threshold (default: 0.10)
```

**GPU Memory Modes:**
- `model_full_load` - Fastest, highest memory usage
- `model_cpu_offload` - Balanced (recommended)
- `sequential_cpu_offload` - Slowest, lowest memory usage
- `model_cpu_offload_and_qfloat8` - Very low memory (quantized)

### Output

```bash
--save_path PATH          # Output directory (default: samples/inference_results)
```

## Examples

### 1. Basic Usage with Best Model

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png
```

### 2. Use Latest Checkpoint

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint latest \
  --output_dir output/train \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png
```

### 3. Specific Checkpoint

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint output/train/checkpoint-10000 \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png
```

### 4. ASL Citizen Validation Data

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/pose/VIDEO.mp4 \
  --ref_image /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames/VIDEO.png \
  --height 224 \
  --width 224 \
  --fps 7 \
  --video_length 49
```

### 5. High Quality Generation

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png \
  --num_inference_steps 50 \
  --guidance_scale 7.5
```

### 6. Memory-Efficient Inference

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png \
  --gpu_memory_mode sequential_cpu_offload \
  --mixed_precision fp16
```

### 7. Fast Inference with TeaCache

```bash
python scripts/inference_from_checkpoint.py \
  --checkpoint best \
  --output_dir output/train \
  --control_video data/pose_video.mp4 \
  --ref_image data/reference.png \
  --enable_teacache \
  --teacache_threshold 0.15
```

### 8. Batch Processing

```bash
#!/bin/bash
for video in data/pose_videos/*.mp4; do
  name=$(basename "$video" .mp4)
  python scripts/inference_from_checkpoint.py \
    --checkpoint best \
    --output_dir output/train \
    --control_video "$video" \
    --ref_image "data/ref_images/${name}.png" \
    --save_path samples/batch_results
done
```

## Checkpoint Structure

The script expects checkpoints in the following structure:

```
output/train/
├── best_model/
│   └── best_model.pt              # Best model checkpoint
├── checkpoint-1000/
│   └── training_state.pt          # Checkpoint at step 1000
├── checkpoint-2000/
│   └── training_state.pt          # Checkpoint at step 2000
└── ...
```

**Checkpoint Files:**
- `best_model.pt` - Contains: model_state_dict, optimizer_state_dict, epoch, step, val_loss
- `training_state.pt` - Contains: model_state_dict, optimizer_state_dict, epoch, step

## Output Files

Generated videos are saved with descriptive filenames:

```
samples/inference_results/
├── video_name_epoch5_step10000.mp4
├── video_name_epoch8_step16000.mp4
└── ...
```

Filename format: `{control_video_name}_{epoch}{epoch_num}_step{step_num}.mp4`

## Tips & Best Practices

### Quality vs Speed

**Higher Quality (slower):**
```bash
--num_inference_steps 50
--guidance_scale 7.5
--gpu_memory_mode model_full_load
```

**Faster (lower quality):**
```bash
--num_inference_steps 15
--guidance_scale 5.0
--enable_teacache
--teacache_threshold 0.20
```

### Memory Usage

**Low Memory (~16GB VRAM):**
```bash
--gpu_memory_mode sequential_cpu_offload
--mixed_precision fp16
```

**High Memory (~40GB VRAM):**
```bash
--gpu_memory_mode model_full_load
--mixed_precision bf16
```

### Video Resolution

Match your training resolution:
- Training at 224x224: Use `--height 224 --width 224`
- Training at 576x1024: Use `--height 576 --width 1024`

### Guidance Scale

- `3.0-5.0`: More diverse, less prompt adherence
- `6.0-7.5`: Balanced (recommended)
- `8.0-10.0`: Strong prompt adherence, less diversity

### Inference Steps

- `10-15`: Fast preview
- `20-25`: Good quality (recommended)
- `30-50`: High quality
- `50+`: Diminishing returns

## Troubleshooting

### Error: "No checkpoints found"
- Verify `--output_dir` points to training output directory
- Check that checkpoints exist in the directory

### Error: "CUDA out of memory"
- Use `--gpu_memory_mode sequential_cpu_offload`
- Reduce `--video_length`
- Use `--mixed_precision fp16`

### Error: "Control video not found"
- Verify paths to `--control_video` and `--ref_image`
- Use absolute paths if relative paths don't work

### Poor Quality Output
- Increase `--num_inference_steps` (e.g., 50)
- Adjust `--guidance_scale` (try 7.0-8.0)
- Check that checkpoint is trained enough
- Verify input video/image quality

## Integration with Training

After training:

1. **Check best model:**
   ```bash
   ls output/train/best_model/
   ```

2. **Run inference on validation set:**
   ```bash
   python scripts/inference_from_checkpoint.py \
     --checkpoint best \
     --output_dir output/train \
     --control_video /path/to/validation/pose/video.mp4 \
     --ref_image /path/to/validation/ref/image.png
   ```

3. **Compare checkpoints:**
   ```bash
   # Best model
   python scripts/inference_from_checkpoint.py --checkpoint best ...

   # Latest checkpoint
   python scripts/inference_from_checkpoint.py --checkpoint latest ...

   # Specific checkpoint
   python scripts/inference_from_checkpoint.py --checkpoint output/train/checkpoint-5000 ...
   ```

## See Also

- `scripts/train.py` - Training script
- `scripts/inference_examples.sh` - Example commands
- `README.md` - Main project README
