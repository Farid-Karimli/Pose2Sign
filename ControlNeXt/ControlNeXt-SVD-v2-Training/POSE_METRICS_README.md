# Pose Faithfulness Metrics Integration

This document describes the pose faithfulness evaluation system integrated into the ControlNeXt training pipeline.

## Overview

Pose faithfulness metrics measure how accurately the generated videos follow the input pose conditioning. This is critical for sign language generation where precise hand, body, and facial movements are essential for meaning.

## Metrics Computed

### 1. **MPJPE (Mean Per Joint Position Error)**
- Average Euclidean distance between corresponding keypoints
- Computed separately for body, hands, and face
- **Lower is better** (0 = perfect match)
- Units: Normalized image coordinates [0, 1]

### 2. **PCK (Percentage of Correct Keypoints)**
- Percentage of keypoints within a threshold distance (default: 0.05)
- **Higher is better** (1.0 = 100% match)
- Most intuitive metric for overall pose accuracy

### 3. **Temporal Smoothness**
- Measures motion jitter/jumpiness (3rd derivative of position)
- **Lower is better** (smoother motion)
- Important for natural-looking sign language

### 4. **Component Breakdown**
- **Body Pose**: 33 MediaPipe keypoints (30% weight)
- **Hand Pose**: 21 keypoints per hand (50% weight - most important for ASL!)
- **Face Pose**: 468 face mesh landmarks (20% weight)

### 5. **Combined Score**
- Weighted average of body, hands, and face metrics
- Hands weighted at 50% due to their importance in sign language
- Single metric for overall pose faithfulness

## What Changed

### Files Modified

1. **`train_svd.py`**
   - Added `compute_pose_faithfulness_metrics()` function at line 398
   - Integrated pose metrics into validation loop at line 1946-1952
   - Updated metrics reporting to include pose metrics
   - Added pose metrics to tensorboard/wandb logging

2. **New File: `pose_metrics.py`**
   - Standalone module for pose faithfulness evaluation
   - Can be used independently for evaluating existing videos
   - Uses MediaPipe for keypoint extraction

### How It Works During Training

1. **During validation** (every `--validation_steps`):
   - Model generates videos from pose conditioning
   - **Image quality metrics** computed (SSIM, PSNR, LPIPS, FID, VFID)
   - **Pose faithfulness metrics** computed (NEW!)
   - Both sets of metrics logged and saved

2. **Pose metrics computation**:
   - Extract poses from input pose video (conditioning)
   - Extract poses from generated video
   - Compare keypoint positions frame-by-frame
   - Compute MPJPE, PCK, and smoothness scores

3. **Output**:
   - Metrics logged to console during training
   - Saved to JSON: `metrics_step_{global_step}.json`
   - Saved to text report: `metrics_step_{global_step}.txt`
   - Logged to tensorboard/wandb as `val/body_mpjpe`, `val/hands_pck`, etc.

## Example Output

### Console Output
```
Computing pose faithfulness for video 0 (example_sign)
Video 0 image quality metrics:
  SSIM: 0.8542
  PSNR: 24.32 dB
  LPIPS: 0.1234

Video 0 pose faithfulness metrics:
  Combined PCK: 0.8750 (87.50%)
  Hands PCK: 0.8234 (82.34%)
  Body PCK: 0.9123 (91.23%)
```

### Metrics Report
```
================================================================================
VALIDATION METRICS REPORT - Step 10000
================================================================================

Number of validation videos: 5

--------------------------------------------------------------------------------
AVERAGE METRICS ACROSS ALL VIDEOS
--------------------------------------------------------------------------------
IMAGE QUALITY METRICS:
  SSIM:  0.8542
  PSNR:  24.32 dB
  LPIPS: 0.1234

POSE FAITHFULNESS METRICS:
  Combined MPJPE: 0.034567
  Combined PCK:   0.8750 (87.50%)

  Body Pose:
    MPJPE:      0.029123
    PCK:        0.9123 (91.23%)
    Smoothness: 0.002341

  Hand Pose:
    MPJPE:      0.041234
    PCK:        0.8234 (82.34%)
    Smoothness: 0.003456

  Face Pose:
    MPJPE:      0.038912
    PCK:        0.8891 (88.91%)
```

## Interpreting the Metrics

### Good vs. Bad Scores

**MPJPE (Lower is better)**
- < 0.02: Excellent pose matching
- 0.02 - 0.05: Good pose matching
- 0.05 - 0.10: Moderate pose matching
- \> 0.10: Poor pose matching

**PCK @ 0.05 threshold (Higher is better)**
- \> 90%: Excellent
- 80% - 90%: Good
- 70% - 80%: Moderate
- < 70%: Poor

**Temporal Smoothness (Lower is better)**
- < 0.005: Very smooth (natural motion)
- 0.005 - 0.010: Moderately smooth
- \> 0.010: Jittery motion (needs improvement)

### What to Focus On

For sign language generation, **prioritize in this order**:
1. **Hands PCK** - Most critical for sign meaning
2. **Combined PCK** - Overall pose accuracy
3. **Temporal Smoothness** - Motion naturalness
4. **Body PCK** - Supporting body posture
5. **Face PCK** - Facial expressions (important for ASL grammar)

## Using the Standalone Evaluator

You can also evaluate videos independently:

```python
from pose_metrics import evaluate_pose_faithfulness

results = evaluate_pose_faithfulness(
    pose_video_path="path/to/pose_input.mp4",
    generated_video_path="path/to/generated_output.mp4",
    pck_threshold=0.05,
    verbose=True,
    save_results="results.json"
)

print(f"Hands PCK: {results['hands']['pck']:.2%}")
print(f"Combined Score: {results['combined']['weighted_pck']:.2%}")
```

## Training with Pose Metrics

No changes needed to your training command! The pose metrics are automatically computed during validation:

```bash
accelerate launch --config_file ./deepspeed.yaml train_svd.py \
  --pretrained_model_name_or_path=stabilityai/stable-video-diffusion-img2vid-xt-1-1 \
  --output_dir="./outputs" \
  --validation_steps=2000 \
  --num_validation_images=5 \
  # ... other args
```

## Troubleshooting

### MediaPipe Import Errors
If you get errors about MediaPipe not being installed:
```bash
pip install mediapipe
```

### Memory Issues
Pose extraction requires additional memory. If you encounter OOM errors:
- Reduce `--num_validation_images`
- The pose extractor is automatically cleaned up after each video

### Missing Pose Detections
If poses aren't detected in some frames:
- Check that pose videos have visible people
- Metrics gracefully handle missing detections (reported as NaN)
- Check the console logs for warnings

## Performance Impact

- **Validation time increase**: ~20-30% longer validation
- **Training speed**: No impact (only runs during validation)
- **Memory**: Small increase during validation only
- **Storage**: Negligible (metrics stored as JSON)

## Future Improvements

Potential enhancements:
1. Add trajectory-based metrics (motion paths)
2. Implement sign-specific metrics (e.g., handshape classification)
3. Add visualization tools for pose comparison
4. Support for multi-person pose tracking

## Questions?

Check the implementation in:
- `pose_metrics.py` - Core pose evaluation functions
- `train_svd.py:398-512` - Integration into training
- `train_svd.py:1946-1976` - Validation loop usage
