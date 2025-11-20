# Metrics Integration in train_svd.py

## Overview

The validation loop in `train_svd.py` now automatically computes comprehensive video quality metrics during validation. Metrics are computed, logged, and saved to disk for each validation checkpoint.

## Metrics Computed

The following metrics are computed for each validation video:

1. **SSIM (Structural Similarity Index)** - Higher is better (0-1)
   - Measures structural similarity between frames
   - Frame-based metric

2. **PSNR (Peak Signal-to-Noise Ratio)** - Higher is better (dB)
   - Measures reconstruction quality
   - Frame-based metric

3. **MSE (Mean Squared Error)** - Lower is better
   - Measures pixel-wise difference
   - Frame-based metric

4. **LPIPS (Learned Perceptual Image Patch Similarity)** - Lower is better (0-1)
   - Measures perceptual similarity using learned features
   - Requires `lpips` package: `pip install lpips`
   - Frame-based metric

5. **FID (Fréchet Inception Distance)** - Lower is better
   - Measures distributional similarity using Inception V3 features
   - Treats all frames as a distribution
   - Requires torchvision with Inception V3

6. **VFID (Video FID)** - Lower is better
   - Video-specific variant of FID
   - Treats video frames as a temporal distribution
   - Requires torchvision with Inception V3

## Normalization Handling

The metrics implementation respects the normalization expectations of each metric:

- **Input to metrics**: PIL Images are automatically converted to numpy arrays (H, W, C) in RGB format with uint8 values [0-255]
- **SSIM/PSNR**: Images are normalized to [0, 1] internally
- **LPIPS**: Images are normalized to [-1, 1] internally
- **FID/VFID**: Images are resized to 299x299 and normalized with ImageNet statistics

All normalization is handled automatically within the metrics functions.

## Validation Flow

During validation (every `--validation_steps`):

1. **Load validation data**: Reference frames, pose videos, and ground truth ASL videos
2. **Generate videos**: Run the model to generate sign language videos
3. **Save videos**: Save side-by-side comparisons of pose and generated videos
4. **Compute metrics**: For each generated video:
   - Compare against ground truth frame-by-frame
   - Compute all 6 metrics (if dependencies available)
   - Log individual video metrics to console
5. **Generate report**: Create comprehensive metrics report with:
   - Per-video metrics
   - Average metrics across all validation videos
6. **Save reports**: Save both JSON and human-readable text reports
7. **Log to tracker**: Log average metrics to tensorboard/wandb

## Output Files

For each validation checkpoint at step `N`, the following files are saved in `outputs/validation_images/validation_N_vids/`:

1. **Videos**: `v{i}_{basename}.mp4` - Side-by-side comparison videos
2. **Metrics JSON**: `metrics_step_N.json` - Machine-readable metrics
3. **Metrics Report**: `metrics_step_N.txt` - Human-readable report

### Example metrics_step_N.txt:

```
================================================================================
VALIDATION METRICS REPORT - Step 1000
================================================================================

Number of validation videos: 3

--------------------------------------------------------------------------------
AVERAGE METRICS ACROSS ALL VIDEOS
--------------------------------------------------------------------------------
  SSIM:  0.8523
  PSNR:  24.56 dB
  MSE:   0.001234
  LPIPS: 0.1234
  FID:   12.3456
  VFID:  11.2345

--------------------------------------------------------------------------------
PER-VIDEO METRICS
--------------------------------------------------------------------------------

Video 0:
  SSIM:  0.8621
  PSNR:  25.12 dB
  MSE:   0.001156
  LPIPS: 0.1198
  FID:   11.9876
  VFID:  10.8765

Video 1:
  SSIM:  0.8456
  PSNR:  24.23 dB
  MSE:   0.001289
  LPIPS: 0.1245
  FID:   12.4567
  VFID:  11.3456

...
```

## Console Output

During validation, you'll see output like:

```
Running validation...
Generating 3 videos.
Found 3 matching validation triplets (ref + pose + GT)
Generating video 0 (01581845388185399-GLASS 2) with 42 frames
Generating video 1 (026756336707836725-LIMITED) with 38 frames
Computing metrics for video 0 (01581845388185399-GLASS 2)
Video 0 metrics:
  SSIM: 0.8621
  PSNR: 25.12 dB
  LPIPS: 0.1198
...
================================================================================
VALIDATION SUMMARY - Step 1000
================================================================================
Evaluated 3 videos
Average metrics:
  SSIM: 0.8523
  PSNR: 24.56
  MSE: 0.0012
  LPIPS: 0.1234
  FID: 12.3456
  VFID: 11.2345
================================================================================
```

## Validation Data Structure

The validation data is expected to be organized as:

```
/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/
├── ref_frames/          # Reference frames (PNG/JPG)
│   ├── basename1.png
│   └── basename2.png
├── pose/                # Pose condition videos (MP4)
│   ├── basename1.mp4
│   └── basename2.mp4
└── asl/                 # Ground truth ASL videos (MP4)
    ├── basename1.mp4
    └── basename2.mp4
```

Files are matched by basename (without extension). Only triplets with matching basenames in all three folders will be used for validation.

## Dependencies

Required:
- numpy
- scipy
- scikit-image
- torch
- torchvision

Optional (for all metrics):
- `pip install lpips` - For LPIPS metric
- torchvision with Inception V3 - For FID/VFID metrics

If optional dependencies are missing, those metrics will be skipped gracefully.

## Integration with Training

The metrics are automatically computed during validation. You don't need to modify your training command. The metrics will be:

1. Logged to your tracker (tensorboard/wandb) under the `val/` namespace
2. Saved as JSON and text files alongside validation videos
3. Printed to console for immediate feedback

## Notes

- Metrics computation adds ~5-10 seconds per validation video
- FID/VFID are more computationally expensive than frame-based metrics
- All metrics handle different video lengths by truncating to the shorter length
- Metrics are only computed when ground truth videos are available
- The generated frames are properly converted from PIL to numpy format before metric calculation
