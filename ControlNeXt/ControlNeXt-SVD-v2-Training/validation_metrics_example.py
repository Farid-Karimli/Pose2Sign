"""
Example usage of metrics for validation in train_svd.py

This file demonstrates how to use the metrics from metrics.py
for validation during training.
"""

import torch
from metrics import (
    calculate_ssim,
    calculate_psnr,
    calculate_lpips,
    calculate_mse,
    calculate_fid_from_videos,
    calculate_vfid,
    compare_videos,
    compare_multiple_video_pairs
)


def compute_validation_metrics(generated_frames, gt_frames, device='cuda'):
    """
    Compute all metrics for validation during training.

    Args:
        generated_frames: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        gt_frames: list of numpy arrays (H,W,C) in RGB, uint8 [0-255]
        device: device to run neural network models on ('cuda' or 'cpu')

    Returns:
        dict with all metrics
    """
    metrics = {}

    # Frame-based metrics (always computed)
    metrics['ssim'] = calculate_ssim(generated_frames, gt_frames)
    metrics['psnr'] = calculate_psnr(generated_frames, gt_frames)
    mse_per_frame, mean_mse = calculate_mse(generated_frames, gt_frames)
    metrics['mse'] = mean_mse

    # Perceptual metric (requires lpips)
    try:
        metrics['lpips'] = calculate_lpips(generated_frames, gt_frames, device=device)
    except Exception as e:
        print(f"LPIPS calculation failed: {e}")
        metrics['lpips'] = None

    # Distribution-based metrics (requires torchvision)
    try:
        metrics['fid'] = calculate_fid_from_videos(generated_frames, gt_frames, device=device)
    except Exception as e:
        print(f"FID calculation failed: {e}")
        metrics['fid'] = None

    try:
        metrics['vfid'] = calculate_vfid(generated_frames, gt_frames, device=device)
    except Exception as e:
        print(f"VFID calculation failed: {e}")
        metrics['vfid'] = None

    return metrics


def log_validation_metrics(metrics, step, logger=None, accelerator=None):
    """
    Log validation metrics to tensorboard/wandb.

    Args:
        metrics: dict with metric values
        step: current training step
        logger: logger instance (optional)
        accelerator: accelerator instance for logging (optional)
    """
    log_dict = {}

    # Always log frame-based metrics
    log_dict['val/ssim'] = metrics['ssim']
    log_dict['val/psnr'] = metrics['psnr']
    log_dict['val/mse'] = metrics['mse']

    # Log optional metrics if available
    if metrics.get('lpips') is not None:
        log_dict['val/lpips'] = metrics['lpips']

    if metrics.get('fid') is not None:
        log_dict['val/fid'] = metrics['fid']

    if metrics.get('vfid') is not None:
        log_dict['val/vfid'] = metrics['vfid']

    # Log to accelerator if available
    if accelerator is not None:
        accelerator.log(log_dict, step=step)

    # Print metrics
    if logger is not None:
        logger.info(f"Validation metrics at step {step}:")
        for key, value in log_dict.items():
            logger.info(f"  {key}: {value:.4f}")
    else:
        print(f"\nValidation metrics at step {step}:")
        for key, value in log_dict.items():
            print(f"  {key}: {value:.4f}")

    return log_dict


# Example integration with train_svd.py validation loop:
"""
# In train_svd.py, after generating validation videos:

# Convert generated PIL frames to numpy arrays
generated_frames_np = [np.array(frame) for frame in generated_frames]
gt_frames_np = [np.array(frame) for frame in gt_frames]

# Compute metrics
from validation_metrics_example import compute_validation_metrics, log_validation_metrics
metrics = compute_validation_metrics(generated_frames_np, gt_frames_np, device=accelerator.device)

# Log metrics
log_validation_metrics(metrics, global_step, logger=logger, accelerator=accelerator)
"""


if __name__ == "__main__":
    # Example: comparing videos from files
    print("Example 1: Compare two video files")
    inference_path = "path/to/generated_video.mp4"
    gt_path = "path/to/ground_truth_video.mp4"

    # This will compute all metrics and print results
    # results = compare_videos(inference_path, gt_path, device='cuda')

    print("\nExample 2: Compare multiple video pairs")
    video_pairs = [
        ("generated1.mp4", "gt1.mp4"),
        ("generated2.mp4", "gt2.mp4"),
        ("generated3.mp4", "gt3.mp4"),
    ]

    # This will compute metrics for all pairs and print summary
    # all_results = compare_multiple_video_pairs(video_pairs, device='cuda')

    print("\nSee the docstrings above for integration examples.")
