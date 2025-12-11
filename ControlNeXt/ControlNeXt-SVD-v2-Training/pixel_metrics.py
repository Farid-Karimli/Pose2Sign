#!/usr/bin/env python
"""
Pixel-based metrics for video comparison: MSE, PSNR, SSIM
Includes region-specific metrics for hands and face.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Mean Squared Error between two images."""
    if img1.shape != img2.shape:
        return np.nan
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio."""
    mse = calculate_mse(img1, img2)
    if mse == 0 or np.isnan(mse):
        return 100.0  # Perfect match
    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate Structural Similarity Index."""
    if img1.shape != img2.shape:
        return np.nan

    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2

    return ssim(img1_gray, img2_gray)


def extract_frame_at_timestamp(video_path: str, timestamp: float) -> Optional[np.ndarray]:
    """Extract a single frame at a specific timestamp."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def get_hand_bounding_box(hand_landmarks, image_shape: Tuple[int, int], padding: float = 0.1) -> Optional[Tuple[int, int, int, int]]:
    """
    Get bounding box around hand landmarks with padding.

    Args:
        hand_landmarks: MediaPipe hand landmarks (21 points)
        image_shape: (height, width) of image
        padding: Percentage of bbox size to add as padding

    Returns:
        (x1, y1, x2, y2) or None
    """
    if hand_landmarks is None or len(hand_landmarks) == 0:
        return None

    h, w = image_shape[:2]

    # Extract x, y coordinates (normalized 0-1)
    xs = hand_landmarks[:, 0]
    ys = hand_landmarks[:, 1]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Add padding
    width = x_max - x_min
    height = y_max - y_min
    x_min = max(0, x_min - width * padding)
    x_max = min(1, x_max + width * padding)
    y_min = max(0, y_min - height * padding)
    y_max = min(1, y_max + height * padding)

    # Convert to pixel coordinates
    x1 = int(x_min * w)
    y1 = int(y_min * h)
    x2 = int(x_max * w)
    y2 = int(y_max * h)

    return (x1, y1, x2, y2)


def evaluate_pixel_metrics(
    video1_path: str,
    video2_path: str,
    sample_timestamps: Optional[List[float]] = None,
    verbose: bool = True
) -> Dict:
    """
    Evaluate pixel-level metrics between two videos.

    Args:
        video1_path: Path to first video (GT)
        video2_path: Path to second video (generated)
        sample_timestamps: If provided, only evaluate at these timestamps
        verbose: Print progress

    Returns:
        Dictionary with MSE, PSNR, SSIM metrics
    """
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("Cannot open one or both videos")

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    # If no timestamps provided, sample both videos at min FPS
    if sample_timestamps is None:
        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        duration1 = frame_count1 / fps1
        duration2 = frame_count2 / fps2
        min_duration = min(duration1, duration2)
        sample_fps = min(fps1, fps2)
        num_samples = int(min_duration * sample_fps)
        sample_timestamps = [i / sample_fps for i in range(num_samples)]

    cap1.release()
    cap2.release()

    # Calculate metrics at each timestamp
    mse_list = []
    psnr_list = []
    ssim_list = []

    iterator = tqdm(sample_timestamps, desc="Computing pixel metrics") if verbose else sample_timestamps

    for timestamp in iterator:
        frame1 = extract_frame_at_timestamp(video1_path, timestamp)
        frame2 = extract_frame_at_timestamp(video2_path, timestamp)

        if frame1 is None or frame2 is None:
            continue

        # Resize if needed
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

        mse = calculate_mse(frame1, frame2)
        psnr = calculate_psnr(frame1, frame2)
        ssim_val = calculate_ssim(frame1, frame2)

        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim_val)

    results = {
        'num_frames': len(mse_list),
        'mse': {
            'mean': np.mean(mse_list) if mse_list else np.nan,
            'std': np.std(mse_list) if mse_list else np.nan,
            'min': np.min(mse_list) if mse_list else np.nan,
            'max': np.max(mse_list) if mse_list else np.nan,
        },
        'psnr': {
            'mean': np.mean(psnr_list) if psnr_list else np.nan,
            'std': np.std(psnr_list) if psnr_list else np.nan,
            'min': np.min(psnr_list) if psnr_list else np.nan,
            'max': np.max(psnr_list) if psnr_list else np.nan,
        },
        'ssim': {
            'mean': np.mean(ssim_list) if ssim_list else np.nan,
            'std': np.std(ssim_list) if ssim_list else np.nan,
            'min': np.min(ssim_list) if ssim_list else np.nan,
            'max': np.max(ssim_list) if ssim_list else np.nan,
        }
    }

    if verbose:
        print("\n" + "="*60)
        print("PIXEL-LEVEL METRICS")
        print("="*60)
        print(f"Frames evaluated: {results['num_frames']}")
        print(f"\nMSE: {results['mse']['mean']:.2f} ± {results['mse']['std']:.2f}")
        print(f"PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
        print(f"SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
        print("="*60)

    return results


if __name__ == "__main__":
    # Example usage
    gt_video = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/asl/026756336707836725-LIMITED.mp4"
    gen_video = "/restricted/projectnb/cs599dg/faridkar/Pose2Sign/ControlNeXt/ControlNeXt-SVD-v2-Training/OUTPUTS-FINAL/validation_images/validation_44000_vids/v0_026756336707836725-LIMITED.mp4"

    results = evaluate_pixel_metrics(gt_video, gen_video, verbose=True)

    # Save results
    with open("pixel_metrics_results.json", 'w') as f:
        json.dump(results, f, indent=2)
