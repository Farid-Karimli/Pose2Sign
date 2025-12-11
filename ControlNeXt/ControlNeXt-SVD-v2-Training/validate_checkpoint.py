#!/usr/bin/env python
# coding=utf-8
"""Script to run validation on a trained checkpoint with pose metrics."""
import argparse
import logging
import os
import cv2
import re
import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from models.controlnext_vid_svd import ControlNeXtSDVModel
from safetensors.torch import load_file

# Import metrics functions
from metrics import (
    calculate_ssim,
    calculate_psnr,
    calculate_lpips,
    calculate_mse,
    calculate_fid_from_videos,
    calculate_vfid,
    LPIPS_AVAILABLE,
    INCEPTION_AVAILABLE
)

# Import pose faithfulness metrics
from pose_metrics import PoseExtractor

import json

logger = get_logger(__name__, log_level="INFO")

def validate_and_convert_image(image, target_size=(256, 256)):
    if image is None:
        print("Encountered a None image")
        return None

    if isinstance(image, torch.Tensor):
        # Convert PyTorch tensor to PIL Image
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Check for CxHxW format
            if image.shape[0] == 1:  # Convert single-channel grayscale to RGB
                image = image.repeat(3, 1, 1)
            image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            image = Image.fromarray(image)
        else:
            print(f"Invalid image tensor shape: {image.shape}")
            return None
    elif isinstance(image, Image.Image):
        # Resize PIL Image
        image = image.resize(target_size)
    else:
        print("Image is not a PIL Image or a PyTorch tensor")
        return None

    return image

def create_image_grid(images, rows, cols, target_size=(256, 256)):
    valid_images = [validate_and_convert_image(img, target_size) for img in images]
    valid_images = [img for img in valid_images if img is not None]

    if not valid_images:
        print("No valid images to create a grid")
        return None

    w, h = target_size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(valid_images):
        grid.paste(image, box=((i % cols) * w, (i // cols) * h))

    return grid

def write_mp4(video_path, samples, fps=14, audio_bitrate="192k"):
    from moviepy import ImageSequenceClip
    clip = ImageSequenceClip(samples, fps=fps)
    clip.write_videofile(video_path, audio_codec="aac", audio_bitrate=audio_bitrate,
                         ffmpeg_params=["-crf", "18", "-preset", "slow"], logger=None)

def save_vid_side_by_side(batch_output, validation_control_images, output_folder, fps, name):
    # Helper function to convert tensors to PIL images and save as GIF
    flattened_batch_output = [img for sublist in batch_output for img in sublist]
    video_path = output_folder + f'/{name}.mp4'
    final_images = []
    outputs = []
    # Helper function to concatenate images horizontally
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst
    for image_list in zip(validation_control_images, flattened_batch_output):
        predict_img = image_list[1].resize(image_list[0].size)
        result = get_concat_h(image_list[0], predict_img)
        final_images.append(np.array(result))
        outputs.append(np.array(predict_img))
    write_mp4(video_path, final_images, fps=fps)


def load_paired_validation_data(ref_frames_folder, pose_videos_folder, gt_videos_folder=None, num=None):
    """
    Load paired validation data where reference frames, pose videos, and ground truth videos have matching filenames.

    Args:
        ref_frames_folder: Path to folder containing reference frame images (e.g., PNG files)
        pose_videos_folder: Path to folder containing pose videos (e.g., MP4 files)
        gt_videos_folder: Path to folder containing ground truth ASL videos (e.g., MP4 files). Optional.
        num: Number of pairs to load (None = load all)

    Returns:
        List of tuples: [(ref_frame_pil, pose_frames_list, gt_frames_list), ...]
        where ref_frame_pil is a PIL Image, pose_frames_list is a list of PIL Images,
        and gt_frames_list is a list of PIL Images (or None if gt_videos_folder not provided)
    """
    paired_data = []

    # Valid extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

    # Get all files in ref_frames folder
    ref_files = {}
    for filename in os.listdir(ref_frames_folder):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in image_extensions:
            ref_files[basename] = filename

    # Get all files in pose_videos folder
    pose_files = {}
    for filename in os.listdir(pose_videos_folder):
        basename, ext = os.path.splitext(filename)
        if ext.lower() in video_extensions:
            pose_files[basename] = filename

    # Get all files in gt_videos folder if provided
    gt_files = {}
    if gt_videos_folder is not None and os.path.exists(gt_videos_folder):
        for filename in os.listdir(gt_videos_folder):
            basename, ext = os.path.splitext(filename)
            if ext.lower() in video_extensions:
                gt_files[basename] = filename

    # Find matching pairs (basenames that exist in both ref and pose folders)
    common_basenames = sorted(set(ref_files.keys()) & set(pose_files.keys()))

    # If GT folder provided, only keep basenames that also have GT videos
    if gt_videos_folder is not None and gt_files:
        common_basenames = sorted(set(common_basenames) & set(gt_files.keys()))
        logger.info(f"Found {len(common_basenames)} matching validation triplets (ref + pose + GT)")
    else:
        logger.info(f"Found {len(common_basenames)} matching validation pairs (ref + pose, no GT)")

    if len(common_basenames) == 0:
        logger.warning(f"No matching pairs found")
        return paired_data

    # Load each pair/triplet
    for basename in common_basenames[:num]:
        # Load reference frame
        ref_path = os.path.join(ref_frames_folder, ref_files[basename])
        ref_frame = Image.open(ref_path).convert('RGB')

        # Load pose video frames
        pose_path = os.path.join(pose_videos_folder, pose_files[basename])
        cap = cv2.VideoCapture(pose_path)
        pose_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_frames.append(Image.fromarray(frame))
        cap.release()

        # Load GT video frames if available
        gt_frames = None
        if gt_files and basename in gt_files:
            gt_path = os.path.join(gt_videos_folder, gt_files[basename])
            cap = cv2.VideoCapture(gt_path)
            gt_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gt_frames.append(Image.fromarray(frame))
            cap.release()

        if len(pose_frames) > 0:
            gt_info = f" + {len(gt_frames)} GT frames" if gt_frames else " (no GT)"
            paired_data.append((ref_frame, pose_frames, gt_frames, basename))
            logger.info(f"Loaded '{basename}': 1 ref frame + {len(pose_frames)} pose frames{gt_info}")
        else:
            logger.warning(f"Skipping '{basename}': pose video has no frames")

    return paired_data


def compute_video_metrics(generated_frames, gt_frames, device='cuda'):
    """
    Compute all validation metrics between generated and ground truth frames.

    Args:
        generated_frames: list of PIL Images
        gt_frames: list of PIL Images
        device: device to run neural network models on

    Returns:
        dict with all computed metrics
    """
    # Convert PIL images to numpy arrays (H, W, C) in RGB, uint8 [0-255]
    generated_np = [np.array(frame) for frame in generated_frames]
    gt_np = [np.array(frame) for frame in gt_frames]

    # Make sure they have the same number of frames
    min_frames = min(len(generated_np), len(gt_np))
    if len(generated_np) != len(gt_np):
        logger.warning(f"Frame count mismatch: generated={len(generated_np)}, gt={len(gt_np)}. Using first {min_frames} frames.")
        generated_np = generated_np[:min_frames]
        gt_np = gt_np[:min_frames]

    metrics = {}

    # Frame-based metrics (always computed)
    try:
        metrics['ssim'] = float(calculate_ssim(generated_np, gt_np))
    except Exception as e:
        logger.warning(f"SSIM calculation failed: {e}")
        metrics['ssim'] = None

    try:
        metrics['psnr'] = float(calculate_psnr(generated_np, gt_np))
    except Exception as e:
        logger.warning(f"PSNR calculation failed: {e}")
        metrics['psnr'] = None

    try:
        mse_per_frame, mean_mse = calculate_mse(generated_np, gt_np)
        metrics['mse'] = float(mean_mse)
    except Exception as e:
        logger.warning(f"MSE calculation failed: {e}")
        metrics['mse'] = None

    # Perceptual metric (requires lpips)
    if LPIPS_AVAILABLE:
        try:
            metrics['lpips'] = float(calculate_lpips(generated_np, gt_np, device=device))
        except Exception as e:
            logger.warning(f"LPIPS calculation failed: {e}")
            metrics['lpips'] = None
    else:
        metrics['lpips'] = None

    # Distribution-based metrics (requires torchvision)
    if INCEPTION_AVAILABLE:
        try:
            metrics['fid'] = float(calculate_fid_from_videos(generated_np, gt_np, device=device))
        except Exception as e:
            logger.warning(f"FID calculation failed: {e}")
            metrics['fid'] = None

        try:
            metrics['vfid'] = float(calculate_vfid(generated_np, gt_np, device=device))
        except Exception as e:
            logger.warning(f"VFID calculation failed: {e}")
            metrics['vfid'] = None
    else:
        metrics['fid'] = None
        metrics['vfid'] = None

    return metrics


def compute_pose_faithfulness_metrics(pose_frames, generated_frames, body_pck_threshold=0.15,
                                       hands_pck_threshold=0.05, face_pck_threshold=0.05, verbose=False):
    """
    Compute pose faithfulness metrics between pose conditioning frames and generated frames.

    Args:
        pose_frames: list of PIL Images (pose conditioning)
        generated_frames: list of PIL Images (generated output)
        body_pck_threshold: threshold for body PCK metric (default 0.15 for normalized coordinates)
        hands_pck_threshold: threshold for hands PCK metric (default 0.05 for normalized coordinates)
        face_pck_threshold: threshold for face PCK metric (default 0.05 for normalized coordinates)
        verbose: print detailed progress

    Returns:
        dict with pose faithfulness metrics
    """
    try:
        # Initialize pose extractor
        extractor = PoseExtractor()

        # Extract keypoints from both videos
        pose_kps = []
        gen_kps = []

        for pose_frame, gen_frame in zip(pose_frames, generated_frames):
            # Convert PIL to numpy
            pose_np = np.array(pose_frame)
            gen_np = np.array(gen_frame)

            # Extract keypoints
            pose_kp = extractor.extract_from_frame(pose_np)
            gen_kp = extractor.extract_from_frame(gen_np)

            pose_kps.append(pose_kp)
            gen_kps.append(gen_kp)

        extractor.close()

        # Calculate frame-wise metrics
        from pose_metrics import (
            calculate_mpjpe,
            calculate_procrustes_mpjpe,
            calculate_pck,
            calculate_temporal_smoothness,
            align_hand_keypoints
        )

        body_mpjpe_list = []
        body_pmpjpe_list = []  # Procrustes-MPJPE
        body_pck_list = []
        hands_mpjpe_list = []
        hands_pmpjpe_list = []
        hands_pck_list = []
        face_mpjpe_list = []
        face_pmpjpe_list = []
        face_pck_list = []

        for pose_kp, gen_kp in zip(pose_kps, gen_kps):
            # Body metrics (use GT visibility from pose_kp)
            if pose_kp['body'] is not None and gen_kp['body'] is not None:
                body_mpjpe_list.append(calculate_mpjpe(pose_kp['body'], gen_kp['body']))
                body_pmpjpe_list.append(calculate_procrustes_mpjpe(pose_kp['body'], gen_kp['body']))
                body_pck_list.append(calculate_pck(pose_kp['body'], gen_kp['body'], body_pck_threshold))

            # Hand metrics
            pose_hands, gen_hands = align_hand_keypoints(pose_kp['hands'], gen_kp['hands'])
            if pose_hands is not None and gen_hands is not None:
                hands_mpjpe_list.append(calculate_mpjpe(pose_hands, gen_hands, use_visibility=False))
                hands_pmpjpe_list.append(calculate_procrustes_mpjpe(pose_hands, gen_hands, use_visibility=False))
                hands_pck_list.append(calculate_pck(pose_hands, gen_hands, hands_pck_threshold, use_visibility=False))

            # Face metrics
            if pose_kp['face'] is not None and gen_kp['face'] is not None:
                face_mpjpe_list.append(calculate_mpjpe(pose_kp['face'], gen_kp['face'], use_visibility=False))
                face_pmpjpe_list.append(calculate_procrustes_mpjpe(pose_kp['face'], gen_kp['face'], use_visibility=False))
                face_pck_list.append(calculate_pck(pose_kp['face'], gen_kp['face'], face_pck_threshold, use_visibility=False))

        # Calculate temporal smoothness
        body_smoothness = calculate_temporal_smoothness([kp['body'] for kp in gen_kps])
        hands_smoothness = calculate_temporal_smoothness(
            [np.concatenate(kp['hands']) if kp['hands'] and len(kp['hands']) > 0 else None for kp in gen_kps]
        )

        # Compile results
        pose_metrics = {
            'body_mpjpe': float(np.nanmean(body_mpjpe_list)) if body_mpjpe_list else None,
            'body_pmpjpe': float(np.nanmean(body_pmpjpe_list)) if body_pmpjpe_list else None,
            'body_pck': float(np.nanmean(body_pck_list)) if body_pck_list else None,
            'body_smoothness': float(body_smoothness) if not np.isnan(body_smoothness) else None,
            'hands_mpjpe': float(np.nanmean(hands_mpjpe_list)) if hands_mpjpe_list else None,
            'hands_pmpjpe': float(np.nanmean(hands_pmpjpe_list)) if hands_pmpjpe_list else None,
            'hands_pck': float(np.nanmean(hands_pck_list)) if hands_pck_list else None,
            'hands_smoothness': float(hands_smoothness) if not np.isnan(hands_smoothness) else None,
            'face_mpjpe': float(np.nanmean(face_mpjpe_list)) if face_mpjpe_list else None,
            'face_pmpjpe': float(np.nanmean(face_pmpjpe_list)) if face_pmpjpe_list else None,
            'face_pck': float(np.nanmean(face_pck_list)) if face_pck_list else None,
        }

        # Calculate combined weighted score (hands: 50%, body: 30%, face: 20%)
        weights = {'body': 0.3, 'hands': 0.5, 'face': 0.2}
        combined_mpjpe = 0
        combined_pmpjpe = 0
        combined_pck = 0
        total_weight = 0

        for component in ['body', 'hands', 'face']:
            mpjpe_key = f'{component}_mpjpe'
            pmpjpe_key = f'{component}_pmpjpe'
            pck_key = f'{component}_pck'

            if pose_metrics.get(mpjpe_key) is not None:
                combined_mpjpe += pose_metrics[mpjpe_key] * weights[component]
                total_weight += weights[component]
            if pose_metrics.get(pmpjpe_key) is not None:
                combined_pmpjpe += pose_metrics[pmpjpe_key] * weights[component]
            if pose_metrics.get(pck_key) is not None:
                combined_pck += pose_metrics[pck_key] * weights[component]

        pose_metrics['combined_mpjpe'] = float(combined_mpjpe / total_weight) if total_weight > 0 else None
        pose_metrics['combined_pmpjpe'] = float(combined_pmpjpe / total_weight) if total_weight > 0 else None
        pose_metrics['combined_pck'] = float(combined_pck / total_weight) if total_weight > 0 else None

        return pose_metrics

    except Exception as e:
        logger.warning(f"Pose faithfulness calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'body_mpjpe': None,
            'body_pmpjpe': None,
            'body_pck': None,
            'body_smoothness': None,
            'hands_mpjpe': None,
            'hands_pmpjpe': None,
            'hands_pck': None,
            'hands_smoothness': None,
            'face_mpjpe': None,
            'face_pmpjpe': None,
            'face_pck': None,
            'combined_mpjpe': None,
            'combined_pmpjpe': None,
            'combined_pck': None,
        }


def save_metrics_report(all_video_metrics, save_dir, checkpoint_name):
    """
    Save a comprehensive metrics report as JSON and text.

    Args:
        all_video_metrics: list of dicts, each containing metrics for one video pair
        save_dir: directory to save the report
        checkpoint_name: name of the checkpoint being validated
    """
    # Calculate averages across all videos
    metrics_summary = {
        'checkpoint': checkpoint_name,
        'num_videos': len(all_video_metrics),
        'per_video_metrics': all_video_metrics,
        'average_metrics': {}
    }

    # Compute averages for each metric
    metric_names = [
        'ssim', 'psnr', 'mse', 'lpips', 'fid', 'vfid',
        'body_mpjpe', 'body_pmpjpe', 'body_pck', 'body_smoothness',
        'hands_mpjpe', 'hands_pmpjpe', 'hands_pck', 'hands_smoothness',
        'face_mpjpe', 'face_pmpjpe', 'face_pck',
        'combined_mpjpe', 'combined_pmpjpe', 'combined_pck'
    ]
    for metric_name in metric_names:
        values = [v[metric_name] for v in all_video_metrics if v.get(metric_name) is not None]
        if values:
            metrics_summary['average_metrics'][metric_name] = float(np.mean(values))
        else:
            metrics_summary['average_metrics'][metric_name] = None

    # Save JSON report
    json_path = os.path.join(save_dir, f"metrics_{checkpoint_name}.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # Save human-readable text report
    txt_path = os.path.join(save_dir, f"metrics_{checkpoint_name}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"VALIDATION METRICS REPORT - {checkpoint_name}\n")
        f.write(f"=" * 80 + "\n\n")

        f.write(f"Number of validation videos: {len(all_video_metrics)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("AVERAGE METRICS ACROSS ALL VIDEOS\n")
        f.write("-" * 80 + "\n")

        avg_metrics = metrics_summary['average_metrics']

        # Image quality metrics
        f.write("IMAGE QUALITY METRICS:\n")
        if avg_metrics.get('ssim') is not None:
            f.write(f"  SSIM:  {avg_metrics['ssim']:.4f}\n")
        if avg_metrics.get('psnr') is not None:
            f.write(f"  PSNR:  {avg_metrics['psnr']:.2f} dB\n")
        if avg_metrics.get('mse') is not None:
            f.write(f"  MSE:   {avg_metrics['mse']:.6f}\n")
        if avg_metrics.get('lpips') is not None:
            f.write(f"  LPIPS: {avg_metrics['lpips']:.4f}\n")
        if avg_metrics.get('fid') is not None:
            f.write(f"  FID:   {avg_metrics['fid']:.4f}\n")
        if avg_metrics.get('vfid') is not None:
            f.write(f"  VFID:  {avg_metrics['vfid']:.4f}\n")

        # Pose faithfulness metrics
        f.write("\nPOSE FAITHFULNESS METRICS:\n")
        if avg_metrics.get('combined_mpjpe') is not None:
            f.write(f"  Combined MPJPE:   {avg_metrics['combined_mpjpe']:.6f}\n")
        if avg_metrics.get('combined_pmpjpe') is not None:
            f.write(f"  Combined P-MPJPE: {avg_metrics['combined_pmpjpe']:.6f}\n")
        if avg_metrics.get('combined_pck') is not None:
            f.write(f"  Combined PCK:     {avg_metrics['combined_pck']:.4f} ({avg_metrics['combined_pck']*100:.2f}%)\n")

        f.write("\n  Body Pose (PCK@0.15):\n")
        if avg_metrics.get('body_mpjpe') is not None:
            f.write(f"    MPJPE:      {avg_metrics['body_mpjpe']:.6f}\n")
        if avg_metrics.get('body_pmpjpe') is not None:
            f.write(f"    P-MPJPE:    {avg_metrics['body_pmpjpe']:.6f}\n")
        if avg_metrics.get('body_pck') is not None:
            f.write(f"    PCK:        {avg_metrics['body_pck']:.4f} ({avg_metrics['body_pck']*100:.2f}%)\n")
        if avg_metrics.get('body_smoothness') is not None:
            f.write(f"    Smoothness: {avg_metrics['body_smoothness']:.6f}\n")

        f.write("\n  Hand Pose (PCK@0.05):\n")
        if avg_metrics.get('hands_mpjpe') is not None:
            f.write(f"    MPJPE:      {avg_metrics['hands_mpjpe']:.6f}\n")
        if avg_metrics.get('hands_pmpjpe') is not None:
            f.write(f"    P-MPJPE:    {avg_metrics['hands_pmpjpe']:.6f}\n")
        if avg_metrics.get('hands_pck') is not None:
            f.write(f"    PCK:        {avg_metrics['hands_pck']:.4f} ({avg_metrics['hands_pck']*100:.2f}%)\n")
        if avg_metrics.get('hands_smoothness') is not None:
            f.write(f"    Smoothness: {avg_metrics['hands_smoothness']:.6f}\n")

        f.write("\n  Face Pose (PCK@0.05):\n")
        if avg_metrics.get('face_mpjpe') is not None:
            f.write(f"    MPJPE:      {avg_metrics['face_mpjpe']:.6f}\n")
        if avg_metrics.get('face_pmpjpe') is not None:
            f.write(f"    P-MPJPE:    {avg_metrics['face_pmpjpe']:.6f}\n")
        if avg_metrics.get('face_pck') is not None:
            f.write(f"    PCK:        {avg_metrics['face_pck']:.4f} ({avg_metrics['face_pck']*100:.2f}%)\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("PER-VIDEO METRICS\n")
        f.write("-" * 80 + "\n\n")

        for i, video_metrics in enumerate(all_video_metrics):
            f.write(f"Video {i} ({video_metrics.get('basename', 'unknown')}):\n")
            if video_metrics.get('ssim') is not None:
                f.write(f"  SSIM:  {video_metrics['ssim']:.4f}\n")
            if video_metrics.get('psnr') is not None:
                f.write(f"  PSNR:  {video_metrics['psnr']:.2f} dB\n")
            if video_metrics.get('mse') is not None:
                f.write(f"  MSE:   {video_metrics['mse']:.6f}\n")
            if video_metrics.get('lpips') is not None:
                f.write(f"  LPIPS: {video_metrics['lpips']:.4f}\n")
            if video_metrics.get('fid') is not None:
                f.write(f"  FID:   {video_metrics['fid']:.4f}\n")
            if video_metrics.get('vfid') is not None:
                f.write(f"  VFID:  {video_metrics['vfid']:.4f}\n")

            # Add pose metrics
            if video_metrics.get('combined_pck') is not None:
                f.write(f"  Combined PCK: {video_metrics['combined_pck']:.4f}\n")
            if video_metrics.get('hands_pck') is not None:
                f.write(f"  Hands PCK: {video_metrics['hands_pck']:.4f}\n")
            if video_metrics.get('body_pck') is not None:
                f.write(f"  Body PCK: {video_metrics['body_pck']:.4f}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    logger.info(f"Saved metrics report to {json_path} and {txt_path}")

    return metrics_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run validation on a trained checkpoint")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory to validate (e.g., outputs/checkpoint-last)",
    )
    parser.add_argument(
        "--validation_base",
        type=str,
        default="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation",
        help="Base path to validation data folders (ref_frames, pose, asl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save validation outputs (defaults to checkpoint_path/validation_output)",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=None,
        help="Number of validation videos to process (None = all)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of generated videos",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of generated videos",
    )
    parser.add_argument(
        "--sample_n_frames",
        type=int,
        default=14,
        help="Number of frames per batch during inference",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_path, "validation_output")
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_name = os.path.basename(args.checkpoint_path)
    logger.info(f"Validating checkpoint: {checkpoint_name}")
    logger.info(f"Output directory: {args.output_dir}")

    # Determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load models
    logger.info("Loading base models...")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )

    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )

    controlnext = ControlNeXtSDVModel()

    # Load checkpoint weights
    logger.info(f"Loading checkpoint from {args.checkpoint_path}...")

    # Check if this is an unwrapped checkpoint with separate unet and controlnext dirs
    unet_path = os.path.join(args.checkpoint_path, "unet", "diffusion_pytorch_model.bin")
    controlnext_path = os.path.join(args.checkpoint_path, "controlnext", "diffusion_pytorch_model.bin")

    if os.path.exists(unet_path) and os.path.exists(controlnext_path):
        # Load from unwrapped checkpoint (like inference.py does)
        logger.info(f"Loading UNet weights from {unet_path}")
        unet_state_dict = torch.load(unet_path, map_location="cpu")
        unet.load_state_dict(unet_state_dict, strict=False)

        logger.info(f"Loading ControlNeXt weights from {controlnext_path}")
        controlnext_state_dict = torch.load(controlnext_path, map_location="cpu")
        controlnext.load_state_dict(controlnext_state_dict, strict=False)
    else:
        # Try loading using accelerator.load_state (for DeepSpeed checkpoints)
        logger.info("Unwrapped weights not found, trying accelerator.load_state...")
        accelerator.load_state(args.checkpoint_path)

    # Move models to device
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnext.to(accelerator.device, dtype=weight_dtype)

    # Set to eval mode
    image_encoder.eval()
    vae.eval()
    unet.eval()
    controlnext.eval()

    # Create pipeline
    logger.info("Creating inference pipeline...")
    pipeline = StableVideoDiffusionPipelineControlNeXt.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        controlnext=controlnext,
        image_encoder=image_encoder,
        vae=vae,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=False)

    # Load validation data
    logger.info("Loading validation data...")
    validation_pairs = load_paired_validation_data(
        ref_frames_folder=os.path.join(args.validation_base, "ref_frames"),
        pose_videos_folder=os.path.join(args.validation_base, "pose"),
        gt_videos_folder=os.path.join(args.validation_base, "asl"),
        num=args.num_validation_images
    )
    logger.info(f"Loaded {len(validation_pairs)} paired validation samples")

    if len(validation_pairs) == 0:
        logger.error("No validation pairs found! Exiting.")
        return

    # Run validation
    logger.info("Running validation...")
    all_video_metrics = []

    save_dir = os.path.join(args.output_dir, f"validation_{checkpoint_name}_vids")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, (val_ref_frame, val_control_frames, val_gt_frames, basename) in enumerate(validation_pairs):
            num_frames = len(val_control_frames)
            logger.info(f"Generating video {i+1}/{len(validation_pairs)} ({basename}) with {num_frames} frames")

            video_frames = pipeline(
                val_ref_frame,  # Starting reference image
                val_control_frames,  # Conditioning control images (pose)
                height=args.height,
                width=args.width,
                num_frames=num_frames,
                frames_per_batch=args.sample_n_frames,
                decode_chunk_size=4,
                motion_bucket_id=127.,
                fps=30,
                controlnext_cond_scale=1.0,
                min_guidance_scale=3,
                max_guidance_scale=3,
                noise_aug_strength=0.02,
                num_inference_steps=25,
                overlap=4,
            ).frames

            # Save video
            save_vid_side_by_side(
                video_frames,
                val_control_frames,
                save_dir,
                fps=7,
                name=f"v{i}_{basename}"
            )

            # Compute metrics if ground truth is available
            if val_gt_frames is not None:
                logger.info(f"Computing metrics for video {i} ({basename})")

                # Flatten generated frames (video_frames is list of lists)
                generated_frames_flat = [img for sublist in video_frames for img in sublist]

                # Compute image quality metrics
                video_metrics = compute_video_metrics(
                    generated_frames_flat,
                    val_gt_frames,
                    device=accelerator.device
                )

                # Compute pose faithfulness metrics
                logger.info(f"Computing pose faithfulness for video {i} ({basename})")
                pose_metrics = compute_pose_faithfulness_metrics(
                    val_control_frames,
                    generated_frames_flat
                    # Uses default thresholds: body_pck_threshold=0.15, hands/face=0.05
                )

                # Merge metrics
                video_metrics.update(pose_metrics)
                video_metrics['video_index'] = i
                video_metrics['basename'] = basename
                all_video_metrics.append(video_metrics)

                # Log metrics for this video
                logger.info(f"Video {i} metrics:")
                if video_metrics.get('ssim') is not None:
                    logger.info(f"  SSIM: {video_metrics['ssim']:.4f}")
                if video_metrics.get('psnr') is not None:
                    logger.info(f"  PSNR: {video_metrics['psnr']:.2f} dB")
                if video_metrics.get('combined_pck') is not None:
                    logger.info(f"  Combined PCK: {video_metrics['combined_pck']:.4f} ({video_metrics['combined_pck']*100:.2f}%)")
                if video_metrics.get('hands_pck') is not None:
                    logger.info(f"  Hands PCK: {video_metrics['hands_pck']:.4f} ({video_metrics['hands_pck']*100:.2f}%)")
            else:
                logger.warning(f"No ground truth available for video {i} ({basename}), skipping metrics")

    # Save comprehensive metrics report
    if all_video_metrics:
        logger.info("Saving metrics report...")
        metrics_summary = save_metrics_report(all_video_metrics, args.output_dir, checkpoint_name)

        # Print summary
        logger.info("=" * 80)
        logger.info(f"VALIDATION SUMMARY - {checkpoint_name}")
        logger.info("=" * 80)
        logger.info(f"Evaluated {len(all_video_metrics)} videos")

        avg_metrics = metrics_summary['average_metrics']

        logger.info("\nAverage Image Quality Metrics:")
        for metric_name in ['ssim', 'psnr', 'mse', 'lpips', 'fid', 'vfid']:
            value = avg_metrics.get(metric_name)
            if value is not None:
                logger.info(f"  {metric_name.upper()}: {value:.4f}")

        logger.info("\nAverage Pose Faithfulness Metrics:")
        if avg_metrics.get('combined_pck') is not None:
            logger.info(f"  Combined PCK: {avg_metrics['combined_pck']:.4f} ({avg_metrics['combined_pck']*100:.2f}%)")
        if avg_metrics.get('combined_mpjpe') is not None:
            logger.info(f"  Combined MPJPE: {avg_metrics['combined_mpjpe']:.6f}")
        if avg_metrics.get('hands_pck') is not None:
            logger.info(f"  Hands PCK: {avg_metrics['hands_pck']:.4f} ({avg_metrics['hands_pck']*100:.2f}%)")
        if avg_metrics.get('body_pck') is not None:
            logger.info(f"  Body PCK: {avg_metrics['body_pck']:.4f} ({avg_metrics['body_pck']*100:.2f}%)")

        logger.info("=" * 80)
    else:
        logger.warning("No ground truth videos available - metrics not computed")

    logger.info(f"Validation complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
