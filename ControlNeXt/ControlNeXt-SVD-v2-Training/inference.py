#!/usr/bin/env python
# coding=utf-8
"""
Inference script for ControlNeXt-SVD model.

This script performs inference on a single pose video using a trained ControlNeXt model.
It automatically loads the latest checkpoint and finds the corresponding reference frame.

Usage:
    python inference.py --pose_video /path/to/pose/video.mp4
    python inference.py --pose_video /path/to/pose/video.mp4 --checkpoint checkpoint-40000
    python inference.py --pose_video /path/to/pose/video.mp4 --output_dir ./inference_output
"""

import argparse
import logging
import os
import cv2
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers import AutoencoderKLTemporalDecoder, EulerDiscreteScheduler
from models.unet_spatio_temporal_condition_controlnext import UNetSpatioTemporalConditionControlNeXtModel
from models.controlnext_vid_svd import ControlNeXtSDVModel
from pipeline.pipeline_stable_video_diffusion_controlnext import StableVideoDiffusionPipelineControlNeXt
from safetensors.torch import load_file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_logger(name, log_level="INFO"):
    """Get a logger with the specified name and level."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))
    return logger


def find_latest_checkpoint(output_dir):
    """
    Find the latest checkpoint in the output directory.

    Args:
        output_dir: Directory containing checkpoint folders

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        return None

    dirs = os.listdir(output_dir)
    dirs = [d for d in dirs if d.startswith("checkpoint-")]
    if len(dirs) == 0:
        return None

    # Sort by checkpoint number
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
    latest = dirs[-1]

    logger.info(f"Found latest checkpoint: {latest}")
    return os.path.join(output_dir, latest)


def load_checkpoint_models(checkpoint_dir, pretrained_model_name_or_path, device, weight_dtype):
    """
    Load UNet and ControlNeXt models from a checkpoint directory.

    According to README.md, checkpoints saved with DeepSpeed need to be:
    1. Converted to .bin using zero_to_fp32.py (creates pytorch_model.bin)
    2. Unwrapped using unwrap_deepspeed.py to separate unet and controlnext weights

    Args:
        checkpoint_dir: Path to checkpoint directory
        pretrained_model_name_or_path: Base model path for loading other components
        device: Device to load models on
        weight_dtype: Data type for model weights

    Returns:
        Tuple of (unet, controlnext, image_encoder, vae, feature_extractor)
    """
    logger.info(f"Loading checkpoint from {checkpoint_dir}")

    # Check if unwrapped weights exist
    unet_path = os.path.join(checkpoint_dir, "unet", "diffusion_pytorch_model.bin")
    controlnext_path = os.path.join(checkpoint_dir, "controlnext", "diffusion_pytorch_model.bin")

    if not os.path.exists(unet_path) or not os.path.exists(controlnext_path):
        logger.warning(f"Unwrapped weights not found in {checkpoint_dir}")
        logger.info("Looking for pytorch_model.bin (converted from DeepSpeed)...")

        pytorch_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            logger.info("Found pytorch_model.bin. You need to unwrap it first:")
            logger.info(f"  python utils/unwrap_deepspeed.py {checkpoint_dir}")
            raise FileNotFoundError(
                f"Please unwrap the checkpoint first using utils/unwrap_deepspeed.py\n"
                f"Run: python utils/unwrap_deepspeed.py {checkpoint_dir}"
            )
        else:
            logger.info("Looking for DeepSpeed checkpoint files...")
            # Check if this is a raw DeepSpeed checkpoint
            if os.path.exists(os.path.join(checkpoint_dir, "zero_to_fp32.py")):
                logger.info("Found DeepSpeed checkpoint. You need to convert it first:")
                logger.info(f"  1. cd {checkpoint_dir}")
                logger.info(f"  2. python zero_to_fp32.py . pytorch_model.bin")
                logger.info(f"  3. cd -")
                logger.info(f"  4. python utils/unwrap_deepspeed.py {checkpoint_dir}")
                raise FileNotFoundError(
                    f"Please convert and unwrap the DeepSpeed checkpoint first:\n"
                    f"  1. cd {checkpoint_dir}\n"
                    f"  2. python zero_to_fp32.py . pytorch_model.bin\n"
                    f"  3. cd -\n"
                    f"  4. python utils/unwrap_deepspeed.py {checkpoint_dir}"
                )
            else:
                raise FileNotFoundError(
                    f"Could not find model weights in {checkpoint_dir}\n"
                    f"Expected either:\n"
                    f"  - {unet_path} and {controlnext_path} (unwrapped weights)\n"
                    f"  - {pytorch_model_path} (converted DeepSpeed weights)\n"
                    f"  - DeepSpeed checkpoint files with zero_to_fp32.py"
                )

    # Load base models
    logger.info(f"Loading base models from {pretrained_model_name_or_path}")

    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path, subfolder="image_encoder"
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNetSpatioTemporalConditionControlNeXtModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        low_cpu_mem_usage=True,
    )

    # Load ControlNeXt
    logger.info("Initializing ControlNeXt")
    controlnext = ControlNeXtSDVModel()

    # Load checkpoint weights
    logger.info(f"Loading UNet weights from {unet_path}")
    unet_state_dict = torch.load(unet_path, map_location="cpu")
    unet.load_state_dict(unet_state_dict, strict=False)

    logger.info(f"Loading ControlNeXt weights from {controlnext_path}")
    controlnext_state_dict = torch.load(controlnext_path, map_location="cpu")
    controlnext.load_state_dict(controlnext_state_dict, strict=False)

    # Move models to device with correct dtype
    image_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    controlnext.to(device, dtype=weight_dtype)

    return unet, controlnext, image_encoder, vae, feature_extractor


def load_video_frames(video_path):
    """
    Load all frames from a video file.

    Args:
        video_path: Path to video file

    Returns:
        List of PIL Image objects in RGB format
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()
    logger.info(f"Loaded {frame_count} frames from {video_path}")

    if frame_count == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    return frames


def find_reference_frame(pose_video_path, ref_frames_dir="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames"):
    """
    Find the reference frame corresponding to a pose video.

    The reference frame should have the same basename but with .png extension.

    Args:
        pose_video_path: Path to the pose video
        ref_frames_dir: Directory containing reference frames

    Returns:
        PIL Image of the reference frame
    """
    # Get basename without extension
    basename = os.path.splitext(os.path.basename(pose_video_path))[0]

    # Look for corresponding reference frame
    ref_frame_path = os.path.join(ref_frames_dir, f"{basename}.png")

    if not os.path.exists(ref_frame_path):
        # Try .jpg extension
        ref_frame_path = os.path.join(ref_frames_dir, f"{basename}.jpg")

    if not os.path.exists(ref_frame_path):
        raise FileNotFoundError(
            f"Could not find reference frame for '{basename}' in {ref_frames_dir}\n"
            f"Expected: {os.path.join(ref_frames_dir, basename + '.png')} or .jpg"
        )

    logger.info(f"Found reference frame: {ref_frame_path}")
    return Image.open(ref_frame_path).convert('RGB')


def save_video(frames, output_path, fps=7):
    """
    Save a list of PIL Images as a video file.

    Args:
        frames: List of PIL Image objects
        output_path: Path to save the video
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")

    # Get dimensions from first frame
    first_frame = np.array(frames[0])
    h, w = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Write frames
    for frame in frames:
        frame_np = np.array(frame)
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    logger.info(f"Saved video to {output_path}")


def create_side_by_side_frames(pose_frames, generated_frames):
    """
    Create side-by-side frames from pose and generated frames.

    Args:
        pose_frames: List of PIL Image objects (pose video)
        generated_frames: List of PIL Image objects (generated video)

    Returns:
        List of PIL Image objects with frames concatenated horizontally
    """
    if len(pose_frames) != len(generated_frames):
        logger.warning(
            f"Frame count mismatch: pose={len(pose_frames)}, generated={len(generated_frames)}. "
            f"Using minimum length."
        )

    min_len = min(len(pose_frames), len(generated_frames))
    side_by_side_frames = []

    for i in range(min_len):
        pose_img = pose_frames[i]
        gen_img = generated_frames[i]

        # Ensure both images have the same height
        pose_np = np.array(pose_img)
        gen_np = np.array(gen_img)

        h1, w1 = pose_np.shape[:2]
        h2, w2 = gen_np.shape[:2]

        # Resize to same height if needed (use the max height)
        target_height = max(h1, h2)

        if h1 != target_height:
            new_width = int(w1 * target_height / h1)
            pose_img = pose_img.resize((new_width, target_height), Image.LANCZOS)
            pose_np = np.array(pose_img)
            h1, w1 = pose_np.shape[:2]

        if h2 != target_height:
            new_width = int(w2 * target_height / h2)
            gen_img = gen_img.resize((new_width, target_height), Image.LANCZOS)
            gen_np = np.array(gen_img)
            h2, w2 = gen_np.shape[:2]

        # Concatenate horizontally
        combined_np = np.concatenate([pose_np, gen_np], axis=1)
        combined_img = Image.fromarray(combined_np)
        side_by_side_frames.append(combined_img)

    logger.info(f"Created {len(side_by_side_frames)} side-by-side frames")
    return side_by_side_frames


def convert_video_to_gif(video_path, gif_path, fps=7, max_frames=None):
    """
    Convert a video file to an animated GIF.

    Args:
        video_path: Path to input video file
        gif_path: Path to save output GIF
        fps: Frames per second for the GIF
        max_frames: Maximum number of frames to include (None for all frames)
    """
    # Load video frames
    frames = []
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_count >= max_frames:
            break

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
        frame_count += 1

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    # Calculate duration per frame in milliseconds
    duration_ms = int(1000 / fps)

    # Save as GIF
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False
    )

    logger.info(f"Saved GIF with {len(frames)} frames to {gif_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with ControlNeXt-SVD model")

    parser.add_argument(
        "--pose_video",
        type=str,
        required=True,
        help="Path to the pose video file (conditioning input)"
    )
    parser.add_argument(
        "--reference_frame",
        type=str,
        default=None,
        help="Path to reference frame image. If not provided, will search in the default location based on pose video filename."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory (e.g., outputs/checkpoint-40000). If not provided, uses the latest checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_output",
        help="Directory to save output videos"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        help="Path to pretrained base model"
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./outputs",
        help="Directory containing checkpoint folders (used when --checkpoint is not specified)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=576,
        help="Width of generated video"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of generated video"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="Frames per second for output video"
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=float,
        default=127.0,
        help="Motion bucket ID for video generation"
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help="Noise augmentation strength"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=3.0,
        help="Minimum guidance scale"
    )
    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
        help="Maximum guidance scale"
    )
    parser.add_argument(
        "--controlnext_cond_scale",
        type=float,
        default=1.0,
        help="ControlNeXt conditioning scale"
    )
    parser.add_argument(
        "--frames_per_batch",
        type=int,
        default=21,
        help="Number of frames to generate per batch"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=4,
        help="Number of overlapping frames between batches"
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=4,
        help="Chunk size for VAE decoding"
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--reference_frames_dir",
        type=str,
        default="/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames",
        help="Directory containing reference frames (used when --reference_frame is not specified)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup accelerator
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Determine weight dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    logger.info(f"Using dtype: {weight_dtype}")

    # Find checkpoint
    if args.checkpoint is None:
        checkpoint_dir = find_latest_checkpoint(args.checkpoints_dir)
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"No checkpoints found in {args.checkpoints_dir}. "
                f"Please specify --checkpoint or train a model first."
            )
    else:
        checkpoint_dir = args.checkpoint
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    # Load models from checkpoint
    unet, controlnext, image_encoder, vae, feature_extractor = load_checkpoint_models(
        checkpoint_dir=checkpoint_dir,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        device=accelerator.device,
        weight_dtype=weight_dtype
    )

    # Load pose video
    logger.info(f"Loading pose video from {args.pose_video}")
    pose_frames = load_video_frames(args.pose_video)
    num_frames = len(pose_frames)

    # Load or find reference frame
    if args.reference_frame is not None:
        logger.info(f"Loading reference frame from {args.reference_frame}")
        reference_frame = Image.open(args.reference_frame).convert('RGB')
    else:
        logger.info("Searching for reference frame based on pose video filename...")
        reference_frame = find_reference_frame(args.pose_video, args.reference_frames_dir)

    # Create pipeline
    logger.info("Creating inference pipeline")
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

    # Run inference
    logger.info(f"Running inference on {num_frames} frames...")
    logger.info(f"Output size: {args.width}x{args.height}")
    logger.info(f"Frames per batch: {args.frames_per_batch}, Overlap: {args.overlap}")

    with torch.no_grad():
        video_frames = pipeline(
            reference_frame,
            pose_frames,
            height=args.height,
            width=args.width,
            num_frames=num_frames,
            frames_per_batch=args.frames_per_batch,
            decode_chunk_size=args.decode_chunk_size,
            motion_bucket_id=args.motion_bucket_id,
            fps=args.fps,
            controlnext_cond_scale=args.controlnext_cond_scale,
            min_guidance_scale=args.min_guidance_scale,
            max_guidance_scale=args.max_guidance_scale,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.num_inference_steps,
            overlap=args.overlap,
        ).frames

    # Flatten output frames (pipeline returns list of lists)
    output_frames = [frame for batch in video_frames for frame in batch]

    logger.info(f"Generated {len(output_frames)} frames")

    # Resize pose frames to match output dimensions
    logger.info(f"Resizing pose frames to {args.width}x{args.height}...")
    resized_pose_frames = [frame.resize((args.width, args.height), Image.LANCZOS) for frame in pose_frames]

    # Create side-by-side frames
    logger.info("Creating side-by-side video...")
    side_by_side_frames = create_side_by_side_frames(resized_pose_frames, output_frames)

    # Save outputs
    basename = os.path.splitext(os.path.basename(args.pose_video))[0]
    checkpoint_name = os.path.basename(checkpoint_dir).replace("/output_dir", "")

    # Save generated video only
    output_path = os.path.join(args.output_dir, f"{basename}_{checkpoint_name}.mp4")
    save_video(output_frames, output_path, fps=args.fps)

    # Save side-by-side video
    side_by_side_path = os.path.join(args.output_dir, f"{basename}_{checkpoint_name}_side_by_side.mp4")
    save_video(side_by_side_frames, side_by_side_path, fps=args.fps)

    # Convert side-by-side video to GIF
    logger.info("Converting side-by-side video to GIF...")
    gif_path = os.path.join(args.output_dir, f"{basename}_{checkpoint_name}_side_by_side.gif")
    convert_video_to_gif(side_by_side_path, gif_path, fps=args.fps)

    logger.info("=" * 80)
    logger.info("Inference completed successfully!")
    logger.info(f"Input pose video: {args.pose_video}")
    logger.info(f"Reference frame: {args.reference_frame if args.reference_frame else 'auto-detected'}")
    logger.info(f"Checkpoint: {checkpoint_dir}")
    logger.info(f"Generated video: {output_path}")
    logger.info(f"Side-by-side video: {side_by_side_path}")
    logger.info(f"Side-by-side GIF: {gif_path}")
    logger.info(f"Generated {len(output_frames)} frames at {args.width}x{args.height}, {args.fps} fps")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
