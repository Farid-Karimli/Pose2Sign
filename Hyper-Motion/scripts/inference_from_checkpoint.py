"""
Inference script that loads trained checkpoints and generates ASL videos.

This script can load checkpoints from training (including best_model) and perform inference
on pose control videos to generate sign language videos.
"""

import os
import sys
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from hyper.dist import set_multi_gpus_devices
from hyper.models import (AutoencoderKLWan, CLIPModel,
                          WanT5EncoderModel, WanTransformer3DModel)
from hyper.models.cache_utils import get_teacache_coefficients
from hyper.pipeline import WanhyperPipeline
from hyper.utils.fp8_optimization import (convert_model_weight_to_float8,
                                          convert_weight_dtype_wrapper,
                                          replace_parameters_by_name)
from hyper.utils.utils import (filter_kwargs, get_video_to_video_latent,
                               save_videos_grid)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HyperMotion Inference from Training Checkpoint')

    # Model config
    parser.add_argument('--config_path', type=str, default='config/wan2.1/wan_civitai.yaml',
                        help='Path to model config file')
    parser.add_argument('--model_name', type=str, default='./ckpts',
                        help='Path to base pretrained model (contains vae, text_encoder, etc.)')
    parser.add_argument('--checkpoint', type=str, default='latest',
                        help='Path to checkpoint or "latest" or "best" to auto-find in output_dir')
    parser.add_argument('--output_dir', type=str, default='output/train',
                        help='Output directory where checkpoints are saved (for auto-finding)')

    # Input files
    parser.add_argument('--control_video', type=str, required=True,
                        help='Path to pose control video')
    parser.add_argument('--ref_image', type=str, required=True,
                        help='Path to reference image')

    # Generation params
    parser.add_argument('--prompt', type=str,
                        default='A person signing a word in American Sign Language. High quality, masterpiece, best quality, high resolution.',
                        help='Text prompt for generation')
    parser.add_argument('--negative_prompt', type=str,
                        default='Twisted body, limb deformities',
                        help='Negative prompt for generation')
    parser.add_argument('--guidance_scale', type=float, default=6.0,
                        help='Guidance scale for classifier-free guidance')
    parser.add_argument('--num_inference_steps', type=int, default=25,
                        help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=43,
                        help='Random seed for generation')

    # Video params
    parser.add_argument('--video_length', type=int, default=49,
                        help='Number of frames to generate')
    parser.add_argument('--height', type=int, default=576,
                        help='Video height')
    parser.add_argument('--width', type=int, default=1024,
                        help='Video width')
    parser.add_argument('--fps', type=int, default=16,
                        help='FPS for output video')

    # GPU settings
    parser.add_argument('--gpu_memory_mode', type=str, default='model_cpu_offload',
                        choices=['model_full_load', 'model_cpu_offload',
                                'model_cpu_offload_and_qfloat8', 'sequential_cpu_offload'],
                        help='GPU memory management mode')
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['fp16', 'bf16', 'fp32'],
                        help='Mixed precision type')

    # TeaCache
    parser.add_argument('--enable_teacache', action='store_true',
                        help='Enable TeaCache for faster inference')
    parser.add_argument('--teacache_threshold', type=float, default=0.10,
                        help='TeaCache threshold (0.05-0.20)')

    # Output
    parser.add_argument('--save_path', type=str, default='samples/inference_results',
                        help='Output directory for generated videos')

    args = parser.parse_args()
    return args


def find_checkpoint(output_dir, checkpoint_type='latest'):
    """
    Find checkpoint path based on type.

    Args:
        output_dir: Directory containing checkpoints
        checkpoint_type: 'latest', 'best', or a specific path

    Returns:
        Path to checkpoint directory containing training_state.pt or best_model.pt
    """
    if checkpoint_type == 'best':
        best_model_path = os.path.join(output_dir, 'best_model')
        if os.path.exists(os.path.join(best_model_path, 'best_model.pt')):
            print(f"Found best model at: {best_model_path}")
            return best_model_path, 'best_model.pt'
        else:
            raise FileNotFoundError(f"Best model not found at {best_model_path}")

    elif checkpoint_type == 'latest':
        # Find latest checkpoint by step number
        checkpoint_dirs = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
        if not checkpoint_dirs:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")

        # Sort by step number
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: int(x.split('-')[-1]))
        latest_checkpoint = checkpoint_dirs[-1]

        if os.path.exists(os.path.join(latest_checkpoint, 'training_state.pt')):
            print(f"Found latest checkpoint at: {latest_checkpoint}")
            return latest_checkpoint, 'training_state.pt'
        else:
            raise FileNotFoundError(f"training_state.pt not found in {latest_checkpoint}")

    else:
        # Assume checkpoint_type is a direct path
        checkpoint_path = checkpoint_type
        if os.path.isdir(checkpoint_path):
            # Check for training_state.pt or best_model.pt
            if os.path.exists(os.path.join(checkpoint_path, 'training_state.pt')):
                return checkpoint_path, 'training_state.pt'
            elif os.path.exists(os.path.join(checkpoint_path, 'best_model.pt')):
                return checkpoint_path, 'best_model.pt'
            else:
                raise FileNotFoundError(f"No checkpoint file found in {checkpoint_path}")
        elif os.path.isfile(checkpoint_path):
            # Direct path to checkpoint file
            return os.path.dirname(checkpoint_path), os.path.basename(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")


def load_checkpoint_weights(transformer, checkpoint_dir, checkpoint_file):
    """
    Load trained weights into transformer model.

    Args:
        transformer: Transformer model
        checkpoint_dir: Directory containing checkpoint
        checkpoint_file: Name of checkpoint file

    Returns:
        epoch, step from checkpoint (if available)
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        val_loss = checkpoint.get('val_loss', None)

        print(f"Checkpoint info: Epoch {epoch}, Step {step}")
        if val_loss is not None:
            print(f"Validation loss: {val_loss:.4f}")
    else:
        # Assume checkpoint is just the state dict
        state_dict = checkpoint
        epoch, step = 0, 0

    # Load state dict
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"Warning: Missing keys in checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys[:5]}...")

    print("Checkpoint loaded successfully!")
    return epoch, step


def main():
    args = get_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set weight dtype
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    print(f"Using dtype: {weight_dtype}")

    # Load config
    config = OmegaConf.load(args.config_path)

    # Find checkpoint
    checkpoint_dir, checkpoint_file = find_checkpoint(args.output_dir, args.checkpoint)

    # Load Transformer
    print("Loading transformer model...")
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    # Load checkpoint weights
    epoch, step = load_checkpoint_weights(transformer, checkpoint_dir, checkpoint_file)

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    # Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Load Text encoder
    print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Load CLIP Image Encoder
    print("Loading CLIP image encoder...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Load Scheduler
    print("Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Create Pipeline
    print("Creating pipeline...")
    pipeline = WanhyperPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )

    # Setup GPU memory management
    print(f"Setting up GPU memory mode: {args.gpu_memory_mode}")
    if args.gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"])
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    # Setup TeaCache if enabled
    if args.enable_teacache:
        coefficients = get_teacache_coefficients(args.model_name)
        if coefficients is not None:
            print(f"Enabling TeaCache with threshold {args.teacache_threshold}")
            pipeline.transformer.enable_teacache(
                coefficients, args.num_inference_steps, args.teacache_threshold,
                num_skip_start_steps=5, offload=False
            )

    # Setup generator
    generator = torch.Generator(device=device).manual_seed(args.seed)

    # Prepare inputs
    print("Preparing inputs...")
    print(f"Control video: {args.control_video}")
    print(f"Reference image: {args.ref_image}")

    if not os.path.exists(args.control_video):
        raise FileNotFoundError(f"Control video not found: {args.control_video}")
    if not os.path.exists(args.ref_image):
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")

    with torch.no_grad():
        # Adjust video length for VAE compression
        video_length = int((args.video_length - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1
        if video_length != args.video_length:
            print(f"Adjusted video length from {args.video_length} to {video_length} for VAE compression")

        sample_size = [args.height, args.width]

        # Load and process input video and reference image
        input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(
            args.control_video,
            video_length=video_length,
            sample_size=sample_size,
            fps=args.fps,
            ref_image=args.ref_image
        )

        print(f"Generating video with {args.num_inference_steps} steps...")
        print(f"Prompt: {args.prompt}")

        # Generate video
        sample = pipeline(
            args.prompt,
            num_frames=video_length,
            negative_prompt=args.negative_prompt,
            height=sample_size[0],
            width=sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            control_video=input_video,
            ref_image=ref_image,
            clip_image=clip_image,
        ).videos

    # Save results
    print("Saving results...")
    os.makedirs(args.save_path, exist_ok=True)

    # Generate filename
    control_name = Path(args.control_video).stem
    checkpoint_name = f"epoch{epoch}_step{step}" if epoch > 0 or step > 0 else "pretrained"
    output_filename = f"{control_name}_{checkpoint_name}.mp4"
    video_path = os.path.join(args.save_path, output_filename)

    # Save video (skip first frame as it's the reference)
    sample_without_first_frame = sample[:, :, 1:]
    save_videos_grid(sample_without_first_frame, video_path, fps=args.fps)

    print(f"Video saved to: {video_path}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
