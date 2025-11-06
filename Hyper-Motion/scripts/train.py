import os
import sys
import argparse
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from hyper.models import (AutoencoderKLWan, CLIPModel,
                          WanT5EncoderModel, WanTransformer3DModel)
from ASL_dataset import ASLVideoDataset
from hyper.utils.utils import filter_kwargs


class SimpleTextEncoder(torch.nn.Module):
    """Simple learnable text embedding for memory-constrained training."""
    def __init__(self, vocab_size=256384, embedding_dim=4096, max_length=512):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None):
        """Returns embedded tokens."""
        embeddings = self.embedding(input_ids)
        return (embeddings,)  # Return tuple to match T5 encoder interface


def setup_logging(output_dir, rank):
    """Setup logging configuration"""
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.ERROR)
    return logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='HyperMotion Training Script for ASL Citizen Dataset')

    # Model config
    parser.add_argument('--config_path', type=str, default='config/wan2.1/wan_civitai.yaml',
                        help='Path to model config file')
    parser.add_argument('--model_name', type=str, default='./ckpts',
                        help='Path to pretrained model checkpoint')

    # ASL Citizen Dataset params
    parser.add_argument('--data_folder', type=str, required=True,
                        help='Path to ASL Citizen dataset folder (contains /asl and /pose subdirectories)')
    parser.add_argument('--validation_folder', type=str, default=None,
                        help='Path to validation dataset folder. If not specified, no validation is performed.')
    parser.add_argument('--sample_n_frames', type=int, default=14,
                        help='Number of frames to sample from each video')
    parser.add_argument('--interval_frame', type=int, default=3,
                        help='Interval between sampled frames (1=consecutive, 2=every other)')
    parser.add_argument('--random_start', action='store_true',
                        help='Randomly sample start frame for each video clip')
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Run validation every N epochs (default: 1, set to 0 to disable)')

    # Training params
    parser.add_argument('--output_dir', type=str, default='output/train',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['linear', 'cosine', 'constant'],
                        help='Learning rate scheduler type')

    # Video params
    parser.add_argument('--height', type=int, default=224,
                        help='Video height')
    parser.add_argument('--width', type=int, default=224,
                        help='Video width')
    parser.add_argument('--fps', type=int, default=7,
                        help='Target FPS for videos')

    # Model training params
    parser.add_argument('--mixed_precision', type=str, default='bf16',
                        choices=['no', 'fp16', 'bf16'],
                        help='Mixed precision training')
    parser.add_argument('--prompt', type=str,
                        default='A person signing a word in American Sign Language. High quality, masterpiece, best quality, high resolution.',
                        help='Text prompt for all training samples')
    parser.add_argument('--cfg_prob', type=float, default=0.1,
                        help='Probability of dropping text for classifier-free guidance training')

    # Checkpointing
    parser.add_argument('--checkpoint_every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--keep_checkpoints', type=int, default=5,
                        help='Number of checkpoints to keep')

    # Optimization
    parser.add_argument('--gradient_checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    parser.add_argument('--freeze_vae', action='store_true', default=True,
                        help='Freeze VAE during training')
    parser.add_argument('--freeze_text_encoder', action='store_true', default=True,
                        help='Freeze text encoder during training')
    parser.add_argument('--freeze_image_encoder', action='store_true', default=True,
                        help='Freeze image encoder during training')
    parser.add_argument('--use_simple_text_encoder', action='store_true',
                        help='Use a simple learnable text embedding instead of full T5 encoder (saves memory)')

    # Transformer freezing options
    parser.add_argument('--freeze_patch_embedding', action='store_true',
                        help='Freeze patch embedding layers in transformer')
    parser.add_argument('--freeze_text_embedding', action='store_true',
                        help='Freeze text embedding layers in transformer')
    parser.add_argument('--freeze_time_embedding', action='store_true',
                        help='Freeze time embedding layers in transformer')
    parser.add_argument('--freeze_transformer_layers', type=str, default=None,
                        help='Freeze specific transformer layers (e.g., "0-15" or "0,1,2,3")')
    parser.add_argument('--freeze_self_attention', action='store_true',
                        help='Freeze all self-attention layers')
    parser.add_argument('--freeze_cross_attention', action='store_true',
                        help='Freeze all cross-attention layers')
    parser.add_argument('--freeze_ffn', action='store_true',
                        help='Freeze all feed-forward networks')
    parser.add_argument('--train_motion_scale_only', action='store_true',
                        help='Only train motion_scale and space_scale_factor parameters')

    # Logging
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()
    return args


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def parse_layer_range(layer_str, num_layers):
    """Parse layer range string into list of layer indices.

    Args:
        layer_str: String like "0-15" or "0,1,2,3" or None
        num_layers: Total number of layers

    Returns:
        List of layer indices to freeze, or None if layer_str is None
    """
    if layer_str is None:
        return None

    layers = []
    parts = layer_str.split(',')
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))

    # Validate layer indices
    layers = [l for l in layers if 0 <= l < num_layers]
    return layers


def apply_transformer_freezing(transformer, args, logger):
    """Apply selective freezing to transformer model.

    Args:
        transformer: WanTransformer3DModel instance
        args: Training arguments with freezing options
        logger: Logger for reporting
    """
    num_frozen = 0
    num_trainable = 0

    # First, unfreeze everything (in case model was frozen before)
    for param in transformer.parameters():
        param.requires_grad = True

    # Freeze patch embedding
    if args.freeze_patch_embedding:
        transformer.patch_embedding.requires_grad_(False)
        num_frozen += sum(p.numel() for p in transformer.patch_embedding.parameters())
        logger.info("Frozen: patch_embedding")

    # Freeze text embedding
    if args.freeze_text_embedding:
        transformer.text_embedding.requires_grad_(False)
        num_frozen += sum(p.numel() for p in transformer.text_embedding.parameters())
        logger.info("Frozen: text_embedding")

    # Freeze time embedding
    if args.freeze_time_embedding:
        transformer.time_embedding.requires_grad_(False)
        transformer.time_projection.requires_grad_(False)
        num_frozen += sum(p.numel() for p in transformer.time_embedding.parameters())
        num_frozen += sum(p.numel() for p in transformer.time_projection.parameters())
        logger.info("Frozen: time_embedding and time_projection")

    # Train only motion_scale parameters
    if args.train_motion_scale_only:
        # Freeze everything first
        for param in transformer.parameters():
            param.requires_grad = False

        # Unfreeze only motion_scale and space_scale_factor
        for name, module in transformer.named_modules():
            if hasattr(module, 'motion_scale'):
                module.motion_scale.requires_grad = True
                logger.info(f"Trainable: {name}.motion_scale")
            if hasattr(module, 'space_scale_factor'):
                module.space_scale_factor.requires_grad = True
                logger.info(f"Trainable: {name}.space_scale_factor")

        num_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        num_frozen = sum(p.numel() for p in transformer.parameters() if not p.requires_grad)
        logger.info(f"Training only motion_scale parameters: {num_trainable:,} trainable, {num_frozen:,} frozen")
        return

    # Parse layer range if specified
    freeze_layers = parse_layer_range(args.freeze_transformer_layers, len(transformer.blocks))

    # Freeze specific transformer blocks or components
    for i, block in enumerate(transformer.blocks):
        # Freeze entire block if specified
        if freeze_layers is not None and i in freeze_layers:
            block.requires_grad_(False)
            num_frozen += sum(p.numel() for p in block.parameters())
            logger.info(f"Frozen: block {i} (all components)")
            continue

        # Freeze specific components within blocks
        if args.freeze_self_attention:
            block.self_attn.requires_grad_(False)
            num_frozen += sum(p.numel() for p in block.self_attn.parameters())
            if i == 0:  # Log once
                logger.info("Frozen: self_attn in all blocks")

        if args.freeze_cross_attention:
            block.cross_attn.requires_grad_(False)
            num_frozen += sum(p.numel() for p in block.cross_attn.parameters())
            if i == 0:  # Log once
                logger.info("Frozen: cross_attn in all blocks")

        if args.freeze_ffn:
            block.ffn.requires_grad_(False)
            num_frozen += sum(p.numel() for p in block.ffn.parameters())
            if i == 0:  # Log once
                logger.info("Frozen: ffn in all blocks")

    # Count trainable parameters
    num_trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transformer.parameters())

    logger.info(f"Transformer parameter summary:")
    logger.info(f"  Total: {total_params:,} parameters")
    logger.info(f"  Trainable: {num_trainable:,} parameters ({100 * num_trainable / total_params:.2f}%)")
    logger.info(f"  Frozen: {total_params - num_trainable:,} parameters ({100 * (total_params - num_trainable) / total_params:.2f}%)")


def compute_flow_matching_loss(model, vae, text_encoder, clip_image_encoder, tokenizer,
                                batch, noise_scheduler, weight_dtype, device, prompt, cfg_prob):
    """
    Compute flow matching loss for training.

    Flow matching trains the model to predict the velocity field between noise and data.

    Args:
        model: The transformer model
        vae: VAE for encoding/decoding
        text_encoder: Text encoder
        clip_image_encoder: CLIP image encoder
        tokenizer: Text tokenizer
        batch: Batch from ASLVideoDataset containing:
            - pixel_values: [B, T, C, H, W] target ASL video
            - guide_values: [B, T, C, H, W] pose control video
            - reference_image: [B, C, H, W] reference frame
        noise_scheduler: Flow matching scheduler
        weight_dtype: Weight dtype for computation
        device: Device to use
        prompt: Text prompt for all samples
        cfg_prob: Probability of dropping text for classifier-free guidance
    """
    # Extract batch data
    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)  # [B, T, C, H, W]
    guide_values = batch["guide_values"].to(device, dtype=weight_dtype)  # [B, T, C, H, W]
    reference_image = batch["reference_image"].to(device, dtype=weight_dtype)  # [B, C, H, W]

    batch_size = pixel_values.shape[0]

    # Prepare text prompts (with classifier-free guidance dropout)
    text_prompts = []
    for _ in range(batch_size):
        if np.random.random() < cfg_prob:
            text_prompts.append("")  # Empty prompt for CFG
        else:
            text_prompts.append(prompt)

    # Encode text (text encoder is on CPU)
    text_inputs = tokenizer(
        text_prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to('cpu')  # Keep on CPU for text encoder

    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )[0]  # Unpack tuple - text encoder returns (x, )
        # Move to GPU for transformer
        encoder_hidden_states = encoder_hidden_states.to(device, dtype=weight_dtype)

    # Encode videos to latent space using VAE (VAE is on CPU)
    with torch.no_grad():
        # Encode target video
        b, t, c, h, w = pixel_values.shape
        # Rearrange to [b, c, t, h, w] for VAE encoding
        pixel_values_reshaped = pixel_values.permute(0, 2, 1, 3, 4).cpu().float()  # Move to CPU for VAE
        latents = vae.encode(pixel_values_reshaped).latent_dist.sample()
        # latents shape: [b, latent_c, latent_t, latent_h, latent_w]
        # Permute back to [b, latent_t, latent_c, latent_h, latent_w]
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents * vae.spacial_compression_ratio
        latents = latents.to(device, dtype=weight_dtype)  # Move back to GPU

        # Encode control/guide video (pose)
        guide_reshaped = guide_values.permute(0, 2, 1, 3, 4).cpu().float()  # Move to CPU for VAE
        control_latents = vae.encode(guide_reshaped).latent_dist.sample()
        control_latents = control_latents.permute(0, 2, 1, 3, 4)
        control_latents = control_latents.to(device, dtype=weight_dtype)  # Move back to GPU

        # Encode reference image for CLIP (CLIP is on CPU)
        # CLIP encoder expects a list of [C, 1, H, W] tensors
        # reference_image is [B, C, H, W], so we convert each batch item to [C, 1, H, W]
        clip_images_list = [img.unsqueeze(1).cpu().float() for img in reference_image]  # Move to CPU for CLIP
        clip_image_features = clip_image_encoder(clip_images_list)
        clip_image_features = clip_image_features.to(device, dtype=weight_dtype)  # Move back to GPU

    # Sample timesteps
    timesteps = torch.randint(
        0, noise_scheduler.config.num_train_timesteps,
        (batch_size,), device=device
    ).long()

    # Sample noise
    noise = torch.randn_like(latents)

    # Add noise to latents according to flow matching schedule
    # Flow matching: x_t = (1 - t) * noise + t * x_0
    # where t is normalized to [0, 1]
    t_normalized = timesteps.float() / noise_scheduler.config.num_train_timesteps
    t_normalized = t_normalized.view(-1, 1, 1, 1, 1)

    noisy_latents = (1 - t_normalized) * noise + t_normalized * latents

    # The target for flow matching is the velocity: v = x_0 - noise
    target = latents - noise

    # Predict the velocity using the model
    # This is image-to-video (i2v) with pose control:
    # - x (control_latents): pose conditioning video
    # - y (noisy_latents): the video to denoise
    # - clip_fea: CLIP features from reference image (for identity/appearance)
    # - context: text embeddings (fixed generic prompt for all videos)

    b, t, c, h, w = noisy_latents.shape

    # Prepare control latents (pose) as input x - model expects [C, T, H, W] per sample
    # Model expects 52 total channels: 16 (y/noisy) + 36 (x/control)
    # We have 16 channels from control_latents, need to add 20 more
    control_input = control_latents.permute(0, 2, 1, 3, 4)  # [B, 16, T, H, W]

    # Add zero padding to reach 36 channels (52 - 16 from y)
    # Or duplicate control latents to provide more conditioning signal
    extra_control = torch.zeros(b, 20, t, h, w, device=control_input.device, dtype=control_input.dtype)
    control_input = torch.cat([control_input, extra_control], dim=1)  # [B, 36, T, H, W]

    control_input_list = [control_input[i] for i in range(b)]  # List of [36, T, H, W]

    # Prepare noisy latents as y (conditional input in i2v mode)
    noisy_input = noisy_latents.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    noisy_input_list = [noisy_input[i] for i in range(b)]  # List of [C, T, H, W]

    # Prepare text context
    context_list = [encoder_hidden_states[i] for i in range(b)]  # List of [L, D]

    # Add dtype attribute to list for model compatibility
    class TensorList(list):
        def __init__(self, items):
            super().__init__(items)
            self.dtype = items[0].dtype if len(items) > 0 else None

    control_input_list = TensorList(control_input_list)
    noisy_input_list = TensorList(noisy_input_list)

    # Compute sequence length after patching (depends on patch size)
    # With patch size (1, 2, 2), seq_len = T * (H/2) * (W/2)
    seq_len = t * (h // 2) * (w // 2)

    model_pred = model(
        x=control_input_list,  # Pose control video
        t=timesteps,
        context=context_list,  # Text embeddings (fixed)
        seq_len=seq_len,
        clip_fea=clip_image_features,  # CLIP features from reference image
        y=noisy_input_list,  # Noisy video to denoise (i2v mode)
        cond_flag=True
    )

    # Convert model output back to [B, T, C, H, W] format
    # model_pred is [B, C, T, H, W], target is [B, T, C, H, W]
    model_pred = model_pred.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]

    # Compute MSE loss
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

    return loss


def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, output_dir, rank, keep_checkpoints):
    """Save training checkpoint"""
    if rank != 0:
        return

    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model state
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, os.path.join(checkpoint_dir, 'training_state.pt'))

    logging.info(f"Saved checkpoint to {checkpoint_dir}")

    # Clean up old checkpoints
    checkpoints = sorted([d for d in os.listdir(output_dir) if d.startswith('checkpoint-')],
                        key=lambda x: int(x.split('-')[1]))
    if len(checkpoints) > keep_checkpoints:
        for old_checkpoint in checkpoints[:-keep_checkpoints]:
            old_path = os.path.join(output_dir, old_checkpoint)
            if os.path.isdir(old_path):
                import shutil
                shutil.rmtree(old_path)
                logging.info(f"Removed old checkpoint: {old_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """Load training checkpoint"""
    checkpoint = torch.load(os.path.join(checkpoint_path, 'training_state.pt'), map_location='cpu')

    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['step']


def validate(model, vae, text_encoder, clip_image_encoder, tokenizer, val_dataloader,
             noise_scheduler, weight_dtype, device, prompt, rank, logger, epoch, output_dir,
             num_samples=4):
    """
    Run validation on the validation dataset and generate sample videos.

    Args:
        num_samples: Number of sample videos to generate and save

    Returns:
        float: Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    if rank == 0:
        logger.info("Running validation...")

    # Store first batch for sample generation
    first_batch = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, desc="Validation", disable=rank != 0)):
            # Save first batch for sample generation
            if batch_idx == 0 and rank == 0:
                first_batch = batch

            # Compute loss using the same function as training
            loss = compute_flow_matching_loss(
                model,
                vae,
                text_encoder,
                clip_image_encoder,
                tokenizer,
                batch,
                noise_scheduler,
                weight_dtype,
                device,
                prompt,
                cfg_prob=0.0  # No CFG dropout during validation
            )

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Synchronize across distributed processes
    if torch.distributed.is_initialized():
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.AVG)
        avg_loss = avg_loss_tensor.item()

    # Generate sample videos (only on rank 0)
    if rank == 0 and first_batch is not None:
        logger.info(f"Generating {num_samples} validation samples...")
        try:
            generate_validation_samples(
                model, vae, text_encoder, clip_image_encoder, tokenizer,
                first_batch, noise_scheduler, weight_dtype, device, prompt,
                epoch, output_dir, num_samples, logger
            )
        except Exception as e:
            logger.error(f"Error generating validation samples: {e}")

    model.train()

    if rank == 0:
        logger.info(f"Validation loss: {avg_loss:.4f}")

    return avg_loss


def generate_validation_samples(model, vae, text_encoder, clip_image_encoder, tokenizer,
                                 batch, noise_scheduler, weight_dtype, device, prompt,
                                 epoch, output_dir, num_samples, logger):
    """Generate and save validation video samples."""
    import imageio
    import numpy as np

    # Create validation samples directory
    samples_dir = os.path.join(output_dir, 'validation_samples')
    os.makedirs(samples_dir, exist_ok=True)

    # Limit to num_samples
    batch_size = min(num_samples, batch["pixel_values"].shape[0])

    # Extract batch data (only first num_samples)
    pixel_values = batch["pixel_values"][:batch_size].to(device, dtype=weight_dtype)
    guide_values = batch["guide_values"][:batch_size].to(device, dtype=weight_dtype)
    reference_image = batch["reference_image"][:batch_size].to(device, dtype=weight_dtype)

    # Encode inputs
    with torch.no_grad():
        # Encode text
        text_prompts = [prompt] * batch_size
        text_inputs = tokenizer(
            text_prompts,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to('cpu')

        encoder_hidden_states = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )[0]
        encoder_hidden_states = encoder_hidden_states.to(device, dtype=weight_dtype)

        # Encode control video
        b, t, c, h, w = guide_values.shape
        guide_reshaped = guide_values.permute(0, 2, 1, 3, 4).cpu().float()
        control_latents = vae.encode(guide_reshaped).latent_dist.sample()
        control_latents = control_latents.permute(0, 2, 1, 3, 4).to(device, dtype=weight_dtype)

        # Encode reference image for CLIP
        clip_images_list = [img.unsqueeze(1).cpu().float() for img in reference_image]
        clip_image_features = clip_image_encoder(clip_images_list).to(device, dtype=weight_dtype)

        # Generate samples using the model (simplified inference)
        # Start from random noise
        latent_shape = (batch_size, t, 16, h // 8, w // 8)
        latents = torch.randn(latent_shape, device=device, dtype=weight_dtype)

        # Prepare inputs for model
        control_input = control_latents.permute(0, 2, 1, 3, 4)
        extra_control = torch.zeros(batch_size, 20, t, h // 8, w // 8,
                                    device=control_input.device, dtype=control_input.dtype)
        control_input = torch.cat([control_input, extra_control], dim=1)
        control_input_list = [control_input[i] for i in range(batch_size)]

        noisy_input = latents.permute(0, 2, 1, 3, 4)
        noisy_input_list = [noisy_input[i] for i in range(batch_size)]

        context_list = [encoder_hidden_states[i] for i in range(batch_size)]

        # Add dtype attribute for compatibility
        class TensorList(list):
            def __init__(self, items):
                super().__init__(items)
                self.dtype = items[0].dtype if len(items) > 0 else None

        control_input_list = TensorList(control_input_list)
        noisy_input_list = TensorList(noisy_input_list)

        # Single denoising step for visualization (full sampling would be too slow)
        seq_len = t * (h // 8 // 2) * (w // 8 // 2)
        timesteps = torch.tensor([500] * batch_size, device=device).long()

        model_pred = model(
            x=control_input_list,
            t=timesteps,
            context=context_list,
            seq_len=seq_len,
            clip_fea=clip_image_features,
            y=noisy_input_list,
            cond_flag=True
        )

        # Decode latents to videos
        model_pred = model_pred.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        model_pred = model_pred.cpu().float()
        decoded_videos = vae.decode(model_pred).sample

        # Convert to numpy and save
        decoded_videos = decoded_videos.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] -> [B, T, H, W, C]
        decoded_videos = ((decoded_videos + 1.0) / 2.0).clamp(0, 1)  # [-1, 1] -> [0, 1]
        decoded_videos = (decoded_videos.cpu().numpy() * 255).astype(np.uint8)

        # Also save ground truth and control
        gt_videos = ((pixel_values.cpu() + 1.0) / 2.0).clamp(0, 1)
        gt_videos = (gt_videos.permute(0, 1, 3, 4, 2).numpy() * 255).astype(np.uint8)

        control_videos = ((guide_values.cpu() + 1.0) / 2.0).clamp(0, 1)
        control_videos = (control_videos.permute(0, 1, 3, 4, 2).numpy() * 255).astype(np.uint8)

        # Save videos
        for i in range(batch_size):
            # Save generated video
            video_path = os.path.join(samples_dir, f'epoch{epoch:04d}_sample{i:02d}_generated.mp4')
            imageio.mimsave(video_path, decoded_videos[i], fps=7, codec='libx264')

            # Save ground truth
            gt_path = os.path.join(samples_dir, f'epoch{epoch:04d}_sample{i:02d}_groundtruth.mp4')
            imageio.mimsave(gt_path, gt_videos[i], fps=7, codec='libx264')

            # Save control (pose)
            control_path = os.path.join(samples_dir, f'epoch{epoch:04d}_sample{i:02d}_control.mp4')
            imageio.mimsave(control_path, control_videos[i], fps=7, codec='libx264')

        logger.info(f"Saved {batch_size} validation samples to {samples_dir}")


def main():
    args = get_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logger = setup_logging(args.output_dir, rank)

    if rank == 0:
        logger.info(f"Starting training with config: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")

    # Set seed
    set_seed(args.seed + rank)

    # Determine weight dtype
    if args.mixed_precision == 'fp16':
        weight_dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # Load config
    config = OmegaConf.load(args.config_path)

    # Initialize models
    if rank == 0:
        logger.info("Loading models...")

    # Transformer (main trainable model)
    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(args.model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Apply transformer freezing strategies
    if rank == 0:
        logger.info("Applying freezing strategies to transformer...")
    apply_transformer_freezing(transformer, args, logger)

    # VAE - Keep on CPU to save GPU memory (will be used with torch.no_grad())
    if rank == 0:
        logger.info("Loading VAE on CPU to save GPU memory...")
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to('cpu', dtype=torch.float32)  # Keep on CPU in float32

    if args.freeze_vae:
        vae.requires_grad_(False)
        vae.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Text encoder - Keep on CPU to save GPU memory
    if rank == 0:
        logger.info("Loading Text Encoder on CPU to save GPU memory...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    ).to('cpu')  # Keep on CPU

    if args.freeze_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # CLIP Image Encoder - Keep on CPU to save GPU memory
    if rank == 0:
        logger.info("Loading CLIP on CPU to save GPU memory...")
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to('cpu', dtype=torch.float32)  # Keep on CPU

    if args.freeze_image_encoder:
        clip_image_encoder.requires_grad_(False)
        clip_image_encoder.eval()

    # Noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Move transformer to device
    transformer = transformer.to(device)

    # Wrap with DDP if using distributed training
    if world_size > 1:
        transformer = DDP(transformer, device_ids=[local_rank], output_device=local_rank)

    # Setup optimizer (only for trainable parameters)
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if rank == 0:
        num_trainable = sum(p.numel() for p in trainable_params)
        num_total = sum(p.numel() for p in transformer.parameters())
        logger.info(f"Optimizing {len(trainable_params)} parameter groups")
        logger.info(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100 * num_trainable / num_total:.2f}%)")

        # Critical check: ensure we have trainable parameters
        if num_trainable == 0:
            raise RuntimeError(
                "No trainable parameters found in the model! "
                "Please check your freezing configuration. "
                "At least one parameter must be trainable for training to work."
            )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Setup ASL Citizen training dataset
    if rank == 0:
        logger.info(f"Loading ASL Citizen training dataset from {args.data_folder}")

    train_dataset = ASLVideoDataset(
        data_folder_path=args.data_folder,
        sample_n_frames=args.sample_n_frames,
        interval_frame=args.interval_frame,
        width=args.width,
        height=args.height,
        normalize_to_neg1_1=True,  # Normalize to [-1, 1] for HyperMotion
        random_start=args.random_start,
    )

    if rank == 0:
        logger.info(f"Training dataset loaded: {len(train_dataset)} video pairs")

    # Setup training dataloader
    if world_size > 1:
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Setup validation dataset (if provided)
    val_dataloader = None
    if args.validation_folder is not None and args.validate_every > 0:
        if rank == 0:
            logger.info(f"Loading ASL Citizen validation dataset from {args.validation_folder}")

        val_dataset = ASLVideoDataset(
            data_folder_path=args.validation_folder,
            sample_n_frames=args.sample_n_frames,
            interval_frame=args.interval_frame,
            width=args.width,
            height=args.height,
            normalize_to_neg1_1=True,
            random_start=False,  # No random start for validation (consistent evaluation)
        )

        if rank == 0:
            logger.info(f"Validation dataset loaded: {len(val_dataset)} video pairs")

        # Setup validation dataloader (no distributed sampler for validation)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
    elif rank == 0:
        logger.info("No validation dataset specified or validation disabled")

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Load from checkpoint if specified
    start_epoch = 0
    global_step = 0
    if args.resume_from_checkpoint is not None:
        if rank == 0:
            logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        start_epoch, global_step = load_checkpoint(
            args.resume_from_checkpoint, transformer, optimizer, lr_scheduler
        )

    # Setup gradient scaler for mixed precision
    scaler = GradScaler() if args.mixed_precision == 'fp16' else None

    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = None

    # Training loop
    if rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Batch size per device = {args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        logger.info(f"  Text prompt: {args.prompt}")
        if val_dataloader is not None:
            logger.info(f"  Validation every {args.validate_every} epoch(s)")

    for epoch in range(start_epoch, args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        transformer.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            disable=rank != 0,
            desc=f"Epoch {epoch}"
        )

        for step, batch in progress_bar:
            # Compute loss
            with autocast(dtype=weight_dtype) if args.mixed_precision != 'no' else torch.cuda.amp.autocast(enabled=False):
                loss = compute_flow_matching_loss(
                    transformer.module if isinstance(transformer, DDP) else transformer,
                    vae,
                    text_encoder,
                    clip_image_encoder,
                    tokenizer,
                    batch,
                    noise_scheduler,
                    weight_dtype,
                    device,
                    args.prompt,
                    args.cfg_prob
                )
                loss = loss / args.gradient_accumulation_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

                # Optimizer step
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # Logging
                if global_step % args.log_every == 0 and rank == 0:
                    logger.info(
                        f"Epoch: {epoch}, Step: {global_step}, Loss: {loss.item() * args.gradient_accumulation_steps:.4f}, "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.6f}"
                    )

                # Checkpointing
                if global_step % args.checkpoint_every == 0:
                    save_checkpoint(transformer, optimizer, lr_scheduler, epoch, global_step,
                                  args.output_dir, rank, args.keep_checkpoints)

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item() * args.gradient_accumulation_steps})

        # End of epoch
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        if rank == 0:
            logger.info(f"Epoch {epoch} completed. Average training loss: {avg_epoch_loss:.4f}")

        # Run validation if enabled
        if val_dataloader is not None and (epoch + 1) % args.validate_every == 0:
            val_loss = validate(
                transformer.module if isinstance(transformer, DDP) else transformer,
                vae,
                text_encoder,
                clip_image_encoder,
                tokenizer,
                val_dataloader,
                noise_scheduler,
                weight_dtype,
                device,
                args.prompt,
                rank,
                logger,
                epoch,
                args.output_dir,
                num_samples=4
            )

            # Save best model
            if rank == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")

                # Save best model
                best_model_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(best_model_dir, exist_ok=True)

                if isinstance(transformer, DDP):
                    model_state = transformer.module.state_dict()
                else:
                    model_state = transformer.state_dict()

                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(best_model_dir, 'best_model.pt'))

                logger.info(f"Saved best model to {best_model_dir}")
                best_model_path = best_model_dir

        # Save checkpoint at end of epoch
        save_checkpoint(transformer, optimizer, lr_scheduler, epoch + 1, global_step,
                       args.output_dir, rank, args.keep_checkpoints)

    if rank == 0:
        logger.info("Training completed!")
        if best_model_path is not None:
            logger.info(f"Best model saved at: {best_model_path} (val_loss: {best_val_loss:.4f})")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
