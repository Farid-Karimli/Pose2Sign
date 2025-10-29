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
    parser.add_argument('--sample_n_frames', type=int, default=49,
                        help='Number of frames to sample from each video')
    parser.add_argument('--interval_frame', type=int, default=1,
                        help='Interval between sampled frames (1=consecutive, 2=every other)')
    parser.add_argument('--random_start', action='store_true',
                        help='Randomly sample start frame for each video clip')

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
    parser.add_argument('--height', type=int, default=576,
                        help='Video height')
    parser.add_argument('--width', type=int, default=1024,
                        help='Video width')
    parser.add_argument('--fps', type=int, default=16,
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

    # Encode text
    text_inputs = tokenizer(
        text_prompts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        encoder_hidden_states = text_encoder(
            input_ids=text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask
        )

    # Encode videos to latent space using VAE
    with torch.no_grad():
        # Encode target video
        b, t, c, h, w = pixel_values.shape
        # Rearrange to [b, c, t, h, w] for VAE encoding
        pixel_values_reshaped = pixel_values.permute(0, 2, 1, 3, 4)
        latents = vae.encode(pixel_values_reshaped).latent_dist.sample()
        # latents shape: [b, latent_c, latent_t, latent_h, latent_w]
        # Permute back to [b, latent_t, latent_c, latent_h, latent_w]
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents * vae.spacial_compression_ratio

        # Encode control/guide video (pose)
        guide_reshaped = guide_values.permute(0, 2, 1, 3, 4)
        control_latents = vae.encode(guide_reshaped).latent_dist.sample()
        control_latents = control_latents.permute(0, 2, 1, 3, 4)

        # Encode reference image for CLIP
        # CLIP encoder expects a list of [C, 1, H, W] tensors
        # reference_image is [B, C, H, W], so we convert each batch item to [C, 1, H, W]
        clip_images_list = [img.unsqueeze(1) for img in reference_image]  # List of [C, 1, H, W]
        clip_image_features = clip_image_encoder(clip_images_list)

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
    model_pred = model(
        noisy_latents,
        timesteps,
        encoder_hidden_states=encoder_hidden_states,
        control_latents=control_latents,
        clip_image_features=clip_image_features,
        return_dict=False
    )[0]

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

    # VAE
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(args.model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(device, dtype=weight_dtype)

    if args.freeze_vae:
        vae.requires_grad_(False)
        vae.eval()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).to(device)

    if args.freeze_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    # CLIP Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(args.model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(device, dtype=weight_dtype)

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
        logger.info(f"Optimizing {len(trainable_params)} parameter groups")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Setup ASL Citizen dataset
    if rank == 0:
        logger.info(f"Loading ASL Citizen dataset from {args.data_folder}")

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
        logger.info(f"Dataset loaded: {len(train_dataset)} video pairs")

    # Setup dataloader
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

    # Training loop
    if rank == 0:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Batch size per device = {args.batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        logger.info(f"  Text prompt: {args.prompt}")

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
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save checkpoint at end of epoch
        save_checkpoint(transformer, optimizer, lr_scheduler, epoch + 1, global_step,
                       args.output_dir, rank, args.keep_checkpoints)

    if rank == 0:
        logger.info("Training completed!")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
