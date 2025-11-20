import argparse
import os
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn as nn

# --- Crucial Imports from your Project ---
from vid_dataset import ASLVideoDataset 
from diffusers import AutoencoderKLTemporalDecoder
from src.models.unet_spatio_temporal_condition_fullControlnet import UNetSpatioTemporalConditionControlNetModel
from src.pipelines.pipeline_stable_video_diffusion_fullControlnet_MStack_train import StableVideoDiffusionPipelineControlNet
from src.models.fullControlnet_sdv_MStack import ControlNetSDVModel

from metrics import calculate_mse, calculate_psnr, calculate_ssim

# --- Helper functions for saving output ---
mse_loss_fn = nn.MSELoss()

import cv2
import numpy as np

def load_video_as_frames(video_path):
    """Load video and return a list of frames as NumPy arrays (H,W,C) in RGB order)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV loads as BGR, convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames

def export_to_gif(frames, output_gif_path):
    """Saves a list of PIL Images or numpy arrays as a GIF."""
    pil_frames = [Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame for frame in frames]
    pil_frames[0].save(
        output_gif_path, format='GIF', append_images=pil_frames[1:],
        save_all=True, duration=1000 // 7, loop=0
    )

def export_to_video(video_frames, output_video_path, fps):
    """Saves a list of numpy arrays as an MP4 video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    video_writer.release()

def tensor_to_pil(tensor):
    """Converts a [-1, 1] PyTorch tensor to a PIL Image."""
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    numpy_array = tensor.permute(1, 2, 0).cpu().numpy()
    image_array = (numpy_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16

    # Load the components that were NOT trained but are required by the pipeline
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.base_model_path, 
        subfolder="image_encoder", 
        revision=args.revision, 
        variant="fp16", 
        torch_dtype=weight_dtype,
        ignore_mismatched_sizes=True
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.base_model_path, subfolder="vae", revision=args.revision, variant="fp16", torch_dtype=weight_dtype
    )

    # Load your custom and trained components
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.base_model_path, subfolder="unet", variant="fp16", torch_dtype=weight_dtype
    )
    controlnet = ControlNetSDVModel.from_pretrained(
        os.path.join(args.checkpoint_path, "controlnet"), torch_dtype=weight_dtype, 
    )

    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args.base_model_path,
        unet=unet,
        image_encoder=image_encoder,
        controlnet=controlnet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
        low_cpu_mem_usage=True, 
        ignore_mismatched_sizes=True
    )

    pipeline.to(device)

    return pipeline



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16

    pipeline = load_model(args)

    print(f"Initializing ASLVideoDataset from: {args.data_path}")
    dataset = ASLVideoDataset(data_folder_path=args.data_path, width=args.width, height=args.height)
    print(f"Dataset initialized with {len(dataset)} items.")

    total_mse = 0
    total_psnr = 0

    for idx in range(args.data_item_start, min(args.data_item_end, len(dataset))):
        print(f"\n--- Processing item {idx} / {len(dataset)} ---")
       
        batch = dataset[idx]

        # --- Prepare inputs ---
        first_frame_tensor = batch["pixel_values"][0]
        conditioning_image = tensor_to_pil(first_frame_tensor)
        print(f"Conditioning image size: {conditioning_image.size}")

        controlnet_cond = batch["guide_values"].to(device, dtype=weight_dtype)
        print(f"ControlNet condition shape: {controlnet_cond.shape}")

        # --- Run inference ---
        print("Generating video...")
        generator = torch.manual_seed(args.seed) if args.seed is not None else None
        video_frames = pipeline(
            image=conditioning_image,
            controlnet_condition=controlnet_cond, 
            height=args.height,
            width=args.width,
            fps=7,
            noise_aug_strength=0.02,
            generator=generator,
            num_inference_steps=25
        ).frames[0]

        # --- Convert generated frames to NumPy CHW for metrics ---
        gen_frames_chw = [np.array(img).transpose(2,0,1) for img in video_frames]

        # --- Convert ground-truth frames to NumPy CHW ---
        gt_frames_chw = [frame.cpu().numpy() for frame in batch["pixel_values"]]

        # --- Compute metrics between inference and ground-truth ---
        #mse_per_frame, avg_mse = calculate_mse(gen_frames_chw, gt_frames_chw)
        #psnr_per_frame, avg_psnr = calculate_psnr(gen_frames_chw, gt_frames_chw)

        #print(f"Item {idx} - Avg MSE: {avg_mse:.6f}, Avg PSNR: {avg_psnr:.2f} dB")

        # --- Save inference video ---
        output_path = os.path.join(args.output_path, f"{idx}_generated.mp4")
        output_frames_np = [np.array(img) for img in video_frames]
        export_to_video(output_frames_np, output_path, fps=7)

        # --- Ground-truth video from pixel_values ---
        gt_output_path = os.path.join(args.output_path, f"{idx}_ground_truth.mp4")
        gt_frames_np = [np.array(tensor_to_pil(frame)) for frame in batch["pixel_values"]]
        export_to_video(gt_frames_np, gt_output_path, fps=7)

        loaded_inference = load_video_as_frames(output_path)
        loaded_gt = load_video_as_frames(gt_output_path)

        # Convert to CHW float32 for metrics
        loaded_inference_chw = [frame.transpose(2,0,1).astype(np.float32) for frame in loaded_inference]
        loaded_gt_chw = [frame.transpose(2,0,1).astype(np.float32) for frame in loaded_gt]

        # # Compute MSE and PSNR
        # mse_per_frame, avg_mse = calculate_mse(resized_inference, resized_gt)
        # psnr = calculate_psnr(resized_inference, resized_gt)
        # ssim_value = calculate_ssim(resized_inference, resized_gt)

        # total_mse+= avg_mse
        # total_psnr += psnr

        # print(f"Loaded videos metrics - Avg MSE: {avg_mse:.6f}, Avg PSNR: {psnr:.2f} dB")

        print(f"✅ Done for item {idx}")

    # print('Average Psnr: ', total_psnr/len(dataset), ' Average MSE: ', total_mse/len(dataset))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SVD ControlNet using ASLVideoDataset.")
    
    # --- Core Paths ---
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-video-diffusion-img2vid", help="Path to the base SVD model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the training checkpoint directory.")
    parser.add_argument("--output_path", type=str, default="output.gif", help="Path to save the generated video or GIF.")
    
    # --- Input Data Specification ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to the root data folder for ASLVideoDataset.")
    parser.add_argument("--data_item_start", type=int, default=0, help="Starting index of the data item to process.")
    parser.add_argument("--data_item_end", type=int, default=1, help="Ending index (exclusive) of the data item to process.")

    # --- Generation Parameters ---
    parser.add_argument("--width", type=int, default=64, help="Width of the generated video.")
    parser.add_argument("--height", type=int, default=64, help="Height of the generated video.")
    parser.add_argument("--num_frames", type=int, default=14, help="Number of frames to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--revision", type=str, default=None, help="Revision of the base model.")
    
    args = parser.parse_args()
    main(args)