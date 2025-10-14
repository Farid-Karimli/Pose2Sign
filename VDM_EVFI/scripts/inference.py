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

def calculate_psnr(vid1, vid2):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""

    to_tensor= transforms.ToTensor()

    num_frames = len(vid1)

    psnr_list= []

    for i in range(num_frames):
        img1= vid1[i]
        img2= vid2[i]


        img1= to_tensor(img1)
        img2= to_tensor(img2)


        if mse_loss_fn(img1, img2) == 0:
            psnr_i= 100
        psnr_i= 10*torch.log10(65025/mse_loss_fn(img1, img2)).detach().cpu().numpy()

        psnr_list.append(psnr_i)
    

    return np.mean(psnr_list)


def calculate_mse(vid1, vid2):

    to_tensor = transforms.ToTensor()
    num_frames = len(vid1)
    assert num_frames == len(vid2), "Videos must have the same number of frames"
    
    mse_list = []

    for i in range(num_frames):
        img1 = to_tensor(vid1[i])
        img2 = to_tensor(vid2[i])

        mse = torch.mean((img1 - img2) ** 2).item()  # compute MSE
        mse_list.append(mse)
    
    mean_mse = sum(mse_list) / num_frames
    return mse_list, mean_mse


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

def main(args):
    # --- 1. Setup Device and DType ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16

    # --- 2. Load All Model Components Explicitly ---
    print("Loading all model components...")

    # Load the components that were NOT trained but are required by the pipeline
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        args.base_model_path, subfolder="image_encoder", revision=args.revision, variant="fp16", torch_dtype=weight_dtype
    )
    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        args.base_model_path, subfolder="vae", revision=args.revision, variant="fp16", torch_dtype=weight_dtype
    )

    # Load your custom and trained components
    unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
        args.base_model_path, subfolder="unet", variant="fp16", torch_dtype=weight_dtype
    )
    controlnet = ControlNetSDVModel.from_pretrained(
        os.path.join(args.checkpoint_path, "controlnet"), torch_dtype=weight_dtype
    )

    # Move all models to the correct device
    image_encoder.to(device)
    vae.to(device)
    unet.to(device)
    controlnet.to(device)
    
    # --- 3. Initialize Custom Pipeline with All Components ---
    print("Initializing custom pipeline with all explicit components...")
    # This now correctly mirrors your validation script's setup
    pipeline = StableVideoDiffusionPipelineControlNet.from_pretrained(
        args.base_model_path,
        unet=unet,
        image_encoder=image_encoder,
        controlnet=controlnet,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.to(device)
    pipeline.enable_model_cpu_offload()

    print(f"Initializing ASLVideoDataset from: {args.data_path}")
    dataset = ASLVideoDataset(data_folder_path=args.data_path)

    total_mse = 0
    total_psnr = 0

    for idx in range(len(dataset)):
        print(f"\n--- Processing item {idx} / {len(dataset)} ---")
        batch = dataset[idx]

        # --- Prepare inputs ---
        first_frame_tensor = batch["pixel_values"][0]
        conditioning_image = tensor_to_pil(first_frame_tensor)
        controlnet_cond = batch["guide_values"].to(device, dtype=weight_dtype)

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
        output_path = f"{args.output_path.rstrip('.mp4')}_{idx}_inference.mp4"
        output_frames_np = [np.array(img) for img in video_frames]
        export_to_video(output_frames_np, output_path, fps=7)

        # --- Ground-truth video from pixel_values ---
        gt_output_path = f"{args.output_path.rstrip('.mp4')}_{idx}_gt.mp4"
        gt_frames_np = [np.array(tensor_to_pil(frame)) for frame in batch["pixel_values"]]
        export_to_video(gt_frames_np, gt_output_path, fps=7)

        loaded_inference = load_video_as_frames(output_path)
        loaded_gt = load_video_as_frames(gt_output_path)

        # Convert to CHW float32 for metrics
        loaded_inference_chw = [frame.transpose(2,0,1).astype(np.float32) for frame in loaded_inference]
        loaded_gt_chw = [frame.transpose(2,0,1).astype(np.float32) for frame in loaded_gt]

        # Compute MSE and PSNR
        mse_per_frame, avg_mse = calculate_mse(loaded_inference_chw, loaded_gt_chw)
        psnr = calculate_psnr(loaded_inference_chw, loaded_gt_chw)

        total_mse+= avg_mse
        total_psnr += psnr

        print(f"Loaded videos metrics - Avg MSE: {avg_mse:.6f}, Avg PSNR: {psnr:.2f} dB")

        print(f"✅ Done for item {idx}")

    print('Average Psnr: ', total_psnr/len(dataset), ' Average MSE: ', total_mse/len(dataset))

'''
    # --- 4. Load Data using ASLVideoDataset ---
    print(f"Initializing ASLVideoDataset from: {args.data_path}")
    dataset = ASLVideoDataset(data_folder_path=args.data_path)

    if not 0 <= args.item_index < len(dataset):
        print(f"Error: --item_index must be between 0 and {len(dataset) - 1}.")
        return

    print(f"Loading item at index: {args.item_index}")
    batch = dataset[args.item_index]

    # --- 5. Prepare Inputs for the Pipeline ---
    first_frame_tensor = batch["pixel_values"][0]
    conditioning_image = tensor_to_pil(first_frame_tensor)
    controlnet_cond = batch["guide_values"].to(device, dtype=weight_dtype)

    # --- 6. Run Inference ---
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
        num_inference_steps = 50
    ).frames[0]

    # --- 7. Save Output ---
    print(f"Saving output to {args.output_path}...")
    output_frames_np = [np.array(img) for img in video_frames]

    if args.output_path.endswith(".mp4"):
        export_to_video(output_frames_np, args.output_path, fps=7)
    else:
        export_to_gif(video_frames, args.output_path)

    print("Inference complete!")
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SVD ControlNet using ASLVideoDataset.")
    
    # --- Core Paths ---
    parser.add_argument("--base_model_path", type=str, default="stabilityai/stable-video-diffusion-img2vid", help="Path to the base SVD model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the training checkpoint directory.")
    parser.add_argument("--output_path", type=str, default="output.gif", help="Path to save the generated video or GIF.")
    
    # --- Input Data Specification ---
    parser.add_argument("--data_path", type=str, required=True, help="Path to the root data folder for ASLVideoDataset.")
    parser.add_argument("--item_index", type=int, default=0, help="The index of the item to load from the dataset.")

    # --- Generation Parameters ---
    parser.add_argument("--width", type=int, default=64, help="Width of the generated video.")
    parser.add_argument("--height", type=int, default=64, help="Height of the generated video.")
    parser.add_argument("--num_frames", type=int, default=14, help="Number of frames to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--revision", type=str, default=None, help="Revision of the base model.")
    
    args = parser.parse_args()
    main(args)

'''
python inference.py \
  --checkpoint_path /restricted/projectnb/cs599dg/Pose2Sign/VDM_EVFI/scripts/checkpoints_old/checkpoint-550 \
  --data_path /restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/1000_10030_videos \
  --output_path ./generated_from_item_0.gif
'''
#  --item_index 0 \