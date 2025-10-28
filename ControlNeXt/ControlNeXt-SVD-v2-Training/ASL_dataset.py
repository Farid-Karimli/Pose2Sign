import os, io, csv, math, random
import numpy as np
from einops import rearrange

import torch
from decord import VideoReader
import json

import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import cv2

from copy import deepcopy
from PIL import Image
import torch.nn.functional as F


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255.


def random_select_continual_sequence(sequence, nums, inter=2):
    length = len(sequence)
    inter = min(inter, length // nums)
    inter_length = inter * (nums - 1) + 1
    if inter_length > length:
        return None
    bg_idx = random.randint(0, length - inter_length)
    idx = []
    for i in range(nums):
        idx.append(sequence[bg_idx + i * inter])
    return idx


def draw_mask(frame, x0, y0, x1, y1, score=1., margin=10):
    H, W = frame.shape[-2:]
    x0 = int(x0 * W)
    x1 = int(x1 * W)
    y0 = int(y0 * H)
    y1 = int(y1 * H)
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, W), min(y1 + margin, H)
    frame[..., y0:y1, x0:x1] = score
    return frame

class ASLVideoDataset(Dataset):
    def __init__(
        self,
        data_folder_path,
        sample_n_frames=14,
        interval_frame=1,
        width=64,
        height=64,
        normalize_to_neg1_1=False,
        random_start=False,
    ):
        """
        Paired ASL RGB <-> pose video dataset returning contiguous clips.
        Args:
            data_folder_path: root folder containing 'asl' and 'pose' subdirectories
            sample_n_frames: number of frames to sample (T)
            interval_frame: interval between sampled frames (1=consecutive, 2=every other, etc.)
            width,height: resize target
            normalize_to_neg1_1: if True, pixel_values and reference_image are normalized to [-1,1]
            random_start: if True sample a random start; else start=0
        """
        self.data_folder = data_folder_path
        self.sample_n_frames = int(sample_n_frames)
        self.interval_frame = int(interval_frame)
        self.width = int(width)
        self.height = int(height)
        self.normalize = bool(normalize_to_neg1_1)
        self.random_start = bool(random_start)

        self.asl_video_dir = data_folder_path + '/asl'
        self.pose_video_dir = data_folder_path + '/pose'

        asl_video_dir = data_folder_path + '/asl'
        pose_video_dir = data_folder_path + '/pose'
        

        # build list of filenames that exist in both dirs
        asl_files = set([f for f in os.listdir(asl_video_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
        pose_files = set([f for f in os.listdir(pose_video_dir) if f.lower().endswith((".mp4", ".avi", ".mov"))])
        
        self.video_list = sorted(list(asl_files & pose_files))
        if len(self.video_list) == 0:
            raise RuntimeError("No matching video pairs found in the provided directories.")

    def __len__(self):
        return len(self.video_list)

    def _sample_contiguous_indices(self, total_frames: int):
        """Return a list of frame indices sampled with interval_frame stride.

           Args:
               total_frames: total number of frames available in the video

           Returns:
               List of frame indices. If video is too short, pads by repeating last frame.

           Example:
               sample_n_frames=4, interval_frame=2, total_frames=20
               - Required span: (4-1)*2 + 1 = 7 frames
               - If random_start=False, start=0: indices = [0, 2, 4, 6]
               - If random_start=True, start could be 5: indices = [5, 7, 9, 11]
        """
        T = self.sample_n_frames
        interval = self.interval_frame

        if total_frames <= 0:
            raise ValueError("Video has no frames.")

        # Calculate the span of frames we need
        required_span = (T - 1) * interval + 1

        if total_frames >= required_span:
            # We have enough frames to sample with the desired interval
            if self.random_start:
                max_start = total_frames - required_span
                start = random.randint(0, max_start)
            else:
                start = 0
            return list(range(start, start + required_span, interval))
        else:
            # Video is too short - sample what we can and pad
            # First, try to get as many frames as possible with the interval
            max_samples_with_interval = (total_frames - 1) // interval + 1

            if max_samples_with_interval >= T:
                # We can get enough samples with the interval
                indices = list(range(0, T * interval, interval))
            else:
                # Get what we can with interval, then pad
                indices = list(range(0, max_samples_with_interval * interval, interval))
                # Pad by repeating the last frame
                pad_count = T - len(indices)
                indices = indices + [total_frames - 1] * pad_count

            return indices

    def _load_single(self, index: int):
        """Load a single sample (pixel_values, guide_values, reference_image).
           pixel_values, guide_values: tensors [T, C, H, W]
           reference_image: tensor [C, H, W]
        """
        name = self.video_list[index]
        rgb_path = os.path.join(self.asl_video_dir, name)
        pose_path = os.path.join(self.pose_video_dir, name)

        # load video readers
        rgb_reader = VideoReader(rgb_path)
        pose_reader = VideoReader(pose_path)

        total_frames = min(len(rgb_reader), len(pose_reader))
        frame_indices = self._sample_contiguous_indices(total_frames)

        # read frames into numpy arrays
        rgb_frames = np.array([rgb_reader[i].asnumpy() for i in frame_indices])  # (T,H,W,C)
        pose_frames = np.array([pose_reader[i].asnumpy() for i in frame_indices])  # may be BGR

        # convert pose frames BGR->RGB (safe even if already RGB in many cases)
        for i in range(len(pose_frames)):
            try:
                pose_frames[i] = cv2.cvtColor(pose_frames[i], cv2.COLOR_BGR2RGB)
            except Exception:
                # if conversion fails, keep as-is
                pass

        # to tensors [T,C,H,W], values in [0,1]
        pixel_values = numpy_to_pt(rgb_frames)    # (T,C,H,W)
        guide_values = numpy_to_pt(pose_frames)   # (T,C,H,W)
        reference_image = pixel_values[0]         # (C,H,W)

        # resize to (H,W) if requested (using bilinear interpolation)
        if (pixel_values.shape[-2] != self.height) or (pixel_values.shape[-1] != self.width):
            # pixel_values: (T,C,H0,W0) -> interpolate expects (N,C,H,W), so pass as is
            pixel_values = F.interpolate(pixel_values, size=(self.height, self.width), mode='bilinear', align_corners=False)
            guide_values = F.interpolate(guide_values, size=(self.height, self.width), mode='bilinear', align_corners=False)
            reference_image = F.interpolate(reference_image.unsqueeze(0), size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(0)

        # normalize pixel_values and reference_image to [-1,1] if requested
        if self.normalize:
            pixel_values = (pixel_values - 0.5) / 0.5
            reference_image = (reference_image - 0.5) / 0.5
            # guide_values intentionally left in [0,1] (no normalization), consistent with prior behavior

        return pixel_values, guide_values, reference_image

    def get_batch(self, idx_or_indices):
        """Load one sample or a list of samples.
           - If idx_or_indices is int: returns (pixel_values[T,C,H,W], guide_values[T,C,H,W], reference_image[C,H,W])
           - If idx_or_indices is list/tuple/np.ndarray: returns stacked tensors:
               pixel_values: [B, T, C, H, W]
               guide_values: [B, T, C, H, W]
               reference_image: [B, C, H, W]
        """
        if isinstance(idx_or_indices, slice):
            rng = range(idx_or_indices.start or 0, idx_or_indices.stop or 0, idx_or_indices.step or 1)
            samples = [self._load_single(int(i)) for i in rng]
            px = torch.stack([s[0] for s in samples], dim=0)   # [B,T,C,H,W]
            gv = torch.stack([s[1] for s in samples], dim=0)   # [B,T,C,H,W]
            ri = torch.stack([s[2] for s in samples], dim=0)   # [B,C,H,W]
            return px, gv, ri
        else:
            return self._load_single(int(idx_or_indices))

    def __getitem__(self, idx):
        px, gv, ri = self.get_batch(idx)
        # Create a uniform hands_mask (no hand detection for ASL dataset)
        # Shape: [T, 1, H, W], all ones to give equal weight to all regions
        hands_mask = torch.ones((px.shape[0], 1, px.shape[2], px.shape[3]))

        return {
            "pixel_values": px,           # [T, C, H, W]
            "guide_values": gv,           # [T, C, H, W]
            "reference_image": ri,        # [C, H, W]
            "hands_mask": hands_mask      # [T, 1, H, W]
        }
