import imageio
import numpy as np
import math
import csv
import os
import os.path
import json
import pandas as pd
from torchvision import transforms
from utils import videotransforms

import cv2
import torch
import torch.utils.data as data_utl

from IPython.display import Image

from utils.annotate import PoseStore, _build_hand_heatmaps_from_kps

# This code was taken and appropriated from the MS ASL repository: https://github.com/microsoft/ASL-citizen-code/blob/main/I3D/aslcitizen_dataset.py 

# loads rgb frames from video path, centering and downsizing as needed
def load_rgb_frames_from_video(video_path, max_frames=64, resize=(256, 256)):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    start = 0
    frameskip = 1
    
    # Adjust FPS dynamically based on length of video
    frameskip = 1
    if total_frames >= 96:
        frameskip = 2
    if total_frames >= 160:
        frameskip = 3

    # Set start frame so the video is "centered" across frames
    if frameskip == 3:
        start = np.clip(int((total_frames - 192) // 2), 0, 160)
    elif frameskip == 2:
        start = np.clip(int((total_frames - 128) // 2), 0, 96)
    else:
        start = np.clip(int((total_frames - 64) // 2), 0, 64)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    
    for offset in range(0, min(max_frames * frameskip, int(total_frames - start))):
        success, img = vidcap.read()
        if offset % frameskip == 0:
            w, h, c = img.shape
            if w < 226 or h < 226:
                d = 226. - min(w, h)
                sc = 1 + d / min(w, h)
                img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            if w > 256 or h > 256:
                img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))
            img = (img / 255.) * 2 - 1
            frames.append(img)
    return np.asarray(frames, dtype=np.float32)

def load_rgb_frames_from_video2(video_path, target_frames=16, resize=(224, 224)):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate the interval to sample frames evenly
    frame_interval = max(1, total_frames // target_frames)
    
    frames_to_capture = set(np.round(np.linspace(0, total_frames - 1, target_frames)).astype(int))
    
    captured_frames = 0
    for f in range(total_frames):
        success, img = vidcap.read()
        if not success:
            break
        if f in frames_to_capture:
            img = cv2.resize(img, resize)
            # Normalize to [0, 1] range as expected by VideoMAE
            img = img.astype(np.float32) / 255.0
            
            frames.append(img)
            captured_frames += 1
            if captured_frames >= target_frames:
                break  # Stop if we've captured enough frames

    vidcap.release()
    return np.asarray(frames, dtype=np.float32)


def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))


class ASLCitizen(data_utl.Dataset):
    def __init__(self,
                 datadir,
                 transforms,
                 video_file,
                 gloss_dict=None,
                 num_frames=64,
                 n_classes=None,
                 selected_classes=None,
                 posemap_json_path: str | None = None,
                 pose_shards_dir: str | None = None,
                 pose_sigma: float = 4.0):     
        """
        datadir: the directory of videos
        transforms: the transforms to be applied each video
        video_file: CSV file containing the video paths and labels
        gloss_dict: the dictionary of the glosses (words)
        num_frames: the number of frames to be sampled from the video
        n_classes: if provided, limit to this many classes (deterministic selection)
        selected_classes: if provided, use only these specific classes
        """
        self.transforms = transforms
        self.video_paths = []
        self.video_info = []
        self.labels = []
        self.num_frames = num_frames
        self.gloss_dict = gloss_dict if gloss_dict else dict()
        self.posemap_json_path = posemap_json_path
        self.posemaps = None
        self.pose_sigma = pose_sigma

        # Legacy single-JSON heatmaps (if you already have them)
        if self.posemap_json_path and os.path.exists(self.posemap_json_path):
            try:
                with open(self.posemap_json_path, 'r') as f:
                    self.posemaps = json.load(f)
                print(f"Loaded posemaps from {self.posemap_json_path} ({len(self.posemaps)} entries)")
            except Exception as e:
                print(f"Failed to load posemaps from {self.posemap_json_path}: {e}")
                self.posemaps = None

        # NEW: sharded keypoints store
        self.pose_store = PoseStore(pose_shards_dir) if pose_shards_dir else None

        print("--------------")
        print("Initializing dataset...")
        print(f"Datadir: {datadir}")
        print(f"Video file: {video_file}")
        print(f"Num frames: {num_frames}")
        print(f"Transforms: {self.transforms}")

        if not gloss_dict: #initialize gloss dict if not passed in as argument
            self.gloss_dict = {}
            g_count = 0
        
            gloss_list = []
            video_df = pd.read_csv(video_file)
            for i, row in video_df.iterrows():
                    g = row['class'].strip()
                    if g not in gloss_list:
                        gloss_list.append(g)
            gloss_list.sort()

            for i in range(len(gloss_list)):
                g = gloss_list[i]
                ind = i
                self.gloss_dict[g] = ind
        else:
            self.gloss_dict = gloss_dict
            g_count = len(gloss_dict)

        print(f'Number of glossary items: {len(self.gloss_dict)}')

        # Handle class subset selection
        if selected_classes is not None:
            # Use specific classes provided
            self.gloss_dict = {g: i for i, g in enumerate(sorted(selected_classes))}
            print(f"Using {len(self.gloss_dict)} specified classes.")
        elif n_classes and n_classes < len(self.gloss_dict):
            # Deterministically select first n_classes alphabetically
            gloss_list = sorted(list(self.gloss_dict.keys()))[:n_classes]
            self.gloss_dict = {g: i for i, g in enumerate(gloss_list)}
            print(f"Reduced glossary to {n_classes} classes (deterministic selection).")
            
        df = pd.read_csv(video_file)
        for i, row in df.iterrows():
            signer_id = row['signer_id']
            filepath = os.path.join(datadir, os.path.basename(row['filepath']))
            class_name = row['class'].strip()
            label = row['label']

            if class_name not in self.gloss_dict:
                continue

            self.video_paths.append(filepath)
            self.video_info.append(signer_id)
            self.video_info.append(class_name)
            # Map the original label to the new label space
            new_label = self.gloss_dict[class_name]
            self.labels.append(new_label)

        print(f"Length of gloss dict: {len(self.gloss_dict)}")
        print("--------------")
        print()

        # Load precomputed posemaps if available
        if self.posemap_json_path and os.path.exists(self.posemap_json_path):
            try:
                with open(self.posemap_json_path, 'r') as f:
                    self.posemaps = json.load(f)
                print(f"Loaded posemaps from {self.posemap_json_path} ({len(self.posemaps)} entries)")
            except Exception as e:
                print(f"Failed to load posemaps from {self.posemap_json_path}: {e}")
                self.posemaps = None

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        l = self.labels[index]

        try:
            imgs = load_rgb_frames_from_video2(video_path, self.num_frames)
            imgs = self.pad(imgs, self.num_frames)
            imgs = self.transforms(imgs)
            ret_img = video_to_tensor(imgs)
            ret_img = ret_img.permute(1, 0, 2, 3)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            ret_img = torch.zeros((self.num_frames, 3, 224, 224))

        basename = os.path.basename(video_path)

        # Keypoints store → build heatmaps now
        if getattr(self, "pose_store", None) and self.pose_store.has(basename):
            T, H, W = ret_img.shape[0], ret_img.shape[2], ret_img.shape[3]
            hmaps = []
            for t in range(T):
                kps = self.pose_store.get_frame_kps(basename, t)  # {"L":[...], "R":[...]} in pixel space
                hm = _build_hand_heatmaps_from_kps(kps, H=H, W=W, sigma=self.pose_sigma)  # (2,H,W)
                hmaps.append(hm)
            heatmaps = torch.from_numpy(np.stack(hmaps, axis=0))  # (T,2,H,W)

        else:
            heatmaps = None
            # On the fly computation
            # landmarks_by_frame = annotate_video(ret_img, draw_landmarks=False)
            # heatmaps = build_full_hand_heatmap(landmarks_by_frame)

        return {"video": ret_img, "label": l, "pose_map": heatmaps}

    def __len__(self):
        return len(self.video_paths)

    def pad(self, imgs, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]
            if num_padding:
                prob = np.random.random_sample()
                if prob > 0.5: #pad with first frame
                    pad_img = imgs[0]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
                else: #pad with last frame
                    pad_img = imgs[-1]
                    pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                    padded_imgs = np.concatenate([imgs, pad], axis=0)
        else:
            padded_imgs = imgs
        return padded_imgs

    @staticmethod
    def pad_wrap(imgs, label, total_frames):
        if imgs.shape[0] < total_frames:
            num_padding = total_frames - imgs.shape[0]

            if num_padding:
                pad = imgs[:min(num_padding, imgs.shape[0])]
                k = num_padding // imgs.shape[0]
                tail = num_padding % imgs.shape[0]

                pad2 = imgs[:tail]
                if k > 0:
                    pad1 = np.array(k * [pad])[0]

                    padded_imgs = np.concatenate([imgs, pad1, pad2], axis=0)
                else:
                    padded_imgs = np.concatenate([imgs, pad2], axis=0)
        else:
            padded_imgs = imgs

        label = label[:, 0]
        label = np.tile(label, (total_frames, 1)).transpose((1, 0))

        return padded_imgs, label


    def unnormalize_img(self, img):
        """Un-normalizes the image pixels."""
        # img = (img + 1) / 2
        img = (img * 255).astype("uint8")
        return img

    def create_gif(self, video_tensor, unnormalize, filename="sample.gif"):
        """Prepares a GIF from a video tensor.
        
        The video tensor is expected to have the following shape:
        (num_frames, num_channels, height, width).
        """
        frames = []
        for video_frame in video_tensor:
            frame = video_frame.permute(1, 2, 0).numpy().astype("uint8")
            if unnormalize:
                frame = self.unnormalize_img(video_frame.permute(1, 2, 0).numpy())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        kargs = {"duration": 0.25, "loop": 0}
        imageio.mimsave(filename, frames, "GIF", **kargs)
        return filename

    def display_gif(self, video_tensor, unnormalize = True, gif_name="sample.gif"):
        """Prepares and displays a GIF from a video tensor."""
        #video_tensor = video_tensor.permute(1, 0, 2, 3)
        gif_filename = self.create_gif(video_tensor, unnormalize, gif_name)
        return Image(filename=gif_filename)

    def get_video_by_label(self, label):
        """Returns a video tensor for the first video with the given label."""
        if label is str:
            label = self.gloss_dict[label]
        print(f"Looking for label {label}")
        videopath = np.random.choice([self.video_paths[i] for i in range(len(self.labels)) if self.labels[i] == label])

        video = load_rgb_frames_from_video2(videopath, self.num_frames)
        video = video_to_tensor(video).permute(1, 0, 2, 3)  # T,C,H,W
        return video


if __name__ == "__main__":
    datadir = "./ASL_Citizen/videos"
    video_file = "./ASL_Citizen/train.csv"
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224)])
    num_frames = 16
    ds = ASLCitizen(
        datadir="./ASL_Citizen/videos",
            transforms=train_transforms,                 # should output [T,C,H,W] 224×224
            video_file="./ASL_Citizen/train.csv",
            num_frames=16,
            pose_shards_dir="./pose_out",               # <<< NEW (folder with pose-*.jsonl.gz)
            pose_sigma=4.0,
            n_classes=100
        )

    gloss_dict = ds.gloss_dict

    print(f"Gloss dict: {gloss_dict}")
    ds.get_video_by_label("HELLO")

    