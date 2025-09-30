import os
import json
import cv2
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.draw_landmarks import draw_hand_landmarks_on_image, draw_body_landmarks_on_image
import torch
import gzip
import glob
import base64

# Mediapipe
# -------------------------------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# -------------------------------------
 # Create a pose landmarker instance with the video mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./utils/pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO)
pose_landmarker = PoseLandmarker.create_from_options(options)
# -------------------------------------
# Create a hand landmarker instance with the video mode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

def annotate_video(frames, draw_landmarks=False):
    new_frames = []
    landmarks = []

    # Create a hand landmarker instance with the video mode:
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='./utils/hand_landmarker.task'),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO)

    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    for i in range(len(frames)):
        frame = frames[i]

        original_frame = frame.permute(1, 2, 0).numpy()
        frame = frame.permute(1, 2, 0)
        
        # Convert tensor to numpy array and ensure correct format
        if hasattr(frame, 'numpy'):  # Check if it's a PyTorch tensor
            frame = frame.numpy()
        
        # Ensure the frame is in the correct format for MediaPipe
        # MediaPipe expects uint8 data in range [0, 255]
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:  # If normalized to [0, 1]
                frame = (frame * 255).astype(np.uint8)
            else:  # If in other range
                frame = frame.astype(np.uint8)
        
        # Convert BGR to RGB if needed (MediaPipe expects RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image with correct format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # draw the Hand landmarks
        landmark_result = hand_landmarker.detect_for_video(mp_image, i)
        
        # Store both landmarks and handedness for consistent ordering
        landmarks.append({
            'landmarks': landmark_result.hand_landmarks,
            'handedness': landmark_result.handedness
        })
      
        if draw_landmarks:
            annotated_frame = draw_hand_landmarks_on_image(original_frame, landmark_result)
            new_frames += [annotated_frame]

    # Process landmarks to create tensor-friendly format with consistent handedness
    processed_landmarks = []
    for i, frame_data in enumerate(landmarks):  # 16 frames
        # Initialize with zeros for 2 hands, 21 landmarks each, 3 coordinates
        # Index 0 = Left hand, Index 1 = Right hand
        frame_tensor = np.full((2, 21, 3), np.nan, dtype=np.float32)
        
        landmarks_list = frame_data['landmarks']
        handedness_list = frame_data['handedness']
        
        if len(landmarks_list) > 0:
            # Process detected hands with handedness-aware ordering
            for hand_idx, (hand_landmarks, handedness) in enumerate(zip(landmarks_list, handedness_list)):
                # Determine which tensor position this hand should go to
                if handedness[0].category_name == "Left":
                    tensor_hand_idx = 0  # Left hand goes to index 0
                elif handedness[0].category_name == "Right":
                    tensor_hand_idx = 1  # Right hand goes to index 1
                else:
                    # Fallback: if handedness is unclear, use detection order
                    tensor_hand_idx = hand_idx
                
                # Ensure we don't exceed our tensor bounds
                if tensor_hand_idx < 2:
                    # Fill in the landmarks for this hand
                    for landmark_idx, landmark in enumerate(hand_landmarks):
                        if landmark_idx < 21:  # Ensure we don't exceed 21 landmarks
                            frame_tensor[tensor_hand_idx, landmark_idx, 0] = landmark.x
                            frame_tensor[tensor_hand_idx, landmark_idx, 1] = landmark.y
                            frame_tensor[tensor_hand_idx, landmark_idx, 2] = landmark.z
        
        processed_landmarks.append(frame_tensor)

    # Convert to numpy array for easy tensor operations
    processed_landmarks = np.array(processed_landmarks)  # Shape: (T, 2, 21, 3)
    if draw_landmarks:
        return new_frames, processed_landmarks
    else:
        return processed_landmarks


def save_video(videopath, frames, width, height, fps):
    out = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
    for frame in frames:
        out.write(frame)
    out.release()

def gaussian_splat(H, W, x_px, y_px, sigma=4.0, weight=1.0):
    m = np.zeros((H,W), np.float32)
    x = int(round(x_px)); y = int(round(y_px))
    if 0 <= x < W and 0 <= y < H:
        m[max(0,y-1):y+2, max(0,x-1):x+2] = weight
        m = cv2.GaussianBlur(m, (0,0), sigmaX=sigma, sigmaY=sigma)
        m /= (m.max() + 1e-6)
    return m

def build_hand_heatmaps(L_t_21_3, R_t_21_3, H=224, W=224, sigma=4.0) -> np.array:
    """
    L_t_21_3, R_t_21_3: (21,3) for one frame in *pixel space* (x,y in [0,W],[0,H])
                        can be np.nan if missing.
    Returns: (2,H,W) heatmaps for (left, right)
    """
    def one_hand(h):
        if not np.isfinite(h).any():
            return np.zeros((H,W), np.float32)
        acc = np.zeros((H,W), np.float32)
        for (x,y,z) in h:
            if np.isfinite(x) and np.isfinite(y):
                scaled_x = x * W
                scaled_y = y * H
                acc += gaussian_splat(H, W, scaled_x, scaled_y, sigma=sigma, weight=1.0)
        if acc.max() > 0: acc /= (acc.max()+1e-6)
        return acc

    LH = one_hand(L_t_21_3)
    RH = one_hand(R_t_21_3)
    return np.stack([LH, RH], axis=0).astype(np.float32)  # (2,H,W)



def build_full_hand_heatmap(landmarks_by_frame):
    heatmaps = []
    for frame in landmarks_by_frame:
        left_hand = frame[0]
        right_hand = frame[1]
        heatmap = build_hand_heatmaps(left_hand, right_hand)
        heatmaps.append(heatmap)
        
    return torch.tensor(np.array(heatmaps))  # (T, 2, H, W)


def _b64_to_array(b64_str, dtype, shape):
    arr = np.frombuffer(base64.b64decode(b64_str), dtype=dtype)
    return arr.reshape(shape)

def _gaussian_splat(H, W, x, y, sigma, w=1.0):
    m = np.zeros((H, W), np.float32)
    ix, iy = int(round(x)), int(round(y))
    if 0 <= ix < W and 0 <= iy < H:
        m[max(0, iy-1):iy+2, max(0, ix-1):ix+2] = w
        m = cv2.GaussianBlur(m, (0, 0), sigma)
        m /= (m.max() + 1e-6)
    return m

def _build_hand_heatmaps_from_kps(kps_frame, H=224, W=224, sigma=4.0):
    """
    kps_frame: dict {"L":[(x,y,c)*21], "R":[...]} in *pixel space* for ONE frame.
    Returns (2,H,W) float32 in [0,1] : left, right channels (sum of 21 Gaussians).
    """
    def hand_map(points):
        if not points: return np.zeros((H,W), np.float32)
        acc = np.zeros((H,W), np.float32)
        for (x,y,c) in points:
            if c > 0.3:
                acc += _gaussian_splat(H,W,x,y,sigma, w=c)
        if acc.max() > 0: acc /= acc.max()
        return acc
    LH = hand_map(kps_frame.get("L", []))
    RH = hand_map(kps_frame.get("R", []))
    return np.stack([LH, RH], axis=0).astype(np.float32)  # (2,H,W)

class PoseStore:
    """
    Loads sharded JSONL.GZ keypoints (from extract_posemaps.py) into a memory dict:
      store[video_id] = {
        "T": T, "H": H, "W": W,
        "Lx": uint16 (T,21), "Ly": uint16 (T,21), "Lc": uint8 (T,21),
        "Rx": uint16 (T,21), "Ry": uint16 (T,21), "Rc": uint8 (T,21),
      }
    Decoding to float [0,1] happens on demand in __getitem__.
    """
    def __init__(self, shards_dir: str | None):
        self.data = {}
        if not shards_dir:
            return
        paths = sorted(glob.glob(os.path.join(shards_dir, "pose-*.jsonl.gz")))
        print(f"Found {len(paths)} pose shard(s) in {shards_dir}")
        if not paths:
            return
        loaded = 0
        for p in paths:
            try:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        rec = json.loads(line)
                        if rec.get("status") != "ok": continue
                        vid = rec["video_id"]
                        if "hands" not in rec:  # (if you stored heatmaps instead)
                            continue
                        T = int(rec["T"]); H = int(rec["H"]); W = int(rec["W"])
                        L = rec["hands"]["L"]; R = rec["hands"]["R"]
                        entry = {
                            "T": T, "H": H, "W": W,
                            "Lx": L["x"], "Ly": L["y"], "Lc": L["c"],
                            "Rx": R["x"], "Ry": R["y"], "Rc": R["c"],
                        }
                        self.data[vid] = entry
                        loaded += 1
            except Exception as e:
                print(f"[pose store] warn: failed shard {p}: {e}")
        print(f"[pose store] loaded keypoints for {loaded} videos from {len(paths)} shard(s)")

    def has(self, video_id: str) -> bool:
        return video_id in self.data

    def get_frame_kps(self, video_id: str, t: int):
        """
        Returns {"L":[(x,y,c)*21], "R":[...]} with x,y in pixel space (0..W-1, 0..H-1)
        """
        rec = self.data.get(video_id, None)
        if rec is None: return {"L": [], "R": []}
        T, H, W = rec["T"], rec["H"], rec["W"]
        t = min(max(t, 0), T-1)

        # decode lazily (keep base64 strings in memory)
        Lx = _b64_to_array(rec["Lx"], np.uint16, (T,21)).astype(np.float32) / 65535.0
        Ly = _b64_to_array(rec["Ly"], np.uint16, (T,21)).astype(np.float32) / 65535.0
        Lc = _b64_to_array(rec["Lc"], np.uint8,  (T,21)).astype(np.float32) / 255.0
        Rx = _b64_to_array(rec["Rx"], np.uint16, (T,21)).astype(np.float32) / 65535.0
        Ry = _b64_to_array(rec["Ry"], np.uint16, (T,21)).astype(np.float32) / 65535.0
        Rc = _b64_to_array(rec["Rc"], np.uint8,  (T,21)).astype(np.float32) / 255.0

        # to pixel space of the cropped input (assumes these were saved w.r.t. 224x224 crop)
        px = lambda a, M: (a * (M - 1.0))
        L = [(px(Lx[t,j],W), px(Ly[t,j],H), Lc[t,j]) for j in range(21)]
        R = [(px(Rx[t,j],W), px(Ry[t,j],H), Rc[t,j]) for j in range(21)]
        return {"L": L, "R": R}



if __name__ == '__main__':
    pose_store = PoseStore(shards_dir="./pose_out")
