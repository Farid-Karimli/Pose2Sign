"""
Pose Faithfulness Metrics for Sign Language Video Generation

This module provides metrics to evaluate how well generated videos follow input pose sequences.
Particularly useful for sign language generation where pose accuracy is critical.

Metrics:
- MPJPE: Mean Per Joint Position Error
- PCK: Percentage of Correct Keypoints
- Temporal Smoothness: Measures pose jitter/jumpiness
- Component-wise metrics: Body, Hands, Face separately
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json


class PoseExtractor:
    """Extract pose keypoints from video frames using MediaPipe."""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_from_frame(self, frame: np.ndarray) -> Dict:
        """
        Extract pose, hand, and face landmarks from a single frame.

        Args:
            frame: RGB image (H, W, 3)

        Returns:
            Dictionary with 'body', 'hands', 'face' keypoints
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pose_results = self.pose.process(image_rgb)
        hands_results = self.hands.process(image_rgb)
        face_results = self.face_mesh.process(image_rgb)

        keypoints = {
            'body': None,
            'hands': None,
            'face': None
        }

        # Extract body pose (33 landmarks)
        if pose_results.pose_landmarks:
            body_kps = []
            for lm in pose_results.pose_landmarks.landmark:
                body_kps.append([lm.x, lm.y, lm.z, lm.visibility])
            keypoints['body'] = np.array(body_kps)

        # Extract hand landmarks (21 landmarks per hand, max 2 hands)
        if hands_results.multi_hand_landmarks:
            hands_kps = []
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_kps = []
                for lm in hand_landmarks.landmark:
                    hand_kps.append([lm.x, lm.y, lm.z])
                hands_kps.append(np.array(hand_kps))
            keypoints['hands'] = hands_kps

        # Extract face mesh (468 landmarks)
        if face_results.multi_face_landmarks:
            face_kps = []
            for lm in face_results.multi_face_landmarks[0].landmark:
                face_kps.append([lm.x, lm.y, lm.z])
            keypoints['face'] = np.array(face_kps)

        return keypoints

    def extract_from_video(self, video_path: str, verbose: bool = True) -> List[Dict]:
        """
        Extract pose keypoints from all frames in a video.

        Args:
            video_path: Path to video file
            verbose: Show progress bar

        Returns:
            List of keypoint dictionaries, one per frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_keypoints = []

        iterator = tqdm(range(frame_count), desc="Extracting poses") if verbose else range(frame_count)

        for _ in iterator:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints = self.extract_from_frame(frame)
            all_keypoints.append(keypoints)

        cap.release()
        return all_keypoints

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()


def calculate_mpjpe(kp1: np.ndarray, kp2: np.ndarray, use_visibility: bool = True) -> float:
    """
    Calculate Mean Per Joint Position Error (MPJPE).

    Args:
        kp1, kp2: Keypoint arrays of shape (N, 3) or (N, 4) if visibility included
        use_visibility: If True and visibility data available, only count visible joints

    Returns:
        MPJPE value (lower is better)
    """
    if kp1 is None or kp2 is None:
        return np.nan

    # Handle different shapes
    if kp1.shape != kp2.shape:
        return np.nan

    # Extract x, y, z coordinates
    coords1 = kp1[:, :3]
    coords2 = kp2[:, :3]

    # Calculate Euclidean distance per joint
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))

    # Apply visibility mask if available
    if use_visibility and kp1.shape[1] == 4:
        visibility = kp1[:, 3]
        # Only count joints with visibility > 0.5
        mask = visibility > 0.5
        if mask.sum() > 0:
            distances = distances[mask]

    return np.mean(distances)


def calculate_pck(kp1: np.ndarray, kp2: np.ndarray, threshold: float = 0.05,
                  use_visibility: bool = True) -> float:
    """
    Calculate Percentage of Correct Keypoints (PCK).

    Args:
        kp1, kp2: Keypoint arrays of shape (N, 3) or (N, 4)
        threshold: Distance threshold for "correct" keypoint (normalized coordinates)
        use_visibility: If True and visibility data available, only count visible joints

    Returns:
        PCK value between 0 and 1 (higher is better)
    """
    if kp1 is None or kp2 is None:
        return np.nan

    if kp1.shape != kp2.shape:
        return np.nan

    # Extract x, y, z coordinates
    coords1 = kp1[:, :3]
    coords2 = kp2[:, :3]

    # Calculate Euclidean distance per joint
    distances = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))

    # Apply visibility mask if available
    if use_visibility and kp1.shape[1] == 4:
        visibility = kp1[:, 3]
        mask = visibility > 0.5
        if mask.sum() == 0:
            return np.nan
        distances = distances[mask]

    # Calculate percentage within threshold
    correct = (distances < threshold).sum()
    total = len(distances)

    return correct / total if total > 0 else np.nan


def calculate_temporal_smoothness(keypoints_sequence: List[np.ndarray]) -> float:
    """
    Calculate temporal smoothness of pose sequence (lower = smoother).
    Measures average frame-to-frame velocity changes (jerk).

    Args:
        keypoints_sequence: List of keypoint arrays (T, N, 3) or (T, N, 4)

    Returns:
        Average jerk (acceleration) across all joints
    """
    if len(keypoints_sequence) < 3:
        return np.nan

    # Filter out None values
    valid_kps = [kp for kp in keypoints_sequence if kp is not None]
    if len(valid_kps) < 3:
        return np.nan

    # Stack into (T, N, 3) array
    try:
        kps_array = np.stack([kp[:, :3] for kp in valid_kps])
    except:
        return np.nan

    # Calculate velocity (first derivative)
    velocity = np.diff(kps_array, axis=0)

    # Calculate acceleration (second derivative)
    acceleration = np.diff(velocity, axis=0)

    # Calculate jerk (third derivative) - change in acceleration
    jerk = np.diff(acceleration, axis=0)

    # Average jerk magnitude across all joints and frames
    jerk_magnitude = np.sqrt(np.sum(jerk ** 2, axis=2))
    avg_jerk = np.mean(jerk_magnitude)

    return avg_jerk


def align_hand_keypoints(hands1: Optional[List[np.ndarray]],
                         hands2: Optional[List[np.ndarray]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Align hand keypoints between two frames.
    Handles cases where different numbers of hands are detected.

    Returns:
        Tuple of (hands1_aligned, hands2_aligned) or (None, None)
    """
    if hands1 is None or hands2 is None:
        return None, None

    if len(hands1) == 0 or len(hands2) == 0:
        return None, None

    # If same number of hands, concatenate them
    if len(hands1) == len(hands2):
        return np.concatenate(hands1), np.concatenate(hands2)

    # Otherwise, use Hungarian algorithm or just take first hand
    # For simplicity, take the first hand only
    return hands1[0], hands2[0]


def evaluate_pose_faithfulness(
    pose_video_path: str,
    generated_video_path: str,
    pck_threshold: float = 0.05,
    verbose: bool = True,
    save_results: Optional[str] = None
) -> Dict:
    """
    Evaluate pose faithfulness between pose conditioning video and generated video.

    Args:
        pose_video_path: Path to input pose video (conditioning)
        generated_video_path: Path to generated video
        pck_threshold: Threshold for PCK metric (normalized coordinates)
        verbose: Print detailed results
        save_results: Path to save results JSON (optional)

    Returns:
        Dictionary containing all metrics
    """
    extractor = PoseExtractor()

    # Extract keypoints from both videos
    if verbose:
        print(f"Extracting poses from input video: {pose_video_path}")
    pose_kps = extractor.extract_from_video(pose_video_path, verbose=verbose)

    if verbose:
        print(f"Extracting poses from generated video: {generated_video_path}")
    gen_kps = extractor.extract_from_video(generated_video_path, verbose=verbose)

    extractor.close()

    # Align frame counts
    min_frames = min(len(pose_kps), len(gen_kps))
    if len(pose_kps) != len(gen_kps):
        print(f"Warning: Frame count mismatch. Using first {min_frames} frames.")
        pose_kps = pose_kps[:min_frames]
        gen_kps = gen_kps[:min_frames]

    # Calculate frame-wise metrics
    body_mpjpe_list = []
    body_pck_list = []
    hands_mpjpe_list = []
    hands_pck_list = []
    face_mpjpe_list = []
    face_pck_list = []

    for pose_kp, gen_kp in zip(pose_kps, gen_kps):
        # Body metrics
        if pose_kp['body'] is not None and gen_kp['body'] is not None:
            body_mpjpe_list.append(calculate_mpjpe(pose_kp['body'], gen_kp['body']))
            body_pck_list.append(calculate_pck(pose_kp['body'], gen_kp['body'], pck_threshold))

        # Hand metrics
        pose_hands, gen_hands = align_hand_keypoints(pose_kp['hands'], gen_kp['hands'])
        if pose_hands is not None and gen_hands is not None:
            hands_mpjpe_list.append(calculate_mpjpe(pose_hands, gen_hands, use_visibility=False))
            hands_pck_list.append(calculate_pck(pose_hands, gen_hands, pck_threshold, use_visibility=False))

        # Face metrics
        if pose_kp['face'] is not None and gen_kp['face'] is not None:
            face_mpjpe_list.append(calculate_mpjpe(pose_kp['face'], gen_kp['face'], use_visibility=False))
            face_pck_list.append(calculate_pck(pose_kp['face'], gen_kp['face'], pck_threshold, use_visibility=False))

    # Calculate temporal smoothness
    body_smoothness = calculate_temporal_smoothness([kp['body'] for kp in gen_kps])
    hands_smoothness = calculate_temporal_smoothness(
        [np.concatenate(kp['hands']) if kp['hands'] and len(kp['hands']) > 0 else None for kp in gen_kps]
    )

    # Compile results
    results = {
        'overall': {
            'num_frames': min_frames,
            'body_detected_frames': len(body_mpjpe_list),
            'hands_detected_frames': len(hands_mpjpe_list),
            'face_detected_frames': len(face_mpjpe_list),
        },
        'body': {
            'mpjpe': np.nanmean(body_mpjpe_list) if body_mpjpe_list else np.nan,
            'mpjpe_std': np.nanstd(body_mpjpe_list) if body_mpjpe_list else np.nan,
            'pck': np.nanmean(body_pck_list) if body_pck_list else np.nan,
            'temporal_smoothness': body_smoothness,
        },
        'hands': {
            'mpjpe': np.nanmean(hands_mpjpe_list) if hands_mpjpe_list else np.nan,
            'mpjpe_std': np.nanstd(hands_mpjpe_list) if hands_mpjpe_list else np.nan,
            'pck': np.nanmean(hands_pck_list) if hands_pck_list else np.nan,
            'temporal_smoothness': hands_smoothness,
        },
        'face': {
            'mpjpe': np.nanmean(face_mpjpe_list) if face_mpjpe_list else np.nan,
            'mpjpe_std': np.nanstd(face_mpjpe_list) if face_mpjpe_list else np.nan,
            'pck': np.nanmean(face_pck_list) if face_pck_list else np.nan,
        }
    }

    # Calculate combined score (weighted average)
    # For sign language: hands are most important, then body, then face
    weights = {'body': 0.3, 'hands': 0.5, 'face': 0.2}

    combined_mpjpe = 0
    combined_pck = 0
    total_weight = 0

    for component in ['body', 'hands', 'face']:
        if not np.isnan(results[component]['mpjpe']):
            combined_mpjpe += results[component]['mpjpe'] * weights[component]
            total_weight += weights[component]
        if not np.isnan(results[component]['pck']):
            combined_pck += results[component]['pck'] * weights[component]

    results['combined'] = {
        'weighted_mpjpe': combined_mpjpe / total_weight if total_weight > 0 else np.nan,
        'weighted_pck': combined_pck / total_weight if total_weight > 0 else np.nan,
    }

    # Print results
    if verbose:
        print("\n" + "="*60)
        print("POSE FAITHFULNESS EVALUATION RESULTS")
        print("="*60)
        print(f"Total frames: {results['overall']['num_frames']}")
        print(f"  Body detected: {results['overall']['body_detected_frames']}")
        print(f"  Hands detected: {results['overall']['hands_detected_frames']}")
        print(f"  Face detected: {results['overall']['face_detected_frames']}")
        print("\n" + "-"*60)
        print("BODY POSE:")
        print(f"  MPJPE: {results['body']['mpjpe']:.6f} ± {results['body']['mpjpe_std']:.6f}")
        print(f"  PCK@{pck_threshold}: {results['body']['pck']:.4f} ({results['body']['pck']*100:.2f}%)")
        print(f"  Temporal Smoothness: {results['body']['temporal_smoothness']:.6f}")
        print("\n" + "-"*60)
        print("HAND POSE:")
        print(f"  MPJPE: {results['hands']['mpjpe']:.6f} ± {results['hands']['mpjpe_std']:.6f}")
        print(f"  PCK@{pck_threshold}: {results['hands']['pck']:.4f} ({results['hands']['pck']*100:.2f}%)")
        print(f"  Temporal Smoothness: {results['hands']['temporal_smoothness']:.6f}")
        print("\n" + "-"*60)
        print("FACE:")
        print(f"  MPJPE: {results['face']['mpjpe']:.6f} ± {results['face']['mpjpe_std']:.6f}")
        print(f"  PCK@{pck_threshold}: {results['face']['pck']:.4f} ({results['face']['pck']*100:.2f}%)")
        print("\n" + "-"*60)
        print("COMBINED (Weighted - Hands: 50%, Body: 30%, Face: 20%):")
        print(f"  Weighted MPJPE: {results['combined']['weighted_mpjpe']:.6f}")
        print(f"  Weighted PCK: {results['combined']['weighted_pck']:.4f} ({results['combined']['weighted_pck']*100:.2f}%)")
        print("="*60)

    # Save results if requested
    if save_results:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(save_results, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\nResults saved to: {save_results}")

    return results


def evaluate_multiple_videos(
    video_pairs: List[Tuple[str, str]],
    pck_threshold: float = 0.05,
    verbose: bool = True,
    save_summary: Optional[str] = None
) -> List[Dict]:
    """
    Evaluate pose faithfulness for multiple video pairs.

    Args:
        video_pairs: List of (pose_video_path, generated_video_path) tuples
        pck_threshold: Threshold for PCK metric
        verbose: Print results
        save_summary: Path to save summary JSON

    Returns:
        List of result dictionaries
    """
    all_results = []

    for idx, (pose_path, gen_path) in enumerate(video_pairs):
        if verbose:
            print(f"\n{'#'*60}")
            print(f"Processing pair {idx+1}/{len(video_pairs)}")
            print(f"Pose video: {pose_path}")
            print(f"Generated video: {gen_path}")
            print(f"{'#'*60}")

        try:
            results = evaluate_pose_faithfulness(
                pose_path, gen_path,
                pck_threshold=pck_threshold,
                verbose=verbose
            )
            results['pose_video'] = pose_path
            results['generated_video'] = gen_path
            results['pair_index'] = idx
            all_results.append(results)
        except Exception as e:
            print(f"Error processing pair {idx+1}: {e}")
            all_results.append({
                'pose_video': pose_path,
                'generated_video': gen_path,
                'pair_index': idx,
                'error': str(e)
            })

    # Print summary
    if verbose:
        print("\n" + "#"*60)
        print("SUMMARY ACROSS ALL VIDEO PAIRS")
        print("#"*60)

        valid_results = [r for r in all_results if 'error' not in r]

        if valid_results:
            avg_body_mpjpe = np.nanmean([r['body']['mpjpe'] for r in valid_results])
            avg_body_pck = np.nanmean([r['body']['pck'] for r in valid_results])
            avg_hands_mpjpe = np.nanmean([r['hands']['mpjpe'] for r in valid_results])
            avg_hands_pck = np.nanmean([r['hands']['pck'] for r in valid_results])
            avg_combined_mpjpe = np.nanmean([r['combined']['weighted_mpjpe'] for r in valid_results])
            avg_combined_pck = np.nanmean([r['combined']['weighted_pck'] for r in valid_results])

            print(f"Successfully processed: {len(valid_results)}/{len(video_pairs)} pairs")
            print(f"\nAverage Body MPJPE: {avg_body_mpjpe:.6f}")
            print(f"Average Body PCK: {avg_body_pck:.4f} ({avg_body_pck*100:.2f}%)")
            print(f"Average Hands MPJPE: {avg_hands_mpjpe:.6f}")
            print(f"Average Hands PCK: {avg_hands_pck:.4f} ({avg_hands_pck*100:.2f}%)")
            print(f"Average Combined MPJPE: {avg_combined_mpjpe:.6f}")
            print(f"Average Combined PCK: {avg_combined_pck:.4f} ({avg_combined_pck*100:.2f}%)")

        if len(valid_results) < len(video_pairs):
            print(f"\nFailed to process: {len(video_pairs) - len(valid_results)} pairs")

    # Save summary
    if save_summary:
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj

        serializable_results = convert_to_serializable(all_results)

        with open(save_summary, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        if verbose:
            print(f"\nSummary saved to: {save_summary}")

    return all_results


# Example usage
if __name__ == "__main__":
    # Single video pair example
    pose_video = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/pose/example.mp4"
    generated_video = "./inference_output/example_checkpoint-40000.mp4"

    # Evaluate single pair
    results = evaluate_pose_faithfulness(
        pose_video,
        generated_video,
        pck_threshold=0.05,
        verbose=True,
        save_results="pose_evaluation_results.json"
    )

    # Or evaluate multiple pairs
    # video_pairs = [
    #     (pose_video_1, generated_video_1),
    #     (pose_video_2, generated_video_2),
    # ]
    # all_results = evaluate_multiple_videos(
    #     video_pairs,
    #     save_summary="pose_evaluation_summary.json"
    # )
