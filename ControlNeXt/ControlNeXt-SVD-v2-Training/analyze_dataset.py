#!/usr/bin/env python3
"""Analyze frame counts in ASL video dataset."""

import cv2
import os
import glob
import numpy as np

def analyze_videos(video_dir, sample_size=100):
    """Analyze frame counts in a directory of videos."""
    videos = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

    if len(videos) == 0:
        print(f"No videos found in {video_dir}")
        return

    # Sample if too many
    if len(videos) > sample_size:
        step = len(videos) // sample_size
        videos = videos[::step]

    frame_counts = []
    print(f"Analyzing {len(videos)} videos from {video_dir}...")

    for video in videos:
        try:
            cap = cv2.VideoCapture(video)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_counts.append(count)
            cap.release()
        except Exception as e:
            print(f"Error reading {video}: {e}")

    if not frame_counts:
        print("No valid videos found")
        return

    frame_counts = np.array(frame_counts)

    print(f"\n{'='*60}")
    print(f"Frame Count Statistics (n={len(frame_counts)} videos)")
    print(f"{'='*60}")
    print(f"Min:        {frame_counts.min():>6} frames")
    print(f"Max:        {frame_counts.max():>6} frames")
    print(f"Mean:       {frame_counts.mean():>6.1f} frames")
    print(f"Median:     {np.median(frame_counts):>6.0f} frames")
    print(f"25th %ile:  {np.percentile(frame_counts, 25):>6.0f} frames")
    print(f"75th %ile:  {np.percentile(frame_counts, 75):>6.0f} frames")
    print(f"90th %ile:  {np.percentile(frame_counts, 90):>6.0f} frames")

    # Show recommendations for different interval_frame settings
    print(f"\n{'='*60}")
    print("Recommended settings for fine-tuning:")
    print(f"{'='*60}")

    for interval in [1, 2, 3, 4]:
        for n_frames in [14, 21, 28, 35, 42]:
            required_span = (n_frames - 1) * interval + 1
            coverage = (frame_counts >= required_span).sum() / len(frame_counts) * 100
            print(f"sample_n_frames={n_frames:2d}, interval_frame={interval} → "
                  f"span={required_span:3d} frames, covers {coverage:5.1f}% of videos")

    return frame_counts

if __name__ == "__main__":
    import sys

    # Default to training pose directory
    video_dir = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/training/pose"

    if len(sys.argv) > 1:
        video_dir = sys.argv[1]

    analyze_videos(video_dir, sample_size=200)
