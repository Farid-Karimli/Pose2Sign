import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import os

def extract_pose_video(input_path, output_path, draw_on_blank=True, display_progress=True):
    """
    Extract pose keypoints from a video using MediaPipe and save as a new pose video.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hand = mp.solutions.hands

    # Initialize MediaPipe Pose
    pose = mp_pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    hands = mp_hand.Hands(static_image_mode=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Read input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare output folder and video writer
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if display_progress:
        pbar = tqdm(total=frame_count, desc=f"Processing {os.path.basename(input_path)}", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR -> RGB for MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(image_rgb)
        hands_results = hands.process(image_rgb)

        # Draw pose on blank or original background
        if draw_on_blank:
            pose_frame = np.ones_like(frame, dtype=np.uint8) * 255  # white background
        else:
            pose_frame = frame.copy()

        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                pose_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

        if hands_results.multi_hand_landmarks:

            for num, hand in enumerate(hands_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    pose_frame,
                    hand,
                    mp_hand.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                )

        out.write(pose_frame)

        if display_progress:
            pbar.update(1)

    if display_progress:
        pbar.close()

    cap.release()
    out.release()
    pose.close()

    print(f"✅ Pose video saved to: {output_path}")


if __name__ == "__main__":
    asl_video_path = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/videos"
    asl_video_files = os.listdir(asl_video_path)

    pose_video_path = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/pose_videos"
    
    SUBSET = 1000

    if SUBSET:
        asl_video_files = asl_video_files[:SUBSET]

    for video_file in tqdm(asl_video_files, desc="Processing videos...", unit="video"):
        input_video = os.path.join(asl_video_path, video_file)
        output_video = os.path.join(pose_video_path, video_file)

        extract_pose_video(input_video, output_video, draw_on_blank=True, display_progress=False)
