import os
import cv2
from PIL import Image

def get_ref_frame_from_videos(source_video_folder: str, ref_frame_destination_folder: str = None):
    ref_frames = []
    valid_extensions = {".mp4", ".avi", ".mov", ".mkv"}  # Add or remove extensions as needed

    # Sorting files based on frame number
    sorted_files = os.listdir(source_video_folder)
    print(f"Video files: {sorted_files}")

    for filename in sorted_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            video_path = os.path.join(source_video_folder, filename)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ref_frames.append(Image.fromarray(frame))
                print(f"Shape of ref frame from {filename}: {frame.shape}")
            cap.release()

            basename = os.path.basename(filename)
            ref_frame_path = os.path.join(ref_frame_destination_folder, f"{os.path.splitext(basename)[0]}.png")
            print(f"Saving reference frame to: {ref_frame_path}")
            cv2.imwrite(os.path.join(ref_frame_destination_folder, ref_frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return ref_frames
    

if __name__== "__main__":
    folder_path = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/asl"
    destination = "/restricted/projectnb/cs599dg/Pose2Sign/ASL_Citizen/validation/ref_frames"
    ref_frames = get_ref_frame_from_videos(folder_path, destination)
    print(f"Total reference frames extracted: {len(ref_frames)}")