import os
import cv2
import mediapipe as mp
import numpy as np
from src.data.features import knee_valgus_angle

# Paths
RAW_VIDEOS_DIR = "data/raw_videos"
KEYPOINTS_DIR = "data/keypoints"
FEATURES_DIR = "data/features"

os.makedirs(KEYPOINTS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

def process_video(video_path, video_name):
    cap = cv2.VideoCapture(video_path)
    keypoints_all = []
    angles = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            keypoints = [(lm.x, lm.y, lm.z) for lm in landmarks]
            keypoints_all.append(keypoints)

            # Example: right side indices
            hip = np.array(keypoints[24][:2])   # Right hip
            knee = np.array(keypoints[26][:2])  # Right knee
            ankle = np.array(keypoints[28][:2]) # Right ankle

            angle_valgus = knee_valgus_angle(hip, knee, ankle)
            angles.append(angle_valgus)

    cap.release()

    # Save outputs
    keypoints_path = os.path.join(KEYPOINTS_DIR, f"{video_name}_keypoints.npy")
    features_path = os.path.join(FEATURES_DIR, f"{video_name}_angles.npy")

    np.save(keypoints_path, keypoints_all)
    np.save(features_path, angles)

    print(f"[DONE] {video_name}: {len(keypoints_all)} frames, saved keypoints → {keypoints_path}, angles → {features_path}")


def main():
    videos = [f for f in os.listdir(RAW_VIDEOS_DIR) if f.endswith(".mp4")]
    if not videos:
        print("⚠️ No videos found in data/raw_videos/")
        return

    for video_file in videos:
        video_path = os.path.join(RAW_VIDEOS_DIR, video_file)
        video_name = os.path.splitext(video_file)[0]
        process_video(video_path, video_name)


if __name__ == "__main__":
    main()
