import cv2
import tempfile
import os
from pathlib import Path


def extract_frames(video_path: str, interval_seconds: float = 2.0, max_frames: int = 12) -> tuple[list[str], float, str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        raise ValueError("Invalid FPS detected in video")

    duration = total_frames / fps
    frame_interval = int(fps * interval_seconds)
    if frame_interval < 1:
        frame_interval = 1

    temp_dir = tempfile.mkdtemp(prefix="video_frames_")
    frame_paths = []
    frame_index = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                if max_frames and saved_count >= max_frames:
                    break
                frame_path = os.path.join(temp_dir, f"frame_{saved_count:05d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                saved_count += 1

            frame_index += 1
    finally:
        cap.release()

    if not frame_paths:
        raise RuntimeError("No frames could be extracted from the video")

    return frame_paths, duration, temp_dir
