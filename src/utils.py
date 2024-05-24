import cv2
import torch

from pathlib import Path

def  get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Source https://github.com/huggingface/diffusers/blob/main/src/diffusers/utils/testing_utils.py with adaption
def export_to_video(video_frames: list , filename: str = None, fps: int = 8) -> str:
    """Convert video frames to video."""
    Path("./outputs").mkdir(parents=True, exist_ok=True)
    output_video_path = f"./outputs/{filename}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    return output_video_path