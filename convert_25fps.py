
import os
from concurrent.futures import ThreadPoolExecutor

def convert_video_to_25fps(input_path, output_path):
    """
    Convert a video file to 25fps and save it to the specified output path.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted video file.
    """
    try:
        cmd = f"ffmpeg -y -i {input_path} -vf fps=25 {output_path} -loglevel error"
        os.system(cmd)
    except Exception as e:
        print(f"[Error] Failed to convert {input_path} to 25fps: {e}")

def process_videos(input_folder, output_folder, max_workers=8):
    """
    Convert all videos in a folder to 25fps and save them to another folder.

    Args:
        input_folder (str): Path to the folder containing input video files.
        output_folder (str): Path to save the converted video files.
        max_workers (int): Maximum number of parallel processes.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_file in video_files:
            input_path = os.path.join(input_folder, video_file)
            output_path = os.path.join(output_folder, video_file)
            executor.submit(safe_convert_video, input_path, output_path)

def safe_convert_video(input_path, output_path):
    """
    Safely convert a video to 25fps, skipping on failure.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted video file.
    """
    try:
        convert_video_to_25fps(input_path, output_path)
    except Exception as e:
        print(f"[Error] Skipping {input_path} due to error: {e}")

if __name__ == "__main__":
    input_folder = "./test_dataset_real/AVsync_videos"  # Folder containing original videos
    output_folder = "./test_dataset_real/AVsync_videos_25fps"  # Folder to save converted videos
    max_workers = 64  # Number of parallel processes (adjust based on your system)

    process_videos(input_folder, output_folder, max_workers)
