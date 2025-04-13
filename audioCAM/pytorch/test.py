# import os
# import subprocess
# from concurrent.futures import ThreadPoolExecutor

# # 입력 폴더와 출력 폴더 설정
# input_folder = "./vggsound_sparse_10s_50"  # 원본 비디오 폴더
# output_video_folder = "./video_trimmed"  # 잘린 비디오 저장 폴더
# output_audio_folder = "./audio_trimmed"  # 잘린 오디오 저장 폴더
# os.makedirs(output_video_folder, exist_ok=True)
# os.makedirs(output_audio_folder, exist_ok=True)

# # 자를 길이 설정 (초 단위)
# trim_length = 3.2  # 3.2초

# def process_video_and_audio(file_name):
#     input_path = os.path.join(input_folder, file_name)
#     output_video_path = os.path.join(output_video_folder, file_name)
#     output_audio_path = os.path.join(output_audio_folder, os.path.splitext(file_name)[0] + ".wav")

#     try:
#         # FFmpeg 명령어로 비디오 길이 가져오기
#         duration_cmd = [
#             "ffprobe", "-v", "error", "-show_entries", "format=duration", 
#             "-of", "default=noprint_wrappers=1:nokey=1", input_path
#         ]
#         duration = float(subprocess.check_output(duration_cmd).strip())
        
#         if duration < trim_length:
#             print(f"Skipping {file_name}: Duration is less than 3.2 seconds")
#             return
        
#         # 시작 시간 계산
#         start_time = (duration / 2) - (trim_length / 2)
#         start_time = max(0, start_time)

#         # FFmpeg 비디오 트리밍 명령어
#         ffmpeg_video_cmd = [
#             "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(trim_length),
#             "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-threads", "4", output_video_path,
#             "-y"
#         ]
        
#         # FFmpeg 오디오 추출 및 트리밍 명령어
#         ffmpeg_audio_cmd = [
#             "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(trim_length),
#             "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-threads", "4", output_audio_path,
#             "-y"
#         ]
        
#         # 비디오와 오디오 처리
#         subprocess.run(ffmpeg_video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         subprocess.run(ffmpeg_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print(f"Processed: {file_name} (video and audio)")
#     except Exception as e:
#         print(f"Error processing {file_name}: {e}")

# # 비디오 파일 목록 가져오기
# video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

# # 병렬 처리
# with ThreadPoolExecutor(max_workers=32) as executor:  # 워커 수 조정
#     executor.map(process_video_and_audio, video_files)

# print("All video and audio files have been processed.")


import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# 입력 폴더와 출력 폴더 설정
input_folder = "./vggsound_sparse_10s_50"  # 원본 비디오 폴더
output_video_folder = "./video_trimmed"  # 잘린 비디오 저장 폴더
output_audio_folder = "./audio_trimmed"  # 잘린 오디오 저장 폴더
os.makedirs(output_video_folder, exist_ok=True)
os.makedirs(output_audio_folder, exist_ok=True)
# 자를 길이 설정 (초 단위)
trim_length = 3.2  # 3.2초

def process_video_and_audio(file_name):
    input_path = os.path.join(input_folder, file_name)
    output_video_path = os.path.join(output_video_folder, file_name)
    output_audio_path = os.path.join(output_audio_folder, os.path.splitext(file_name)[0] + ".wav")

    try:
        # FFmpeg 명령어로 비디오 길이 가져오기
        duration_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ]
        duration = float(subprocess.check_output(duration_cmd).strip())
        
        if duration < trim_length:
            print(f"Skipping {file_name}: Duration is less than 3.2 seconds")
            return
        
        # 시작 시간 계산
        start_time = (duration / 2) - (trim_length / 2)
        start_time = max(0, start_time)

        # FFmpeg 비디오 트리밍 명령어
        ffmpeg_video_cmd = [
            "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(trim_length),
            "-c:v", "libx264", "-c:a", "aac", "-strict", "experimental", "-threads", "4", output_video_path,
            "-y"
        ]
        
        # FFmpeg 오디오 추출 및 트리밍 명령어
        ffmpeg_audio_cmd = [
            "ffmpeg", "-i", input_path, "-ss", str(start_time), "-t", str(trim_length),
            "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", "-threads", "4", output_audio_path,
            "-y"
        ]
        
        # 비디오와 오디오 처리
        subprocess.run(ffmpeg_video_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(ffmpeg_audio_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Processed: {file_name} (video and audio)")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# 비디오 파일 목록 가져오기
video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

# 병렬 처리
with ThreadPoolExecutor(max_workers=16) as executor:  # 워커 수 조정
    executor.map(process_video_and_audio, video_files)

print("All video and audio files have been processed.")
