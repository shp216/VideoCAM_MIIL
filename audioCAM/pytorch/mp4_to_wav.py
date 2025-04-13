# import os
# import pandas as pd

# # MP4 파일이 있는 폴더 경로
# video_folder = "MMG_test/MMG_Video"

# # CSV 저장 경로
# output_csv = "MMG_test/Video_filenames.csv"

# # 폴더 내 모든 MP4 파일 검색
# mp4_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# # 확장자 제거 (파일명만 추출)
# file_names = [os.path.splitext(f)[0] for f in mp4_files]

# # DataFrame 생성
# df = pd.DataFrame({"FileName": file_names})

# # CSV 파일로 저장
# df.to_csv(output_csv, index=False)

# print(f"CSV 파일 저장 완료: {output_csv}")

import os
import subprocess

def extract_audio_ffmpeg(video_folder, audio_folder):
    # 오디오 폴더가 없으면 생성
    os.makedirs(audio_folder, exist_ok=True)
    
    # 비디오 폴더 내 모든 파일 탐색
    for file in os.listdir(video_folder):
        if file.endswith(".mp4"):  # mp4 파일만 처리
            video_path = os.path.join(video_folder, file)
            audio_path = os.path.join(audio_folder, os.path.splitext(file)[0] + ".wav")
            
            try:
                # FFmpeg 명령어 실행
                command = [
                    "ffmpeg", "-i", video_path, "-ac", "2", "-ar", "44100", "-b:a", "1411k", "-y", audio_path
                ]
                subprocess.run(command, check=True)
                print(f"[SUCCESS] {file} -> {audio_path}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] {file}: {e}")

# 실행 경로 설정
video_folder = "MMG_test/Curated_Vggsound_Video"
audio_folder = "MMG_test/Curated_Vggsound_Wav"
extract_audio_ffmpeg(video_folder, audio_folder)


