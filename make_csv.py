# import os
# import pandas as pd

# # MP4 파일이 있는 폴더 경로
# video_folder = "MMG_test/Random_Vggsound_Video"

# # CSV 저장 경로
# output_csv = "MMG_test/Random_Vggsound_Video_filenames.csv"

# # 폴더 내 모든 MP4 파일 검색
# mp4_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

# # 확장자 제거 (파일명만 추출)
# file_names = [os.path.splitext(f)[0] for f in mp4_files]

# # DataFrame 생성
# df = pd.DataFrame({"FileName": file_names})

# # CSV 파일로 저장
# df.to_csv(output_csv, index=False)

# print(f"CSV 파일 저장 완료: {output_csv}")

import torch
import os

# 폴더 리스트
#folders = ["MMG_test/BASE_CAM", "MMG_test/MMG_CAM", "MMG_test/MMG+LoRA_CAM", "MMG_test/Random_Vggsound_CAM", "MMG_test/Curated_Vggsound_CAM"]
folders = ["test_dataset_real/AVsync_audios_0s_cam"]
# 각 폴더에서 .pt 파일을 로드하고 전체 max/mean 값을 계산
for folder in folders:
    folder_path = os.path.join(os.getcwd(), folder)  # 현재 디렉토리 기준으로 폴더 경로 설정
    if not os.path.exists(folder_path):
        print(f"폴더 {folder}가 존재하지 않습니다. 건너뜁니다.")
        continue

    all_cam_values = []  # 전체 CAM 값을 저장할 리스트
    
    # 폴더 내의 모든 .pt 파일 처리
    for file in os.listdir(folder_path):
        if file.endswith(".pt"):
            file_path = os.path.join(folder_path, file)
            cam_data = torch.load(file_path, map_location="cpu")  # CPU에서 로드
            if isinstance(cam_data, torch.Tensor):  # 텐서인지 확인
                all_cam_values.append(cam_data.flatten())  # 1D 텐서로 변환하여 저장
            else:
                print(f"경고: {file}에서 로드된 데이터가 Tensor가 아닙니다. 건너뜁니다.")

    if not all_cam_values:
        print(f"폴더 {folder}에 유효한 .pt 파일이 없습니다.")
        continue

    # 모든 데이터를 하나의 텐서로 합치기
    all_cam_tensor = torch.cat(all_cam_values)  # 모든 값들을 하나의 1D 텐서로 결합

    # 전체에서 max와 mean 값 계산
    global_max = all_cam_tensor.max().item()
    global_mean = all_cam_tensor.mean().item()

    # 결과 출력
    print(f"[{folder}]")
    print(f"  전체 데이터의 Max 값: {global_max:.6f}")
    print(f"  전체 데이터의 Mean 값: {global_mean:.6f}")
