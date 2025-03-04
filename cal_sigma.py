import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

class GradCamProcessor:
    def __init__(self, csv_path, audio_cam_dir):
        self.data = pd.read_csv(csv_path)
        self.audio_cam_dir = audio_cam_dir

    def calculate_gradcam_stats(self):
        sum_values = 0  # To store sum of values
        sum_squared_values = 0  # To store sum of squared values
        total_count = 0  # Total number of Grad-CAM values
        global_max = float('-inf')  # To track the global maximum value
        processed_count = 0  # To track the number of successfully processed files

        for idx in tqdm(range(len(self.data)), desc="Processing Grad-CAM files"):
            # Get the row and corresponding video file name
            row = self.data.iloc[idx]
            video_file = row['FileName']

            # Replace .mp4 with .pt to find the corresponding audio CAM file
            gradcam_file_name = video_file + ".pt"
            gradcam_path = os.path.join(self.audio_cam_dir, gradcam_file_name)

            # Check if the Grad-CAM file exists
            if not os.path.exists(gradcam_path):
                print(f"Grad-CAM file not found: {gradcam_path}")
                continue

            # Load the Grad-CAM data
            gradcam = torch.load(gradcam_path)

            # Flatten and convert the Grad-CAM data to NumPy array
            gradcam_values = gradcam.numpy().flatten()
            # Update global maximum value
            global_max = max(global_max, gradcam_values.max())

            # Update sums for incremental std calculation
            sum_values += gradcam_values.sum()
            sum_squared_values += (gradcam_values ** 2).sum()
            total_count += gradcam_values.size

            # Increment processed count
            processed_count += 1

        # Calculate mean and standard deviation
        mean = sum_values / total_count
        variance = (sum_squared_values / total_count) - (mean ** 2)
        std = np.sqrt(variance)

        return mean, std, global_max, processed_count

# Example usage
#csv_path = "./test_dataset_real/test_videos_filenames.csv"
csv_path = "./MMG_test/Curated_Vggsound_Video_filenames.csv"

audio_cam_dir = "./MMG_test/Curated_Vggsound_CAM"

processor = GradCamProcessor(csv_path, audio_cam_dir)
gradcam_mean, gradcam_std, global_max, processed_count = processor.calculate_gradcam_stats()

print(f"Mean of Grad-CAM values: {gradcam_mean}")
print(f"Standard Deviation of Grad-CAM values: {gradcam_std}")
print(f"Global Maximum of Grad-CAM values: {global_max}")
print(f"Total Processed Files: {processed_count}")




# import os
# import torch

# # 폴더 경로 설정
# curated_folder = "./MMG_test/Curated_Vggsound_CAM"
# random_folder = "./MMG_test/Random_Vggsound_CAM"

# # Curated 폴더에 있는 파일 목록 가져오기
# curated_files = sorted(os.listdir(curated_folder))

# # 결과 저장용 리스트
# mismatch_files = []

# # 파일 비교
# for file_name in curated_files:
#     curated_path = os.path.join(curated_folder, file_name)
#     random_path = os.path.join(random_folder, file_name)

#     # 두 번째 폴더에 동일한 파일이 있는지 확인
#     if os.path.exists(random_path):
#         curated_tensor = torch.load(curated_path)
#         random_tensor = torch.load(random_path)

#         # 값 비교 (torch.equal은 완전히 동일한지 확인)
#         if not torch.equal(curated_tensor, random_tensor):
#             mismatch_files.append(file_name)
#     else:
#         print(f"파일 없음: {random_path}")

# # 결과 출력
# if mismatch_files:
#     print(f"값이 다른 파일 수: {len(mismatch_files)}")
#     print("값이 다른 파일 목록:")
#     for file in mismatch_files:
#         print(file)
# else:
#     print("모든 파일이 동일합니다.")
