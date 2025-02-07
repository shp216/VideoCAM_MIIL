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
csv_path = "./test_dataset_real/test_videos_filenames.csv"
audio_cam_dir = "./test_dataset_real/AVsync_audios_2s_cam"

processor = GradCamProcessor(csv_path, audio_cam_dir)
gradcam_mean, gradcam_std, global_max, processed_count = processor.calculate_gradcam_stats()

print(f"Mean of Grad-CAM values: {gradcam_mean}")
print(f"Standard Deviation of Grad-CAM values: {gradcam_std}")
print(f"Global Maximum of Grad-CAM values: {global_max}")
print(f"Total Processed Files: {processed_count}")


