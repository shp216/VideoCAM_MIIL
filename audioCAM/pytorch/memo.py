import os
import torch

def find_max_in_pt_files(directory):
    """
    Load all .pt files in a directory and find the maximum value across all tensors.
    
    Args:
        directory (str): Path to the directory containing .pt files.
    
    Returns:
        float: The maximum value found across all tensors in the .pt files.
    """
    max_value = float('-inf')  # Initialize with the smallest possible value

    for file_name in os.listdir(directory):
        if file_name.endswith('.pt'):  # Process only .pt files
            file_path = os.path.join(directory, file_name)
            try:
                tensor = torch.load(file_path)
                max_value = max(max_value, tensor.max().item())
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    
    return max_value

# Example usage
directory_path = "./test_dataset_real/AVsync_audios_2s_cam"  # Replace with your directory path
max_value = find_max_in_pt_files(directory_path)
print(f"The maximum value across all .pt files is: {max_value}")
