import os

def print_top_folders(folder_path, num_folders=5):
    try:
        if not os.path.exists(folder_path):
            print(f"Error: The folder path '{folder_path}' does not exist.")
            return
        
        folders = []
        for entry in os.scandir(folder_path):
            print(f"Checking: {entry.name}")
            if entry.is_dir():
                folders.append(entry.name)
            if len(folders) == num_folders:
                break

        if folders:
            print("Top folders:")
            for folder in folders:
                print(folder)
        else:
            print(f"No folders found in the path '{folder_path}'.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

folder_path = "./audio_cam"  # 경로를 정확히 설정
print_top_folders(folder_path)


# import os
# import pandas as pd

# def filter_existing_videos(csv_file, folder_path, output_csv):
#     # Read the CSV file
#     df = pd.read_csv(csv_file)

#     # Ensure the column 'FileName' exists
#     if 'FileName' not in df.columns:
#         raise ValueError("The input CSV file must have a 'FileName' column.")

#     # Get the list of actual .pt files in the folder (excluding directory entries)
#     existing_files = set(f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pt'))

#     # Filter the DataFrame to keep only rows where corresponding .pt file exists
#     filtered_df = df[df['FileName'].apply(lambda x: os.path.splitext(x)[0] + '.pt').isin(existing_files)]

#     # Save the filtered DataFrame to the output CSV
#     filtered_df.to_csv(output_csv, index=False)

#     print(f"Filtered data saved to {output_csv}")

# # Example usage
# csv_file = "./video_filenames.csv"  # Input CSV with FileName column
# folder_path = "./audio_cam"        # Folder containing actual .pt files
# output_csv = "./audiovisual_filenames.csv"  # Output filtered CSV

# filter_existing_videos(csv_file, folder_path, output_csv)


# import os
# import pandas as pd

# # Path to the folder containing the video files
# folder_path = "../video_25fps"

# # Output CSV file name
# output_csv = "../video_filenames.csv"

# # Get all file names in the folder
# file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# # Create a DataFrame with a column named 'FileName'
# df = pd.DataFrame(file_names, columns=["FileName"])

# # Save the DataFrame to a CSV file
# df.to_csv(output_csv, index=False)

# print(f"CSV file created successfully: {output_csv}")
