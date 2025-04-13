import os

# 부모 폴더 경로
base_dir = "./avsync_eval_dataset/Random"

# 하위 폴더 목록
subfolders = ["audio", "audiocam", "combined", "video"]

def rename_files():
    for subfolder in subfolders:
        folder_path = os.path.join(base_dir, subfolder)

        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (not found)")
            continue

        # 폴더 내 파일 목록 가져오기
        files = sorted(os.listdir(folder_path))  
        for filename in files:
            old_path = os.path.join(folder_path, filename)

            # 파일 확장자 분리
            if "." in filename:
                name, ext = filename.rsplit(".", 1)
                new_filename = f"random{name}.{ext}"
            else:
                new_filename = f"random{filename}"  # 확장자가 없을 경우

            new_path = os.path.join(folder_path, new_filename)

            # 파일명 변경
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Error renaming {filename}: {e}")

if __name__ == "__main__":
    rename_files()
