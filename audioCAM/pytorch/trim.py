import pandas as pd

# CSV 파일 불러오기
csv_file = "Curated_VGG_IoU.csv"  # CSV 파일 경로를 입력하세요
df = pd.read_csv(csv_file)

# 'iou_score' 컬럼의 평균 계산
if 'iou_score' in df.columns:
    mean_iou = df['iou_score'].mean()
    print(f"IOU Score 평균: {mean_iou:.4f}")
else:
    print("Error: 'iou_score' 컬럼이 CSV 파일에 없습니다.")
