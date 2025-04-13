import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.encoder_utils import patch_config, get_pretrained
from cam_dataset import VideoAudioDataset  # Assuming the dataset is implemented in this file
from tqdm import tqdm
from omegaconf import OmegaConf
from utils import seed_everything
from encoder.phi import Phi
import random
import numpy as np
from accelerate import Accelerator
from utils import custom_collate_fn
import matplotlib.pyplot as plt
import config
import pandas as pd

def evaluate(phi, video_encoder, dataloader, use_half_precision=False):
    """
    Evaluate the model using the provided dataloader.

    Args:
        phi (torch.nn.Module): The main model to evaluate.
        video_encoder (torch.nn.Module): The video encoder module.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        use_half_precision (bool): Whether to use mixed precision during evaluation.

    Returns:
        float: Average losses over the evaluation dataset.
    """
    phi.eval()  # Set model to evaluation mode
    video_encoder.eval()  # Set video encoder to evaluation mode

    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0
    
    total_iou1 = 0.0
    total_iou2 = 0.0
    total_iou3 = 0.0
    total_samples = 0

    with torch.no_grad():  # No gradient computation during evaluation
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating", dynamic_ncols=False)):
            # Move data to GPU
            video = batch['video']
            gradcam_0s = batch['gradcam_0s']
            gradcam_1s = batch['gradcam_1s']
            gradcam_2s = batch['gradcam_2s']
            attention_mask = batch['attention_mask']

            # Forward pass through the video encoder
            with torch.autocast('cuda', enabled=use_half_precision):
                video = video.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
                attention_mask = attention_mask.permute(0, 1, 3, 2, 4, 5)

                hint, _, hint_mask = video_encoder(video, for_loop=False, cont_mask=attention_mask)
                hint_mask_flip = hint_mask[..., 0, 0]
                hint_mask_reduced = ~hint_mask_flip  # True -> False, False -> True

                B, S, tv, D = hint.shape
                hint = hint.view(B, S * tv, D)

                # Process hint mask
                B, S, T = hint_mask_reduced.shape
                num_heads = 8
                seq_len = S * T

                attn_mask_flat = hint_mask_reduced.reshape(B, seq_len)
                attn_mask_3d = attn_mask_flat.unsqueeze(1).expand(B, seq_len, seq_len)
                attn_mask_final = attn_mask_3d.repeat(num_heads, 1, 1)

                # Forward pass through phi
                pred_cam = phi(hint.float(), attn_mask=attn_mask_flat).squeeze(2)
                pred_cam = pred_cam.permute(0, 2, 1)  # (B, 527, Sx8)

            # Compute loss for each sample in the batch
            for i in range(pred_cam.shape[0]):
                gradcam_valid_0s = gradcam_0s[i]
                gradcam_valid_1s = gradcam_1s[i]
                gradcam_valid_2s = gradcam_2s[i]
                gradcam_mask = batch['gradcam_mask'][i]
                gradcam_valid_0s = gradcam_valid_0s[:, gradcam_mask]
                gradcam_valid_1s = gradcam_valid_1s[:, gradcam_mask]
                gradcam_valid_2s = gradcam_valid_2s[:, gradcam_mask]

                pred_cam_mask = hint_mask_flip[i]
                valid_pred_cam_mask = pred_cam_mask.reshape(-1)
                valid_pred_cam = pred_cam[i, :, valid_pred_cam_mask]

                valid_pred_cam = valid_pred_cam.unsqueeze(0)
                interpolated_pred_cam = F.interpolate(
                    valid_pred_cam,
                    size=gradcam_valid_0s.shape[1],
                    mode="linear",
                    align_corners=False
                ).squeeze(0)

                # Calculate losses for each gradcam
                loss1 = F.mse_loss(interpolated_pred_cam, gradcam_valid_0s, reduction="mean")
                loss2 = F.mse_loss(interpolated_pred_cam, gradcam_valid_1s, reduction="mean")
                loss3 = F.mse_loss(interpolated_pred_cam, gradcam_valid_2s, reduction="mean")
                
                def pixelwise_soft_iou(cam1, cam2):
                    # Ensure both CAMs are numpy arrays
                    cam1 = cam1.cpu().numpy()
                    cam2 = cam2.cpu().numpy()
                    
                    # Compute minimum and maximum for each pixel
                    intersection = np.minimum(cam1, cam2)
                    union = np.maximum(cam1, cam2)
                    
                    # Mask for valid pixels (exclude areas where both CAMs are 0)
                    valid_mask = union > 0  # Union > 0 ensures we exclude both-zero pixels
                    
                    # Calculate pixel-wise IOU (avoid division by zero using valid_mask)
                    iou_per_pixel = np.zeros_like(cam1, dtype=np.float32)
                    iou_per_pixel[valid_mask] = intersection[valid_mask] / union[valid_mask]
                    
                    # Calculate mean IOU across valid pixels
                    mean_iou = np.mean(iou_per_pixel[valid_mask]) if np.any(valid_mask) else 0.0
                    
                    return mean_iou, iou_per_pixel

                iou1,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_0s)
                iou2,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_1s)
                iou3,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_2s)

                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss3 += loss3.item()
                
                total_iou1 += iou1
                total_iou2 += iou2
                total_iou3 += iou3
                total_samples += 1

    avg_loss1 = total_loss1 / total_samples if total_samples > 0 else float('inf')
    avg_loss2 = total_loss2 / total_samples if total_samples > 0 else float('inf')
    avg_loss3 = total_loss3 / total_samples if total_samples > 0 else float('inf')
    
    avg_iou1 = total_iou1 / total_samples if total_samples > 0 else 0
    avg_iou2 = total_iou2 / total_samples if total_samples > 0 else 0
    avg_iou3 = total_iou3 / total_samples if total_samples > 0 else 0
    
    print(f"Evaluation completed. Average Loss1: {avg_loss1:.4f}, Average Loss2: {avg_loss2:.4f}, Average Loss3: {avg_loss3:.4f}")
    print(f"Evaluation completed. Soft IoU: {avg_loss1:.4f}, IoU1: {avg_iou1:.4f}, IoU2: {avg_iou2:.4f}, IoU3: {avg_iou3:.4f}")

    # Restore training mode
    phi.train()

    return avg_loss1, avg_loss2, avg_loss3, avg_iou1, avg_iou2, avg_iou3



def save_all_class_time_cam_image(aggregated_cams_array, class_names, save_dir, audio_file):
    """Save CAMs as an image with classes on y-axis and time on x-axis."""
    audio_name = audio_file
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 2 + len(class_names) * 0.5))
    im = ax.imshow(aggregated_cams_array, aspect='auto', origin='upper', cmap='jet', interpolation='nearest')

    for idx in range(1, len(class_names)):
        ax.hlines(y=idx - 0.5, xmin=0, xmax=aggregated_cams_array.shape[1] - 1, colors='white', linestyles='-', linewidth=1.0)

    ax.set_yticks(np.arange(len(class_names))) 
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Time Frames')
    ax.set_title('Aggregated CAM over Time for All Classes')

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{audio_name}_all_classes_combined.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Aggregated CAM image saved at {save_path}")
    

def save_top5_class_time_cam_image(aggregated_cams_array, class_names, save_dir, audio_file, mode="pred"):
    """Save CAMs as an image with the top 5 highest-scoring classes on the y-axis and time on the x-axis."""
    audio_name = audio_file
    os.makedirs(save_dir, exist_ok=True)

    # 각 클래스별 CAM의 평균값을 계산하여 상위 5개 클래스 선택
    avg_scores = np.mean(aggregated_cams_array, axis=1)  # 각 클래스별 평균 CAM 값 계산
    top5_indices = np.argsort(avg_scores)[::-1][:5]  # 상위 5개 클래스 인덱스 가져오기
    top5_class_names = [class_names[i] for i in top5_indices]  # 해당 클래스 이름 가져오기
    top5_cams_array = aggregated_cams_array[top5_indices, :]  # CAM 값 추출

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 4))  # 크기 조정 (5개만 표시되므로 작게)
    im = ax.imshow(top5_cams_array, aspect='auto', origin='upper', cmap='jet', interpolation='nearest')

    # 클래스 경계선 표시
    for idx in range(1, len(top5_class_names)):
        ax.hlines(y=idx - 0.5, xmin=0, xmax=top5_cams_array.shape[1] - 1, colors='white', linestyles='-', linewidth=1.0)

    ax.set_yticks(np.arange(len(top5_class_names))) 
    ax.set_yticklabels(top5_class_names)

    ax.set_xlabel('Time Frames')
    if mode == "pred":
        ax.set_title('Prediction CAM')
    else:
        ax.set_title('Ground Truth CAM')

    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)
    plt.tight_layout()

    # 이미지 저장
    save_path = os.path.join(save_dir, f"{audio_name}_top5_classes.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    print(f"Top 5 Aggregated CAM image saved at {save_path}")


def pixelwise_soft_iou(cam1, cam2):
    # Ensure both CAMs are numpy arrays
    cam1 = cam1.cpu().numpy()
    cam2 = cam2.cpu().numpy()
    
    # Compute minimum and maximum for each pixel
    intersection = np.minimum(cam1, cam2)
    union = np.maximum(cam1, cam2)
    
    # Mask for valid pixels (exclude areas where both CAMs are 0)
    valid_mask = union > 0  # Union > 0 ensures we exclude both-zero pixels
    
    # Calculate pixel-wise IOU (avoid division by zero using valid_mask)
    iou_per_pixel = np.zeros_like(cam1, dtype=np.float32)
    iou_per_pixel[valid_mask] = intersection[valid_mask] / union[valid_mask]
    
    # Calculate mean IOU across valid pixels
    mean_iou = np.mean(iou_per_pixel[valid_mask]) if np.any(valid_mask) else 0.0
    
    return mean_iou, iou_per_pixel

import numpy as np

def pixelwise_soft_iou_log_scaled(cam1, cam2):
    # Ensure both CAMs are numpy arrays
    cam1 = cam1.cpu().numpy()
    cam2 = cam2.cpu().numpy()
    
    # Apply ln(1 + x) scaling
    cam1 = np.log1p(10*cam1)
    cam2 = np.log1p(10*cam2)
    
    cam1 = cam1 / np.max(cam1)
    cam2 = cam2 / np.max(cam2)
    
    # Compute minimum and maximum for each pixel
    intersection = np.minimum(cam1, cam2)
    union = np.maximum(cam1, cam2)
    
    # Mask for valid pixels (exclude areas where both CAMs are 0)
    valid_mask = union > 0  # Union > 0 ensures we exclude both-zero pixels
    
    # Calculate pixel-wise IOU (avoid division by zero using valid_mask)
    iou_per_pixel = np.zeros_like(cam1, dtype=np.float32)
    iou_per_pixel[valid_mask] = intersection[valid_mask] / union[valid_mask]
    
    # Calculate mean IOU across valid pixels
    mean_iou = np.mean(iou_per_pixel[valid_mask]) if np.any(valid_mask) else 0.0
    
    return mean_iou, iou_per_pixel


def evaluate_just1(phi, video_encoder, dataloader, use_half_precision=False):
    """
    Evaluate the model using the provided dataloader.

    Args:
        phi (torch.nn.Module): The main model to evaluate.
        video_encoder (torch.nn.Module): The video encoder module.
        dataloader (torch.utils.data.DataLoader): DataLoader for evaluation.
        use_half_precision (bool): Whether to use mixed precision during evaluation.

    Returns:
        float: Average losses over the evaluation dataset.
    """
    phi.eval()  # Set model to evaluation mode
    video_encoder.eval()  # Set video encoder to evaluation mode

    img_save_dir = "./vggsound_eval_dataset/gt_vggsound_sparse_test/pred_cam_images"
    img_save_dir_gt = "./vggsound_eval_dataset/gt_vggsound_sparse_test/gt_cam_images"
    iou_scores = []  # IOU 점수를 저장할 리스트
    with torch.no_grad():  # No gradient computation during evaluation
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating", dynamic_ncols=False)):
            # Move data to GPU
            video = batch['video']
            gradcam_0s = batch['gradcam_0s']
            # gradcam_1s = batch['gradcam_1s']
            attention_mask = batch['attention_mask']
            filename = batch['FileName']

            # Forward pass through the video encoder
            with torch.autocast('cuda', enabled=use_half_precision):
                video = video.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
                attention_mask = attention_mask.permute(0, 1, 3, 2, 4, 5)

                hint, _, hint_mask = video_encoder(video, for_loop=False, cont_mask=attention_mask)
                hint_mask_flip = hint_mask[..., 0, 0]
                hint_mask_reduced = ~hint_mask_flip  # True -> False, False -> True

                B, S, tv, D = hint.shape
                hint = hint.view(B, S * tv, D)

                # Process hint mask
                B, S, T = hint_mask_reduced.shape
                num_heads = 8
                seq_len = S * T

                attn_mask_flat = hint_mask_reduced.reshape(B, seq_len)
                attn_mask_3d = attn_mask_flat.unsqueeze(1).expand(B, seq_len, seq_len)
                attn_mask_final = attn_mask_3d.repeat(num_heads, 1, 1)

                # Forward pass through phi
                pred_cam = phi(hint.float(), attn_mask=attn_mask_flat).squeeze(2)
                pred_cam = pred_cam.permute(0, 2, 1)  # (B, 527, Sx8)

            # Compute loss for each sample in the batch
            for i in range(pred_cam.shape[0]):
                gradcam_valid_0s = gradcam_0s[i]
                gradcam_mask = batch['gradcam_mask'][i]
                gradcam_valid_0s = gradcam_valid_0s[:, gradcam_mask]
               
                pred_cam_mask = hint_mask_flip[i]
                valid_pred_cam_mask = pred_cam_mask.reshape(-1)
                valid_pred_cam = pred_cam[i, :, valid_pred_cam_mask]

                valid_pred_cam = valid_pred_cam.unsqueeze(0)
                interpolated_pred_cam = F.interpolate(
                    valid_pred_cam,
                    size=gradcam_valid_0s.shape[1],
                    mode="linear",
                    align_corners=False
                ).squeeze(0)
                
                iou1,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_0s)
                #iou1,_ = pixelwise_soft_iou_log_scaled(interpolated_pred_cam, gradcam_valid_0s)

                
                labels = config.labels
                #save_top5_class_time_cam_image(interpolated_pred_cam.detach().cpu().numpy(), labels, img_save_dir, filename[i], mode="pred")
                #save_top5_class_time_cam_image(gradcam_valid_0s.detach().cpu().numpy(), labels, img_save_dir_gt, filename[i], mode="gt")
                iou_scores.append({"FileName": filename[i], "iou_score": iou1})

    # IOU 점수를 CSV 파일로 저장
    iou_scores_df = pd.DataFrame(iou_scores)
    #iou_scores_df.to_csv("./MMG_test/SIoU_scores.csv", index=False)

    print("IOU scores saved to csv")
    print(f"Mean IOU Score: {iou_scores_df['iou_score'].mean():.4f}")


                # # Calculate losses for each gradcam
                # loss1 = F.mse_loss(interpolated_pred_cam, gradcam_valid_0s, reduction="mean")
                # loss2 = F.mse_loss(interpolated_pred_cam, gradcam_valid_1s, reduction="mean")
                # loss3 = F.mse_loss(interpolated_pred_cam, gradcam_valid_2s, reduction="mean")
                
    #             def pixelwise_soft_iou(cam1, cam2):
    #                 # Ensure both CAMs are numpy arrays
    #                 cam1 = cam1.cpu().numpy()
    #                 cam2 = cam2.cpu().numpy()
                    
    #                 # Compute minimum and maximum for each pixel
    #                 intersection = np.minimum(cam1, cam2)
    #                 union = np.maximum(cam1, cam2)
                    
    #                 # Mask for valid pixels (exclude areas where both CAMs are 0)
    #                 valid_mask = union > 0  # Union > 0 ensures we exclude both-zero pixels
                    
    #                 # Calculate pixel-wise IOU (avoid division by zero using valid_mask)
    #                 iou_per_pixel = np.zeros_like(cam1, dtype=np.float32)
    #                 iou_per_pixel[valid_mask] = intersection[valid_mask] / union[valid_mask]
                    
    #                 # Calculate mean IOU across valid pixels
    #                 mean_iou = np.mean(iou_per_pixel[valid_mask]) if np.any(valid_mask) else 0.0
                    
    #                 return mean_iou, iou_per_pixel

    #             iou1,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_0s)
    #             iou2,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_1s)
    #             iou3,_ = pixelwise_soft_iou(interpolated_pred_cam, gradcam_valid_2s)

    #             total_loss1 += loss1.item()
    #             total_loss2 += loss2.item()
    #             total_loss3 += loss3.item()
                
    #             total_iou1 += iou1
    #             total_iou2 += iou2
    #             total_iou3 += iou3
    #             total_samples += 1

    # avg_loss1 = total_loss1 / total_samples if total_samples > 0 else float('inf')
    # avg_loss2 = total_loss2 / total_samples if total_samples > 0 else float('inf')
    # avg_loss3 = total_loss3 / total_samples if total_samples > 0 else float('inf')
    
    # avg_iou1 = total_iou1 / total_samples if total_samples > 0 else 0
    # avg_iou2 = total_iou2 / total_samples if total_samples > 0 else 0
    # avg_iou3 = total_iou3 / total_samples if total_samples > 0 else 0
    
    # print(f"Evaluation completed. Average Loss1: {avg_loss1:.4f}, Average Loss2: {avg_loss2:.4f}, Average Loss3: {avg_loss3:.4f}")
    # print(f"Evaluation completed. Soft IoU: {avg_loss1:.4f}, IoU1: {avg_iou1:.4f}, IoU2: {avg_iou2:.4f}, IoU3: {avg_iou3:.4f}")

    # # Restore training mode
    # phi.train()

    
