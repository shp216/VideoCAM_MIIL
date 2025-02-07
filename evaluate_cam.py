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
