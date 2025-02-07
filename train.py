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

import threading
from queue import Queue

class DynamicBatchDataLoader:
    def __init__(self, dataloader, max_prefetch=10):
        self.dataloader = dataloader
        self.max_prefetch = max_prefetch
        self.queue = Queue(max_prefetch)
        self.thread = threading.Thread(target=self._prefetch)
        self.thread.daemon = True
        self.thread.start()

    def _prefetch(self):
        for batch in self.dataloader:
            self.queue.put(batch)
        self.queue.put(None)  # End of iteration signal

    def __iter__(self):
        while True:
            batch = self.queue.get()
            if batch is None:
                break
            yield batch

    def __len__(self):
        return len(self.dataloader)



def main(args):
    ##################################################################### 
    # Original Setting From ReWas 
    #####################################################################
    seed = args.seed
    seed_everything(seed)
    control_type = args.control_type
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    cfg_path = f'./configs/cfg-{args.synchformer_exp}.yaml'
    synchformer_cfg = OmegaConf.load(cfg_path)
    synchformer_cfg = patch_config(synchformer_cfg)
    
    video_encoder = get_pretrained(args.synchformer_exp, 0)
    # phi = Phi()
    # resume_params = torch.load(args.phi_ckpt_path)
    # resume_new = {k.replace("module.",""): v for k, v in resume_params.items()}
    # phi.load_state_dict(resume_new)
    # Instantiate Phi and load pretrained weights
    phi = Phi()

    # Load pretrained weights
    resume_params = torch.load(args.phi_ckpt_path)
    resume_new = {k.replace("module.", ""): v for k, v in resume_params.items()}

    # Exclude weights for projection2
    filtered_params = {k: v for k, v in resume_new.items() if not k.startswith("projection2")}

    # Load the filtered weights
    phi.load_state_dict(filtered_params, strict=False)
    phi = nn.DataParallel(phi, device_ids=[i for i in range(torch.cuda.device_count())])

    #####################################################################
    # Dataset and DataLoader
    #####################################################################
    train_dataset = VideoAudioDataset(
        csv_file=args.csv_file,
        video_dir=args.video_dir,
        cam_dir=args.cam_dir,
        control_type=control_type,
        synchformer_exp=args.synchformer_exp,
        synchformer_cfg=synchformer_cfg,
        video_encoder=video_encoder,
        phi=phi,
        seed=args.seed,
        duration=args.duration,
        guidance_scale=args.guidance_scale,
        ddim_steps=args.ddim_steps,
        batchsize=args.batchsize,
        re_encode=args.re_encode,
        save_path=save_path
    )
    
    def custom_collate_fn(batch):
        filtered_batch = []  # 조건에 맞는 샘플을 저장할 리스트
        skipped_indices = []  # 제외된 샘플의 인덱스 저장

        # 유효한 샘플 필터링
        for idx, sample in enumerate(batch):
            try:
                # "duration"이 0보다 큰 경우만 유효한 샘플로 간주
                if "video" in sample and sample["duration"] > 0:
                    filtered_batch.append(sample)
                else:
                    skipped_indices.append(idx)
            except Exception as e:
                print(f"[Warning] Sample {idx} skipped due to error: {e}")
                skipped_indices.append(idx)

        # 로그 출력
        if skipped_indices:
            print(f"[Info] Skipped samples: {skipped_indices}")

        # 필터링된 샘플로 배치 생성
        if not filtered_batch:
            return {}  # 유효한 샘플이 없으면 빈 딕셔너리 반환

        collated_batch = {}

        # 각 샘플의 최대 길이 확인
        max_segments = max(sample["video"].shape[0] for sample in filtered_batch)
        max_time = max(sample["gradcam"].shape[1] for sample in filtered_batch)

        # 비디오, gradcam, attention_mask, gradcam_mask 생성
        videos = []
        gradcams = []
        attention_masks = []
        gradcam_masks = []
        v_ranges = []

        for sample in filtered_batch:
            # 비디오 패딩
            video = sample["video"]
            video_padding = torch.zeros(
                (max_segments - video.shape[0], *video.shape[1:]), dtype=video.dtype, device=video.device
            )
            videos.append(torch.cat([video, video_padding], dim=0))  # 패딩된 비디오 추가

            # gradcam 패딩
            gradcam = sample["gradcam"]
            gradcam_padding = torch.zeros(
                (gradcam.shape[0], max_time - gradcam.shape[1]), dtype=gradcam.dtype, device=gradcam.device
            )
            gradcams.append(torch.cat([gradcam, gradcam_padding], dim=1))  # 패딩된 gradcam 추가

            # attention_mask 패딩
            attention_mask = sample["attention_mask"]
            attention_mask_padding = torch.zeros(
                (max_segments - attention_mask.shape[0], *attention_mask.shape[1:]),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_masks.append(torch.cat([attention_mask, attention_mask_padding], dim=0))  # 패딩된 attention_mask 추가

            # gradcam_mask 생성
            gradcam_mask = torch.zeros(max_time, dtype=torch.bool, device=gradcam.device)
            gradcam_mask[:gradcam.shape[1]] = 1  # 유효한 시간 부분만 1로 설정
            gradcam_masks.append(gradcam_mask)

            # v_ranges 패딩
            current_v_ranges = sample["v_ranges"]
            v_range_padding = torch.zeros((max_segments - current_v_ranges.shape[0], 2), dtype=torch.long, device=current_v_ranges.device)
            v_ranges.append(torch.cat([current_v_ranges, v_range_padding], dim=0))  # 패딩된 v_ranges 추가

        # 텐서로 변환
        collated_batch["video"] = torch.stack(videos, dim=0)  # (B, max_segments, ...)
        collated_batch["gradcam"] = torch.stack(gradcams, dim=0)  # (B, 527, max_time)
        collated_batch["attention_mask"] = torch.stack(attention_masks, dim=0)  # (B, max_segments, segment_size_vframes, C, H, W)
        collated_batch["gradcam_mask"] = torch.stack(gradcam_masks, dim=0)  # (B, max_time)
        collated_batch["v_ranges"] = torch.stack(v_ranges, dim=0)  # (B, max_segments, 2)

        return collated_batch

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    dynamic_dataloader = DynamicBatchDataLoader(train_dataloader)

    #####################################################################
    # Optimizer and Scheduler
    #####################################################################
    #optimizer = optim.Adam(phi.parameters(), lr=args.lr, betas=(0.9, 0.999))
    optimizer = optim.AdamW(
    phi.parameters(),
    lr=args.lr,
    betas=(0.9, 0.999),
    weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 0.01  # 기본 weight_decay를 0.01로 설정
    )

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    #####################################################################
    # Training Loop
    #####################################################################
    phi.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}...")
        for step, batch in enumerate(tqdm(dynamic_dataloader, desc=f"Epoch {epoch + 1}")):
            # Move data to GPU
            video = batch['video'].cuda()
            gradcam = batch['gradcam'].cuda()
            # Randomly select a duration (1 to 5 seconds)

            # Print sliced shapes
            # print("Sliced video.shape: ", video.shape)
            # print("Sliced gradcam.shape: ", gradcam.shape)
            # print("attention_mask.shape: ", batch['attention_mask'].shape)
            # print("gradcam_mask.shape: ", batch['gradcam_mask'].shape)
            # Forward pass through video encoder (no gradients)
            with torch.set_grad_enabled(False):
                with torch.autocast('cuda', enabled=synchformer_cfg.training.use_half_precision):
                    #print("Here is the error video shape -> : ", video.shape)
                    video = video.permute(0, 1, 3, 2, 4, 5) # (B, S, C, Tv, H, W)
                    mask = batch['attention_mask']
                    mask = mask.permute(0, 1, 3, 2, 4, 5)  # Same as video permute
                    hint, _, hint_mask = video_encoder(video, for_loop=False, cont_mask=mask)
                    #print(f"hint.shape: {hint.shape}, hint_mask.shape: {hint_mask.shape}")
                    
                    hint_mask_flip = hint_mask[..., 0, 0]
                    hint_mask_reduced = ~hint_mask_flip  # True -> False, False -> True

                    #print("hint_mask_reduced.shape: ", hint_mask_reduced.shape)
                    # hint_mask를 텍스트 파일로 저장
                    # T 차원의 모든 값이 동일한지 확인
                    #print("hint_mask_reduced: ", hint_mask_reduced)
                    B, S, tv, D = hint.shape
                    hint = hint.view(B, S*tv, D)
                    
                    # Hint와 Hint Mask 초기 크기
                    B, S, T = hint_mask_reduced.shape  # [B, S, T]
                    num_heads = 8
                    seq_len = S * T  # Flatten된 seq_len

                    # Hint Mask를 (B, seq_len) 형식으로 변경
                    attn_mask_flat = hint_mask_reduced.reshape(B, seq_len)

                    # Hint Mask를 2D에서 3D로 확장
                    # (B, seq_len) -> (B, seq_len, seq_len)
                    attn_mask_3d = attn_mask_flat.unsqueeze(1).expand(B, seq_len, seq_len)

                    # num_heads를 추가
                    # (B, seq_len, seq_len) -> (num_heads * B, seq_len, seq_len)
                    attn_mask_final = attn_mask_3d.repeat(num_heads, 1, 1)

                    # # Hint Mask의 초기 크기
                    # B, S, T = hint_mask_reduced.shape  # [B, S, T]
                    # num_heads = 8
                    # seq_len = S * T  # Flatten된 seq_len

                    # # (B, S, T) -> (S * T, B)
                    # attn_mask_flat = hint_mask_reduced.permute(1, 2, 0).reshape(seq_len, B)

                    # # (S * T, B) -> (S * T * num_heads, B)
                    # attn_mask_repeated = attn_mask_flat.repeat(num_heads, 1)  # num_heads만큼 반복

                    # # (S * T * num_heads, B) -> (S * T * num_heads, B, B)
                    # attn_mask_final = attn_mask_repeated.unsqueeze(-1).expand(-1, B, B)

                    # print("attn_mask_final.shape: ", attn_mask_final.shape)




            # Forward pass through phi
            pred_cam = phi(hint.float(), attn_mask=attn_mask_final).squeeze(2)
            # Permute pred_cam to swap 1st and 2nd dimensions
            pred_cam = pred_cam.permute(0, 2, 1)  # Shape: (B, 527, Sx8)
            
            # Initialize total loss
            total_loss = 0.0

            #print("###############################################################################################################################")
            # Loop over the batch
            skip_num = 0
            for i in range(pred_cam.shape[0]):  # Loop over batch size
                # GradCAM 유효한 값 추출
                gradcam_valid = gradcam[i]  # Shape: (527, max_time)
                gradcam_mask = batch["gradcam_mask"][i]  # Shape: (max_time,)
                gradcam_valid = gradcam_valid[:, gradcam_mask]  # 유효한 time dimension만 남김 -> (527, valid_time)
                #print("gradcam_valid.shape -> ", gradcam_valid.shape)
                #print("gradcam_mask: ", gradcam_mask)

                # Pred_CAM 유효한 값 추출
                pred_cam_mask = hint_mask_flip[i]  # Shape: (segment, T/2), padding 부분만 True
                valid_pred_cam_mask = pred_cam_mask.reshape(-1)  # Flatten -> (segment * T/2,)
                valid_pred_cam = pred_cam[i, :, valid_pred_cam_mask]  # 유효한 pred_cam만 추출 -> Shape: (527, valid_length)
                #print("valid_pred_cam.shape -> ", valid_pred_cam.shape)
                #print("valid_pred_cam.shape -> ", valid_pred_cam_mask)

                # Pred_CAM을 GradCAM 크기에 맞게 Interpolate
                valid_pred_cam = valid_pred_cam.unsqueeze(0)  # Add batch dim -> Shape: (1, 527, valid_length)
                
                if valid_pred_cam.shape[-1] > 1 and gradcam_valid.shape[-1] > 1:  # valid_length > 1인 경우만 interpolate
                    interpolated_pred_cam = F.interpolate(
                        valid_pred_cam,
                        size=gradcam_valid.shape[1],  # gradcam_valid의 time dimension에 맞춤
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)  # Remove batch dim -> Shape: (527, valid_time)

                    # MSE Loss 계산
                    loss = F.mse_loss(interpolated_pred_cam, gradcam_valid, reduction="mean")
                else:
                    # valid_pred_cam.shape[-1] == 1인 경우, GradCAM에 맞출 수 없으므로 Loss를 계산하지 않음
                    print(f"Skipping interpolation for valid_pred_cam with shape {valid_pred_cam.shape}")
                    loss = 0.0  # 또는 다른 처리를 적용 가능
                    skip_num += 1
                total_loss += loss
                # interpolated_pred_cam = F.interpolate(
                #     valid_pred_cam,
                #     size=gradcam_valid.shape[1],  # gradcam_valid의 time dimension에 맞춤
                #     mode="linear",
                #     align_corners=False,
                # ).squeeze(0)  # Remove batch dim -> Shape: (527, valid_time)

                # # MSE Loss 계산
                # loss = F.mse_loss(interpolated_pred_cam, gradcam_valid, reduction="mean")
                # total_loss += loss
            #print("###############################################################################################################################")


            # 배치 평균 Loss 계산
            batch_loss = total_loss / (pred_cam.shape[0] - skip_num)

            # Backward pass
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print("batch_loss: ", batch_loss.item())
            
            
            if step % args.checkpoint_step == 0:
                checkpoint_path = os.path.join(save_path, f'checkpoint_epoch{epoch}_step{step}.pt')
                checkpoint_data = {
                    'model_state_dict': phi.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                }
                torch.save(checkpoint_data, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")
            
            
            # Save checkpoint
            if step % args.checkpoint_step == 0:
                checkpoint_path = os.path.join(save_path, f'checkpoint_epoch{epoch}_step{step}.pt')
                torch.save(phi.state_dict(), checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

        # Step the scheduler
        lr_scheduler.step()
        print(f"Epoch {epoch + 1}/{args.num_epochs} completed. Loss: {epoch_loss / len(train_dataloader):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #####################################################################
    # Dataset and Paths
    #####################################################################
    parser.add_argument("--csv_file", type=str, default="../valid_video_files.csv", help="Path to the CSV file containing video file names.")
    parser.add_argument("--video_dir", type=str, default="../video", help="Path to the directory containing video files.")
    parser.add_argument("--cam_dir", type=str, default="../CAM", help="Path to the directory containing GradCAM files.")
    parser.add_argument("--save_path", type=str, default="./results_resume", help="Path to save model outputs and checkpoints.")

    #####################################################################
    # Model and Experiment
    #####################################################################
    parser.add_argument("--control_type", type=str, default="energy_video", choices=["energy_audio", "energy_video"], help="Control type.")
    parser.add_argument("--synchformer_exp", type=str, default="24-01-04T16-39-21", help="Synchformer experiment name.")
    parser.add_argument("--phi_ckpt_path", type=str, default="./ckpts/phi_vggsound.ckpt", help="Path to the pretrained phi checkpoint.")

    #####################################################################
    # Training Parameters
    #####################################################################
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for preprocessing.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of video clips in seconds.")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale.")
    parser.add_argument("--ddim_steps", type=int, default=200, help="DDIM sampling steps.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--lr_step", type=int, default=10, help="Step size for learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for learning rate scheduler.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--checkpoint_step", type=int, default=10000, help="Steps between saving checkpoints.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--re_encode", type=bool, default=False, help="Random seed.")


    args = parser.parse_args()

    main(args)
