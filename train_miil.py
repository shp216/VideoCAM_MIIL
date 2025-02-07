import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.encoder_utils import patch_config, get_pretrained
from cam_dataset import VideoAudioDataset, VideoAudioTestDataset  # Assuming the dataset is implemented in this file
from tqdm import tqdm
from omegaconf import OmegaConf
# from utils import seed_everything
from encoder.phi import Phi
import random
import numpy as np
from accelerate import Accelerator
from utils import custom_collate_fn, custom_collate_fn_test, seed_everything
from evaluate_cam import evaluate
import wandb
from load_checkpoint import save_checkpoint, load_checkpoint

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
        save_path=save_path,
        mode="train"
    )
    
    test_dataset = VideoAudioTestDataset(
        csv_file=args.test_csv_file,
        video_dir=args.test_video_dir,
        cam_dir=args.test_cam_dir,
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
        save_path=save_path,
        mode="test"
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Test DataLoader 생성
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,  # Test 배치 크기
        shuffle=False,  # Test 데이터는 순서를 섞지 않음
        num_workers=args.num_workers,  # 동일한 worker 수
        collate_fn=custom_collate_fn_test,  # Train과 동일한 custom_collate_fn 사용
        pin_memory=True
    )
    

    
    accelerator = Accelerator()
    if accelerator.is_main_process:
        wandb.init(project=args.project_name, name=args.run_name, config=args)

    train_loader = accelerator.prepare(train_dataloader)
    test_loader = accelerator.prepare(test_dataloader)
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
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    #####################################################################
    # Training Loop
    #####################################################################
    
    start_epoch, start_step = load_checkpoint(args.checkpoint_path, phi, optimizer, lr_scheduler, accelerator)
    print(f"Start_epoch: {start_epoch}, start_step: {start_step}")

    phi.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()
        if args.resume:
            if epoch < start_epoch:
                continue
        print(f"Starting epoch {epoch + 1}/{args.num_epochs}...")
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", dynamic_ncols=False)
        for step, batch in enumerate(train_loader):
            if args.resume:
                if step <= start_step:
                    progress_bar.update(1)
                    continue
            # Move data to GPU
            video = batch['video']
            gradcam = batch['gradcam']
            
            # def format_nested_list(data, indent=0):
            #     if isinstance(data, list):
            #         formatted_str = "["
            #         if isinstance(data[0], list):  # 중첩 리스트일 경우
            #             formatted_str += "\n"
            #             for item in data:
            #                 formatted_str += " " * (indent + 4) + format_nested_list(item, indent + 4) + ",\n"
            #             formatted_str += " " * indent + "]"
            #         else:  # 값 리스트는 한 줄로
            #             formatted_str += ", ".join(map(str, data)) + "]"
            #     else:
            #         formatted_str = str(data)
            #     return formatted_str

            # Forward pass through video encoder (no gradients)
            with torch.set_grad_enabled(False):
                with torch.autocast('cuda', enabled=synchformer_cfg.training.use_half_precision):
                    #print("Here is the error video shape -> : ", video.shape)
                    video = video.permute(0, 1, 3, 2, 4, 5) # (B, S, C, Tv, H, W)
                    mask = batch['attention_mask']
                    mask = mask.permute(0, 1, 3, 2, 4, 5)  # Same as video permute
                    #print("mask.shape: ", mask.shape)
                    hint, _, hint_mask = video_encoder(video, for_loop=False, cont_mask=mask)
                    #print(f"hint.shape: {hint.shape}, hint_mask.shape: {hint_mask.shape}")
                    
                    hint_mask_flip = hint_mask[..., 0, 0]
                    hint_mask_reduced = ~hint_mask_flip  # True -> False, False -> True

                    B, S, tv, D = hint.shape
                    hint = hint.view(B, S*tv, D)
                    
                    # Hint와 Hint Mask 초기 크기
                    B, S, T = hint_mask_reduced.shape  # [B, S, T]
                    #num_heads = 8
                    seq_len = S*T  # Flatten된 seq_len

                    # Hint Mask를 (B, seq_len) 형식으로 변경
                    attn_mask_flat = hint_mask_reduced.reshape(B, seq_len)

                    # Hint Mask를 2D에서 3D로 확장
                    # (B, seq_len) -> (B, seq_len, seq_len)
                    #attn_mask_3d = attn_mask_flat.unsqueeze(1).expand(B, seq_len, seq_len)

                    # num_heads를 추가
                    # (B, seq_len, seq_len) -> (num_heads * B, seq_len, seq_len)
                    #attn_mask_final = attn_mask_3d.repeat(num_heads, 1, 1)

            # Forward pass through phi
            pred_cam = phi(hint.float(), attn_mask=attn_mask_flat).squeeze(2)
            # Permute pred_cam to swap 1st and 2nd dimensions
            pred_cam = pred_cam.permute(0, 2, 1)  # Shape: (B, 527, Sx8)
                
            # torch.set_printoptions(profile="full")
            # print(pred_cam)
            # torch.set_printoptions(profile="default")
            # Initialize total loss
            total_loss = 0.0

            # Loop over the batch
            for i in range(pred_cam.shape[0]):  # Loop over batch size
                # GradCAM 유효한 값 추출
                gradcam_valid = gradcam[i]  # Shape: (527, max_time)
                gradcam_mask = batch["gradcam_mask"][i]  # Shape: (max_time,)
                gradcam_valid = gradcam_valid[:, gradcam_mask]  # 유효한 time dimension만 남김 -> (527, valid_time)
                #max_value = gradcam_valid.max()
                #print(f"Max value: {max_value.item()}")
                # Pred_CAM 유효한 값 추출
                pred_cam_mask = hint_mask_flip[i]  # Shape: (segment, T/2), padding 부분만 True
                valid_pred_cam_mask = pred_cam_mask.reshape(-1)  # Flatten -> (segment * T/2,)
                valid_pred_cam = pred_cam[i, :, valid_pred_cam_mask]  # 유효한 pred_cam만 추출 -> Shape: (527, valid_length)
                
                #formatted = format_nested_list(valid_pred_cam.cpu().tolist())
                # # 파일에 저장
                # with open(f"valid_pred_cam.txt", "w") as f:
                #     f.write(formatted)

                # formatted = format_nested_list(gradcam_valid.cpu().tolist())
                # # 파일에 저장
                # with open(f"gradcam_valid.txt", "w") as f:
                #     f.write(formatted)
                    
                # Pred_CAM을 GradCAM 크기에 맞게 Interpolate
                valid_pred_cam = valid_pred_cam.unsqueeze(0)  # Add batch dim -> Shape: (1, 527, valid_length)
                interpolated_pred_cam = F.interpolate(
                    valid_pred_cam,
                    size=gradcam_valid.shape[1],  # gradcam_valid의 time dimension에 맞춤
                    mode="linear",
                    align_corners=False,
                ).squeeze(0)  # Remove batch dim -> Shape: (527, valid_time)

                # MSE Loss 계산
                loss = F.mse_loss(interpolated_pred_cam, gradcam_valid, reduction="mean")
                total_loss += loss

                # weights = torch.where(gradcam_valid > 0, 10.0, 1.0)
                # loss = F.mse_loss(interpolated_pred_cam, gradcam_valid, reduction="none")
                # weighted_loss = (loss * weights).mean()
                # total_loss += weighted_loss
                

            # 배치 평균 Loss 계산
            batch_loss = total_loss / pred_cam.shape[0]
            wandb.log({"step": step, "loss": batch_loss})


            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(batch_loss)
            #batch_loss.backward()
            optimizer.step()
            
            if step % args.eval_step == 0:
                avg_loss1, avg_loss2, avg_loss3, avg_iou1, avg_iou2, avg_iou3 = evaluate(phi, video_encoder, test_loader, synchformer_cfg.training.use_half_precision)
                
                if accelerator.is_main_process:
                    wandb.log({
                        "eval/avg_loss1": avg_loss1,
                        "eval/avg_loss2": avg_loss2,
                        "eval/avg_loss3": avg_loss3,
                        "eval/avg_iou1": avg_iou1,
                        "eval/avg_iou2": avg_iou2,
                        "eval/avg_iou3": avg_iou3
                    }, commit=True)
            
            
            if step % args.checkpoint_step == 0:
                checkpoint_path = os.path.join(args.save_path, f"checkpoint_epoch{epoch}_step{step}.pth")
                save_checkpoint(checkpoint_path, phi, optimizer, lr_scheduler, epoch, step, accelerator)
            
            # tqdm 진행률 업데이트
            progress_bar.set_postfix({"Loss": batch_loss.item()})
            progress_bar.update(1)
            

        # Step the scheduler
        lr_scheduler.step()
        torch.cuda.empty_cache()
    progress_bar.close()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #####################################################################
    # Train Dataset and Paths
    #####################################################################
    parser.add_argument("--csv_file", type=str, default="../audiovisual_filenames.csv", help="Path to the CSV file containing video file names.")
    parser.add_argument("--video_dir", type=str, default="../video_25fps", help="Path to the directory containing video files.")
    parser.add_argument("--cam_dir", type=str, default="../audio_cam", help="Path to the directory containing GradCAM files.")
    parser.add_argument("--save_path", type=str, default="./results_resume", help="Path to save model outputs and checkpoints.")
    
    #####################################################################
    # Test Dataset and Paths
    #####################################################################
    parser.add_argument("--test_csv_file", type=str, default="./test_dataset_real/test_videos_filenames.csv", help="Path to the CSV file containing video file names.")
    parser.add_argument("--test_video_dir", type=str, default="./test_dataset_real/AVsync_videos_25fps", help="Path to the directory containing video files.")
    parser.add_argument("--test_cam_dir", type=str, default="./test_dataset_real", help="Path to the directory containing GradCAM files.")

    #####################################################################
    # Model and Experiment
    #####################################################################
    parser.add_argument("--control_type", type=str, default="energy_video", choices=["energy_audio", "energy_video"], help="Control type.")
    parser.add_argument("--synchformer_exp", type=str, default="24-01-04T16-39-21", help="Synchformer experiment name.")
    parser.add_argument("--phi_ckpt_path", type=str, default="./ckpts/phi_vggsound.ckpt", help="Path to the pretrained phi checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="None")

    #####################################################################
    # Training Parameters
    #####################################################################
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for preprocessing.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of DataLoader workers.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of video clips in seconds.")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale.")
    parser.add_argument("--ddim_steps", type=int, default=200, help="DDIM sampling steps.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--lr_step", type=int, default=10, help="Step size for learning rate scheduler.")
    parser.add_argument("--lr_gamma", type=float, default=0.1, help="Gamma for learning rate scheduler.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--checkpoint_step", type=int, default=20000, help="Steps between saving checkpoints.")
    parser.add_argument("--eval_epoch", type=int, default=1, help="epochs between evaluating.")
    parser.add_argument("--eval_step", type=int, default=2500, help="epochs between evaluating.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--re_encode", type=bool, default=False, help="Random seed.")

    #####################################################################
    # wandb
    #####################################################################
    parser.add_argument("--project_name", type=str, default="CAM_Evaluation")
    parser.add_argument("--run_name", type=str, default="Main_test1")

    args = parser.parse_args()

    main(args)
