import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from encoder.encoder_utils import patch_config, get_pretrained
from cam_dataset import VideoAudioDataset, VideoAudioTestDataset, CurationTestDataset  # Assuming the dataset is implemented in this file
from tqdm import tqdm
from omegaconf import OmegaConf
# from utils import seed_everything
from encoder.phi import Phi
import random
import numpy as np
from accelerate import Accelerator
from utils import custom_collate_fn, custom_collate_fn_test, seed_everything, custom_collate_fn_curation
from evaluate_cam import evaluate_just1
import wandb
from load_checkpoint import save_checkpoint, load_checkpoint
import config


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


    #####################################################################
    # Dataset and DataLoader
    #####################################################################
    test_dataset = CurationTestDataset(
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
    
    # Test DataLoader 생성
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,  # Test 배치 크기
        shuffle=False,  # Test 데이터는 순서를 섞지 않음
        num_workers=args.num_workers,  # 동일한 worker 수
        collate_fn=custom_collate_fn_curation,  # Train과 동일한 custom_collate_fn 사용
        pin_memory=True
    )
    
    accelerator = Accelerator()
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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phi = phi.to(device)  # 모델을 GPU로 이동
    start_epoch, start_step = load_checkpoint(args.checkpoint_path, phi, optimizer, lr_scheduler, accelerator)
    evaluate_just1(phi, video_encoder, test_loader, synchformer_cfg.training.use_half_precision)
    

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
    parser.add_argument("--test_csv_file", type=str, default="./MMG_test/Curated_Vggsound_Video_filenames.csv", help="Path to the CSV file containing video file names.")
    parser.add_argument("--test_video_dir", type=str, default="./MMG_test/Curated_Vggsound_Video_25fps", help="Path to the directory containing video files.")
    parser.add_argument("--test_cam_dir", type=str, default="./MMG_test/Curated_Vggsound_CAM", help="Path to the directory containing GradCAM files.")

    # parser.add_argument("--test_csv_file", type=str, default="./avsync_eval_dataset/GT/Video_filenames.csv", help="Path to the CSV file containing video file names.")
    # parser.add_argument("--test_video_dir", type=str, default="./avsync_eval_dataset/GT/video_25fps", help="Path to the directory containing video files.")
    # parser.add_argument("--test_cam_dir", type=str, default="./avsync_eval_dataset/GT/audiocam", help="Path to the directory containing GradCAM files.")

    #####################################################################
    # Model and Experiment
    #####################################################################
    parser.add_argument("--control_type", type=str, default="energy_video", choices=["energy_audio", "energy_video"], help="Control type.")
    parser.add_argument("--synchformer_exp", type=str, default="24-01-04T16-39-21", help="Synchformer experiment name.")
    parser.add_argument("--phi_ckpt_path", type=str, default="./ckpts/phi_vggsound.ckpt", help="Path to the pretrained phi checkpoint.")
    parser.add_argument("--resume", action="store_true", help="Resume training from a checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default="./results_resume/checkpoint_epoch5_step0.pth")

    #####################################################################
    # Training Parameters
    #####################################################################
    parser.add_argument("--batchsize", type=int, default=1, help="Batch size for preprocessing.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--test_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
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
