import os
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch.utils.data as data
import cv2
import torch.nn.functional as F

from audioCAM.pytorch.models import *
from audioCAM.pytorch.models import ResNet38, Wavegram_Logmel_Cnn14
from audioCAM.pytorch.pytorch_utils import move_data_to_device
import config
import argparse



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from encoder.encoder_utils import patch_config, get_pretrained
from cam_dataset import VideoAudioDataset, VideoAudioTestDataset, CurationTestDataset  # Assuming the dataset is implemented in this file
from tqdm import tqdm
from omegaconf import OmegaConf
# from utils import seed_everything
from encoder.phi import Phi
import random
from accelerate import Accelerator
from utils import custom_collate_fn, custom_collate_fn_test, seed_everything, custom_collate_fn_curation
from evaluate_cam import evaluate_just1
import wandb
from load_checkpoint import save_checkpoint, load_checkpoint
import config
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser(description='Audio Tagging and CAM Generation')
    
    ####################################### audiocam args ###############################################################
    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, default="ResNet38_CAM")
    parser.add_argument('--resnet_checkpoint_path', type=str, default="./audioCAM/ResNet38_mAP=0.434.pth")
    parser.add_argument('--audio_folder_path', type=str, default="./MMG/audio")
    parser.add_argument('--cam_save_dir', type=str, default="./MMG/audiocam")
    parser.add_argument('--cam_image_save_dir', type=str, default="./MMG/audiocam_images")
    parser.add_argument('--cam_batch_size', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)
    



    ####################################### camscore args ###############################################################
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
    parser.add_argument("--test_csv_file", type=str, default="./MMG/Video_filenames.csv", help="Path to the CSV file containing video file names.")
    parser.add_argument("--test_video_dir", type=str, default="./MMG/original_video", help="Path to the directory containing video files.")    
    parser.add_argument("--test_video_25fps_dir", type=str, default="./MMG/original_video_25fps", help="Path to the directory containing video files converted to 25fps.")
    parser.add_argument("--test_cam_dir", type=str, default="./MMG/audiocam", help="Path to the directory containing GradCAM files.")
    parser.add_argument('--max_workers', type=int, default=8)

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

    return parser.parse_args()






# Dataset Loader
def load_audio_files(audio_folder_path, sample_rate):
    """Loads audio files and returns waveforms."""
    audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
    waveforms = []
    for audio_file in audio_files:
        audio_path = os.path.join(audio_folder_path, audio_file)
        waveform, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        waveforms.append(waveform)
    return waveforms, audio_files

class AudioDataset(data.Dataset):
    def __init__(self, waveforms, device):
        self.waveforms = waveforms
        self.device = device
    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        waveform = move_data_to_device(waveform, self.device)
        waveform = torch.tensor(waveform, dtype=torch.float32) # (1, audio_length)
        return waveform

class ResNet38_GradCAM(ResNet38):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, device):
        super(ResNet38_GradCAM, self).__init__(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.device = device
        self.feature_maps = None  # Batch-wise feature maps
        self.gradients = None    # Batch-wise gradients

    def save_gradients(self, grad):
        """Hook to save gradients."""
        self.gradients = grad  # Save gradients

    def forward(self, input, mixup_lambda=None):
        """Forward pass with feature map and gradient hooks."""
        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)           # (batch_size, 1, time_steps, mel_bins)
        print("AFter logmel shape: ", x.shape)
        x = x.transpose(1, 3)
        x = self.bn0(x)  # BatchNorm
        x = x.transpose(1, 3)
        print("start x shape: ", x.shape)
        if self.training:
            x = self.spec_augmenter(x)
        
        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        print("AFter conv block1: ", x.shape)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        print("AFter resnet: ", x.shape)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2))
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        print("AFter conv_blcok_after1: ", x.shape)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)

        # Register hook to save gradients
        x.requires_grad_(True)
        x.register_hook(self.save_gradients)
        self.feature_maps = x
        print("feature map shape: ", x.shape)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu_(self.fc1(x))
        embedding = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        output = self.fc_audioset(x)
        print("output: ", output.shape)
        clipwise_output = torch.sigmoid(output)
        return {'clipwise_output': clipwise_output, 'output': output, 'embedding': embedding}

    def generate_Gradcam_All(self, output):
        """Generate Grad-CAM for all classes and batches."""
        self.zero_grad()  # Initialize gradients
        batch_size, num_classes = output.shape

        gradients_list = []  # To store gradients for all classes

        # Compute gradients for each class
        for class_idx in range(num_classes):
            # Select class scores independently for each class
            class_scores = output[:, class_idx]  # Shape: (batch_size,)

            # Compute gradients for the selected class
            gradients = torch.autograd.grad(
                outputs=class_scores,
                inputs=self.feature_maps,
                grad_outputs=torch.ones_like(class_scores, device=output.device),
                retain_graph=True,
                create_graph=False
            )[0]  # Shape: (batch_size, channels, height, width)

            gradients_list.append(gradients.unsqueeze(1))  # Shape: (batch_size, 1, channels, height, width)

        # Stack gradients across classes
        gradients = torch.cat(gradients_list, dim=1)  # Shape: (batch_size, num_classes, channels, height, width)

        # Global Average Pooling for weights
        weights = gradients.mean(dim=(3, 4))  # Shape: (batch_size, num_classes, channels)

        # Compute weighted sum of feature maps for CAMs
        cams = torch.einsum(
            "bnchw,bnc->bnhw",
            self.feature_maps.unsqueeze(1).expand(-1, num_classes, -1, -1, -1),  # Expand feature maps to include class dimension
            weights  # Weights from gradients
        )  # Shape: (batch_size, num_classes, height, width)

        # Apply ReLU and normalize CAMs

        cams = torch.relu(cams)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("cams shape -> ", cams.shape)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # cams_min = cams.view(batch_size, num_classes, -1).min(dim=2, keepdim=True)[0].view(batch_size, num_classes, 1, 1)
        # cams_max = cams.view(batch_size, num_classes, -1).max(dim=2, keepdim=True)[0].view(batch_size, num_classes, 1, 1)
        # cams = (cams - cams_min) / (cams_max - cams_min + 1e-10)
        
        # cams = (cams - cams_min) / (cams_max - cams_min + 1e-10)
        
        # cams = (cams * 255).to(torch.uint8)  # Normalize to [0, 255] and convert to uint8

        return cams  # Shape: (batch_size, num_classes, height, width)





def save_all_class_time_cam_image(aggregated_cams_array, class_names, save_dir, audio_file):
    """Save CAMs as an image with classes on y-axis and time on x-axis."""
    audio_name = audio_file.split(".")[0]
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


def make_csv():
    # MP4 파일이 있는 폴더 경로
    #video_folder = "./avsync_eval_dataset/shifted_10s/video"
    video_folder = args.test_video_dir


    # CSV 저장 경로
    #output_csv = "./avsync_eval_dataset/shifted_10s/Video_filenames.csv"
    output_csv = args.test_csv_file

    # 폴더 내 모든 MP4 파일 검색
    mp4_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    # 확장자 제거 (파일명만 추출)
    file_names = [os.path.splitext(f)[0] for f in mp4_files]

    # DataFrame 생성
    df = pd.DataFrame({"FileName": file_names})

    # CSV 파일로 저장
    df.to_csv(output_csv, index=False)

    print(f"CSV 파일 저장 완료: {output_csv}")

def convert_video_to_25fps(input_path, output_path):
    """
    Convert a video file to 25fps and save it to the specified output path.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted video file.
    """
    try:
        cmd = f"ffmpeg -y -i {input_path} -vf fps=25 {output_path} -loglevel error"
        os.system(cmd)
    except Exception as e:
        print(f"[Error] Failed to convert {input_path} to 25fps: {e}")

def process_videos(input_folder, output_folder, max_workers=8):
    """
    Convert all videos in a folder to 25fps and save them to another folder.

    Args:
        input_folder (str): Path to the folder containing input video files.
        output_folder (str): Path to save the converted video files.
        max_workers (int): Maximum number of parallel processes.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for video_file in video_files:
            input_path = os.path.join(input_folder, video_file)
            output_path = os.path.join(output_folder, video_file)
            executor.submit(safe_convert_video, input_path, output_path)

def safe_convert_video(input_path, output_path):
    """
    Safely convert a video to 25fps, skipping on failure.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the converted video file.
    """
    try:
        convert_video_to_25fps(input_path, output_path)
    except Exception as e:
        print(f"[Error] Skipping {input_path} due to error: {e}")

def audio_tagging_GradCAM_All(args):
    """Perform audio tagging and generate Grad-CAMs."""
    
    sample_rate = args.sample_rate
    audio_folder_path = args.audio_folder_path
    batch_size = args.cam_batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    cam_save_dir = args.cam_save_dir
    classes_num = config.classes_num
    labels = config.labels
    image_save_dir = args.cam_image_save_dir
    os.makedirs(args.cam_save_dir, exist_ok=True)
    # Initialize model
    Model = ResNet38_GradCAM
    model = Model(
        sample_rate=sample_rate,
        window_size=args.window_size,
        hop_size=args.hop_size,
        mel_bins=args.mel_bins,
        fmin=args.fmin,
        fmax=args.fmax,
        classes_num=classes_num,
        device=device
    )

    # Load pretrained model
    checkpoint = torch.load(args.resnet_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Load audio files and create dataloader
    waveforms, audio_files = load_audio_files(audio_folder_path, sample_rate)
    dataset = AudioDataset(waveforms, device)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    global_min = float('inf')
    global_max = float('-inf')

    for batch_idx, waveforms in enumerate(dataloader):
        print("waveforms.shape: ", waveforms.shape)

        print("############################################################################")
        print(waveforms.shape)
        print(type(waveforms))
        print(waveforms.dtype)
        print("############################################################################")
        # Forward pass
        with torch.set_grad_enabled(True):
            output = model(waveforms)

        clipwise_output = output['clipwise_output']
        first_output = output['output']
        #clipwise_output = output['clipwise_output'].detach().cpu().numpy()
        cams = model.generate_Gradcam_All(output['output'])  # Shape: (batch_size, num_classes, height, width)
        print("First cams.shape: ", cams.shape)
        # Spectrogram extraction for resizing reference
        spectrograms = model.logmel_extractor(model.spectrogram_extractor(waveforms))
        spectrogram_shape = spectrograms.shape[-2:]  # (height, width)
        print(f"Spectrogram shape: {spectrogram_shape}")

        
        # Mel Spectrogram 크기에 맞게 Interpolation 수행
        cams = cams.float()
        print("cams before resized.shape: ", cams.shape)

        # Interpolate to match Mel Spectrogram size
        cams_resized = F.interpolate(
            cams,
            size=(spectrogram_shape[0], spectrogram_shape[1]),  # Target size (321, 64)
            mode='bilinear',
            align_corners=False
        )  # Shape: (batch_size, num_classes, 321, 64)
        print("cams_resized.shape: ", cams_resized.shape)
        
        #scaled_cams = cams_resized
        scaled_cams = cams_resized
        scaled_cams = cams_resized * clipwise_output[:, :, None, None]  # Shape: (batch_size, num_classes, 321, 64)
        aggregated_cams = scaled_cams.mean(dim=3)  # Shape: (batch_size, num_classes, 321)
        
        # Normalize aggregated_cams to range [0, 1] for each data sample independently
        aggregated_cams_min = aggregated_cams.view(aggregated_cams.shape[0], -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        aggregated_cams_max = aggregated_cams.view(aggregated_cams.shape[0], -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        
        # Update global min and max
        global_min = min(global_min, aggregated_cams.min().item())
        global_max = max(global_max, aggregated_cams.max().item())

        #normalized_aggregated_cams = (aggregated_cams - aggregated_cams_min) / (aggregated_cams_max - aggregated_cams_min + 1e-10)
        normalized_aggregated_cams = aggregated_cams
        #normalized_aggregated_cams = (aggregated_cams - aggregated_cams_min) / (aggregated_cams_max - aggregated_cams_min + 1e-10)
        
        # Save normalized aggregated CAMs for each audio file
        for i in range(normalized_aggregated_cams.shape[0]):
            audio_file = audio_files[batch_idx * batch_size + i]

            # Generate file name for saving
            audio_name = os.path.splitext(os.path.basename(audio_file))[0]
            save_path = os.path.join(cam_save_dir, f"{audio_name}.pt")

            # Save CAM as a .pt file
            torch.save(normalized_aggregated_cams[i].detach().cpu(), save_path)

            print(f"Saved CAM for {audio_file} to {save_path}")
            
            save_all_class_time_cam_image(normalized_aggregated_cams[i].detach().cpu().numpy(), labels, image_save_dir, audio_file)
            
            print(f"global_min: {global_min}, global_max: {global_max}")
        
        torch.cuda.empty_cache()

        
def calculate_cam_score(args):
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
        video_dir=args.test_video_25fps_dir,
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
    args = get_args()
    make_csv()
    process_videos(args.test_video_dir, args.test_video_25fps_dir, args.max_workers)
    audio_tagging_GradCAM_All(args)
    calculate_cam_score(args)