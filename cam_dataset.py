import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from audioldm.pipeline import rewas_generation, build_control_model

class VideoAudioDataset(Dataset):
    def __init__(self, csv_file, video_dir, cam_dir, control_type, synchformer_exp, synchformer_cfg, video_encoder, phi, 
                 seed=42, duration=5, guidance_scale=3, ddim_steps=200, batchsize=1, re_encode=True, save_path="./results", mode="train"):
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.cam_dir = cam_dir
        self.control_type = control_type
        self.synchformer_cfg = synchformer_cfg
        self.seed = seed
        self.duration = duration
        self.batchsize = batchsize
        self.re_encode = re_encode
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            video_file = row['FileName']

            videopath = os.path.join(self.video_dir, video_file)
            gradcam_file_name = os.path.splitext(video_file)[0] + ".pt"
            # gradcam_results1 = os.path.join(self.cam_dir, "GradCAM_Results", gradcam_file_name)
            # gradcam_results2 = os.path.join(self.cam_dir, "GradCAM_Results2", gradcam_file_name)
            #gradcam_results = os.path.join("cam", gradcam_file_name)
            #print(os.path.join(self.cam_dir, gradcam_file_name))
            gradcam_results = os.path.join(self.cam_dir, gradcam_file_name)

            if os.path.exists(gradcam_results):
                gradcam_file = gradcam_results
            else:
                raise FileNotFoundError(
                    f"GradCAM file not found for {video_file}.\n"
                )

            text = ""
            batch = rewas_generation(
                text=text,
                videos=videopath,
                control_type=self.control_type,
                synchformer_cfg=self.synchformer_cfg,
                original_audio_file_path=None,
                seed=self.seed,
                duration=self.duration,
                batchsize=self.batchsize,
                re_encode=self.re_encode,
                local_rank=0,
                mode=self.mode
            )

            video = batch["model_inputs"].squeeze(0)
            start_sec = batch["start_sec"]
            end_sec = batch["end_sec"]
            
            start_sec = batch["start_sec"]
            end_sec = batch["end_sec"]

            gradcam = torch.load(gradcam_file)
            start_sec_grad = int(start_sec * 100)
            end_sec_grad = int(end_sec * 100)
            gradcam = gradcam[:, start_sec_grad:end_sec_grad+1]
            sigma = 0.00046133497380651534
            # global_max = 0.19706682860851288
            # global_min = 0.0
            # gradcam = (gradcam - global_min) / (global_max - global_min)
            gradcam = gradcam / sigma
            
            #print(f"video.shape: {video.shape}, gradcam: {gradcam.shape}, attention_mask: {batch['attention_mask'].shape}, start_sec: {start_sec}, end_sec: {end_sec}")
            #print(f"video -> {video.shape}, cam -> {gradcam.shape}")
            return {
                "video": video,
                "gradcam": gradcam,
                "duration": self.duration,
                "attention_mask": batch['attention_mask'],
                "v_ranges": batch['v_ranges']
            }
        except Exception as e:
            # 오류 발생 시 기본값 반환
            print(f"[Warning] Error processing index {idx}: {e}")
            return {"video": torch.zeros(1), "start_sec": torch.tensor(-1.0), "end_sec": torch.tensor(-1.0), "gradcam": torch.zeros(1), "duration": -1}



class VideoAudioTestDataset(Dataset):
    def __init__(self, csv_file, video_dir, cam_dir, control_type, synchformer_exp, synchformer_cfg,
                 video_encoder, phi, seed=42, duration=5, guidance_scale=1.0, ddim_steps=50, batchsize=1, 
                 re_encode=True, save_path=None, mode="test"):
        """
        Args:
            csv_file (str): Path to the CSV file containing data information.
            video_dir (str): Path to the video directory.
            cam_dir (str): Path to the base GradCAM directory.
            control_type (str): Control type for generation.
            synchformer_exp (str): SynchFormer experiment identifier.
            synchformer_cfg (dict): Configuration for the SynchFormer model.
            video_encoder (object): Video encoder model.
            phi (object): Model to generate predictions.
            seed (int): Random seed.
            duration (int): Duration for processing.
            guidance_scale (float): Guidance scale for generation.
            ddim_steps (int): Number of DDIM steps.
            batchsize (int): Batch size for processing.
            re_encode (bool): Whether to re-encode video.
            save_path (str): Path to save generated results.
            mode (str): Mode of the dataset ("train", "test", etc.).
        """
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.cam_dir = cam_dir
        self.control_type = control_type
        self.synchformer_exp = synchformer_exp
        self.synchformer_cfg = synchformer_cfg
        self.video_encoder = video_encoder
        self.phi = phi
        self.seed = seed
        self.duration = duration
        self.guidance_scale = guidance_scale
        self.ddim_steps = ddim_steps
        self.batchsize = batchsize
        self.re_encode = re_encode
        self.save_path = save_path
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            video_file = row['FileName'] + '.mp4'

            videopath = os.path.join(self.video_dir, video_file)
            gradcam_file_name = os.path.splitext(video_file)[0] + ".pt"

            # AVsync GradCAM 파일 경로 생성
            avsync_cam_paths = {
                "0s": os.path.join(self.cam_dir, "AVsync_audios_0s_cam", gradcam_file_name),
                "1s": os.path.join(self.cam_dir, "AVsync_audios_1s_cam", gradcam_file_name),
                "2s": os.path.join(self.cam_dir, "AVsync_audios_2s_cam", gradcam_file_name)
            }

                
            # AVsync GradCAM 파일 확인 및 로드
            avsync_cams = {}
            for key, path in avsync_cam_paths.items():
                if os.path.exists(path):
                    avsync_cams[key] = torch.load(path)
                else:
                    raise FileNotFoundError(f"AVsync GradCAM file not found for {video_file} in {key} directory.")

            # Generate batch
            batch = rewas_generation(
                text="",
                videos=videopath,
                control_type=self.control_type,
                synchformer_cfg=self.synchformer_cfg,
                original_audio_file_path=None,
                seed=self.seed,
                duration=self.duration,
                batchsize=self.batchsize,
                re_encode=self.re_encode,
                local_rank=0,
                mode=self.mode
            )

            video = batch["model_inputs"].squeeze(0)
            
            start_sec = batch["start_sec"]
            end_sec = batch["end_sec"]
            
            start_sec_grad = int(start_sec * 100)
            end_sec_grad = int(end_sec * 100)

            avsync_cams["0s"] = avsync_cams["0s"][:, start_sec_grad:end_sec_grad+1]
            avsync_cams["1s"] = avsync_cams["1s"][:, start_sec_grad:end_sec_grad+1]
            avsync_cams["2s"] = avsync_cams["2s"][:, start_sec_grad:end_sec_grad+1]
            
            # global_max_0s = 0.3951365351676941
            # global_max_1s = 0.3577139973640442
            # global_max_2s = 0.30765703320503235
            # global_min = 0.0
            # avsync_cams["0s"] = (avsync_cams["0s"] - global_min) / (global_max_0s - global_min)
            # avsync_cams["1s"] = (avsync_cams["1s"] - global_min) / (global_max_1s - global_min)
            # avsync_cams["2s"] = (avsync_cams["2s"] - global_min) / (global_max_2s - global_min)
            
            sigma_0s = 0.001805894891731441
            sigma_1s = 0.0017171804793179035
            sigma_2s = 0.0015461144503206015
            avsync_cams["0s"] = avsync_cams["0s"] / sigma_0s
            avsync_cams["1s"] = avsync_cams["1s"] / sigma_1s
            avsync_cams["2s"] = avsync_cams["2s"] / sigma_2s

            
            return {
                "video": video,
                "gradcam_0s": avsync_cams["0s"],
                "gradcam_1s": avsync_cams["1s"],
                "gradcam_2s": avsync_cams["2s"],
                "duration": self.duration,
                "attention_mask": batch['attention_mask'],
                "v_ranges": batch['v_ranges'],
            }
        except Exception as e:
            print(f"[Warning] Error processing index {idx}: {e}")
            return {
                "video": torch.zeros(1),
                "gradcam_0s": torch.zeros(1),
                "gradcam_1s": torch.zeros(1),
                "gradcam_2s": torch.zeros(1),
                "duration": -1,
                "attention_mask": torch.zeros(1),
                "v_ranges": torch.zeros(1), 
            }

