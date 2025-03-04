# reference: https://github.com/haoheliu/AudioLDM/blob/main/audioldm/utils.py

import subprocess
import json
import os
import soundfile as sf


import torch
import torchvision
import torchaudio

def default_audioldm_config(model_name="audioldm-s-full"):    
    basic_config = {
        "wave_file_save_path": "./output",
        "id": {
            "version": "v1",
            "name": "default",
            "root": "/mnt/fast/nobackup/users/hl01486/projects/general_audio_generation/AudioLDM-python/config/default/latent_diffusion.yaml",
        },
        "preprocessing": {
            "audio": {"sampling_rate": 16000, "max_wav_value": 32768},
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 8000,
                "freqm": 0,
                "timem": 0,
                "blur": False,
                "mean": -4.63,
                "std": 2.74,
                "target_length": 1024,
            },
        },
        "model": {
            "device": "cuda",
            "target": "audioldm.pipline.LatentDiffusion",
            "params": {
                "base_learning_rate": 5e-06,
                "linear_start": 0.0015,
                "linear_end": 0.0195,
                "num_timesteps_cond": 1,
                "log_every_t": 200,
                "timesteps": 1000,
                "first_stage_key": "fbank",
                "cond_stage_key": "waveform",
                "latent_t_size": 256,
                "latent_f_size": 16,
                "channels": 8,
                "cond_stage_trainable": True,
                "conditioning_key": "film",
                "monitor": "val/loss_simple_ema",
                "scale_by_std": True,
                "unet_config": {
                    "target": "audioldm.latent_diffusion.openaimodel.UNetModel",
                    "params": {
                        "image_size": 64,
                        "extra_film_condition_dim": 512,
                        "extra_film_use_concat": True,
                        "in_channels": 8,
                        "out_channels": 8,
                        "model_channels": 128,
                        "attention_resolutions": [8, 4, 2],
                        "num_res_blocks": 2,
                        "channel_mult": [1, 2, 3, 5],
                        "num_head_channels": 32,
                        "use_spatial_transformer": True,
                    },
                },
                "first_stage_config": {
                    "base_learning_rate": 4.5e-05,
                    "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                    "params": {
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 8,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 8,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
                "cond_stage_config": {
                    "target": "audioldm.clap.encoders.CLAPAudioEmbeddingClassifierFreev2",
                    "params": {
                        "key": "waveform",
                        "sampling_rate": 16000,
                        "embed_mode": "audio",
                        "unconditional_prob": 0.1,
                    },
                },
            },
        },
    }
    
    if("-l-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 256
        basic_config["model"]["params"]["unet_config"]["params"]["num_head_channels"] = 64
    elif("-m-" in model_name):
        basic_config["model"]["params"]["unet_config"]["params"]["model_channels"] = 192
        basic_config["model"]["params"]["cond_stage_config"]["params"]["amodel"] = "HTSAT-base" # This model use a larger HTAST
        
    return basic_config
    

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
        return data


def read_json(dataset_json_file):
    with open(dataset_json_file, "r") as fp:
        data_json = json.load(fp)
    return data_json["data"]


def seed_everything(seed):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def custom_collate_fn_test(batch):
    filtered_batch = []  # 조건에 맞는 샘플을 저장할 리스트
    skipped_indices = []  # 제외된 샘플의 인덱스 저장
    file_names=[]

    # 유효한 샘플 필터링
    for idx, sample in enumerate(batch):
        try:
            # "duration"이 0보다 큰 경우만 유효한 샘플로 간주
            if "video" in sample and sample["duration"] > 0:
                filtered_batch.append(sample)
                file_names.append(sample["file_name"])  # FileName 저장

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
    max_time = max(sample["gradcam_0s"].shape[1] for sample in filtered_batch)

    # 비디오, gradcam, attention_mask, gradcam_mask 생성
    videos = []
    gradcam_0s = []
    gradcam_1s = []
    gradcam_2s = []
    attention_masks = []
    gradcam_masks = []  # gradcam_mask는 하나만 생성
    v_ranges = []

    for sample in filtered_batch:
        # 비디오 패딩
        video = sample["video"]
        video_padding = torch.zeros(
            (max_segments - video.shape[0], *video.shape[1:]), dtype=video.dtype, device=video.device
        )
        videos.append(torch.cat([video, video_padding], dim=0))  # 패딩된 비디오 추가

        # gradcam_0s 패딩
        gradcam = sample["gradcam_0s"]
        gradcam_padding = torch.zeros(
            (gradcam.shape[0], max_time - gradcam.shape[1]), dtype=gradcam.dtype, device=gradcam.device
        )
        gradcam_0s.append(torch.cat([gradcam, gradcam_padding], dim=1))

        # gradcam_1s 패딩
        gradcam = sample["gradcam_1s"]
        gradcam_padding = torch.zeros(
            (gradcam.shape[0], max_time - gradcam.shape[1]), dtype=gradcam.dtype, device=gradcam.device
        )
        gradcam_1s.append(torch.cat([gradcam, gradcam_padding], dim=1))

        # gradcam_2s 패딩
        gradcam = sample["gradcam_2s"]
        gradcam_padding = torch.zeros(
            (gradcam.shape[0], max_time - gradcam.shape[1]), dtype=gradcam.dtype, device=gradcam.device
        )
        gradcam_2s.append(torch.cat([gradcam, gradcam_padding], dim=1))

        # attention_mask 패딩
        attention_mask = sample["attention_mask"]
        attention_mask_padding = torch.zeros(
            (max_segments - attention_mask.shape[0], *attention_mask.shape[1:]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_masks.append(torch.cat([attention_mask, attention_mask_padding], dim=0))  # 패딩된 attention_mask 추가

        # gradcam_mask 생성 (하나만 사용)
        gradcam_mask = torch.zeros(max_time, dtype=torch.bool, device=sample["gradcam_0s"].device)
        gradcam_mask[:sample["gradcam_0s"].shape[1]] = 1  # 유효한 시간 부분만 1로 설정
        gradcam_masks.append(gradcam_mask)

        # v_ranges 패딩
        current_v_ranges = sample["v_ranges"]
        v_range_padding = torch.zeros((max_segments - current_v_ranges.shape[0], 2), dtype=torch.long, device=current_v_ranges.device)
        v_ranges.append(torch.cat([current_v_ranges, v_range_padding], dim=0))  # 패딩된 v_ranges 추가


    # 텐서로 변환
    collated_batch["video"] = torch.stack(videos, dim=0)  # (B, max_segments, ...)
    collated_batch["gradcam_0s"] = torch.stack(gradcam_0s, dim=0)  # (B, gradcam_dim, max_time)
    collated_batch["gradcam_1s"] = torch.stack(gradcam_1s, dim=0)  # (B, gradcam_dim, max_time)
    collated_batch["gradcam_2s"] = torch.stack(gradcam_2s, dim=0)  # (B, gradcam_dim, max_time)
    collated_batch["attention_mask"] = torch.stack(attention_masks, dim=0)  # (B, max_segments, segment_size_vframes, C, H, W)
    collated_batch["gradcam_mask"] = torch.stack(gradcam_masks, dim=0)  # (B, max_time)
    collated_batch["v_ranges"] = torch.stack(v_ranges, dim=0)  # (B, max_segments, 2)
    collated_batch["FileName"] = file_names  # (B,) 리스트 형태로 저장

    return collated_batch


def custom_collate_fn_curation(batch):
    filtered_batch = []  # 조건에 맞는 샘플을 저장할 리스트
    skipped_indices = []  # 제외된 샘플의 인덱스 저장
    file_names=[]

    # 유효한 샘플 필터링
    for idx, sample in enumerate(batch):
        try:
            # "duration"이 0보다 큰 경우만 유효한 샘플로 간주
            if "video" in sample and sample["duration"] > 0:
                filtered_batch.append(sample)
                file_names.append(sample["file_name"])  # FileName 저장

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
    max_time = max(sample["gradcam_0s"].shape[1] for sample in filtered_batch)

    # 비디오, gradcam, attention_mask, gradcam_mask 생성
    videos = []
    gradcam_0s = []
    attention_masks = []
    gradcam_masks = []  # gradcam_mask는 하나만 생성
    v_ranges = []

    for sample in filtered_batch:
        # 비디오 패딩
        video = sample["video"]
        video_padding = torch.zeros(
            (max_segments - video.shape[0], *video.shape[1:]), dtype=video.dtype, device=video.device
        )
        videos.append(torch.cat([video, video_padding], dim=0))  # 패딩된 비디오 추가

        # gradcam_0s 패딩
        gradcam = sample["gradcam_0s"]
        gradcam_padding = torch.zeros(
            (gradcam.shape[0], max_time - gradcam.shape[1]), dtype=gradcam.dtype, device=gradcam.device
        )
        gradcam_0s.append(torch.cat([gradcam, gradcam_padding], dim=1))

        # attention_mask 패딩
        attention_mask = sample["attention_mask"]
        attention_mask_padding = torch.zeros(
            (max_segments - attention_mask.shape[0], *attention_mask.shape[1:]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_masks.append(torch.cat([attention_mask, attention_mask_padding], dim=0))  # 패딩된 attention_mask 추가

        # gradcam_mask 생성 (하나만 사용)
        gradcam_mask = torch.zeros(max_time, dtype=torch.bool, device=sample["gradcam_0s"].device)
        gradcam_mask[:sample["gradcam_0s"].shape[1]] = 1  # 유효한 시간 부분만 1로 설정
        gradcam_masks.append(gradcam_mask)

        # v_ranges 패딩
        current_v_ranges = sample["v_ranges"]
        v_range_padding = torch.zeros((max_segments - current_v_ranges.shape[0], 2), dtype=torch.long, device=current_v_ranges.device)
        v_ranges.append(torch.cat([current_v_ranges, v_range_padding], dim=0))  # 패딩된 v_ranges 추가


    # 텐서로 변환
    collated_batch["video"] = torch.stack(videos, dim=0)  # (B, max_segments, ...)
    collated_batch["gradcam_0s"] = torch.stack(gradcam_0s, dim=0)  # (B, gradcam_dim, max_time)
    collated_batch["attention_mask"] = torch.stack(attention_masks, dim=0)  # (B, max_segments, segment_size_vframes, C, H, W)
    collated_batch["gradcam_mask"] = torch.stack(gradcam_masks, dim=0)  # (B, max_time)
    collated_batch["v_ranges"] = torch.stack(v_ranges, dim=0)  # (B, max_segments, 2)
    collated_batch["FileName"] = file_names  # (B,) 리스트 형태로 저장

    return collated_batch



