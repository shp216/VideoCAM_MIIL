# ReWaS
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)

import yaml
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from scipy.signal import savgol_filter

from utils import seed_everything, default_audioldm_config
from encoder.encoder_utils import get_transforms, get_pretrained
from audioldm.rewas import ReWaS
from audioldm.utilities.audio import read_wav_file
from audioldm.utilities.model_util import set_cond_audio, set_cond_text
from audioldm.utilities.data.utils import re_encode_video, get_video_and_audio, process_video_and_audio, process_video_with_frame


def make_batch_for_control_to_audio(
    text, videos, control_type, synchformer_cfg,  
    waveform=None, fbank=None, batchsize=1, local_rank = None, re_encode=True, duration=None, mode="train"):
    
    if not type(text)==list:
        text = [text] 
        video = [videos] 

    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
        
    model_inputs = []
    fbank_list = []

    if waveform is None:
        waveform = torch.zeros((batchsize, 160000)) 
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    if fbank is None:
        fbank = torch.zeros((batchsize, 1024, 64)) 
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize    
    
    fbank_list.append(fbank)
    
    for idx, video in enumerate(videos):
        
        x, item = get_input(
            video,
            synchformer_cfg,
            control_type=control_type,
            duration=duration,
            local_rank=local_rank,
            re_encode=re_encode,
            mode=mode
            )

        model_inputs.append(x)
            
    fbank = torch.cat(fbank_list,dim=0)
    model_inputs = torch.cat(model_inputs,dim=0)

    batch = {
            "fname": videos,  # list
            "text": text,  # list
            "label_vector": None,
            "waveform": waveform,
            "model_inputs": model_inputs,
            # "energy":control,
            "log_mel_spec": fbank,
            "stft": torch.zeros((batchsize, 1024, 512)), # Not used in inference
            "start_sec": item['start_sec'],
            "end_sec": item['end_sec'],
            "attention_mask": item['mask'],
            "v_ranges": item['v_ranges']

    }
    return batch

def rewas_generation(
        text,
        videos,
        control_type,
        synchformer_cfg,
        original_audio_file_path = None,
        seed=42,
        duration=5,
        batchsize=1,
        re_encode=True,
        local_rank=None,
        mode="train"
    ):
    
    #seed_everything(int(seed))
    
    waveform = None
    
    if original_audio_file_path is not None:
        print(f'{original_audio_file_path=}')
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)

    if type(text)==list:
        batchsize = len(text)

    if not type(videos)==list:
        videos = [videos]

    #start_sec, end_sec 정보 여기 있음
    batch = make_batch_for_control_to_audio(
        text, videos, control_type, synchformer_cfg,
        waveform=waveform, batchsize=batchsize, local_rank=local_rank, re_encode=re_encode, duration=duration, mode=mode) 
    
    #print(f"start_sec: {batch['start_sec']}, end_sec: {batch['end_sec']}")
    return batch
    # x = hint = batch['model_inputs'].to(f'cuda:{local_rank}')
    # y = AudioCAM[:, start_secx100:end_secx100]

########################################################### 여기까지가 dataset + dataloader 함수부분 #########################################################################
# 지금 안된건 segment개수가 14로 고정되어있다는점


    # with torch.set_grad_enabled(False):
    #     with torch.autocast('cuda', enabled=synchformer_cfg.training.use_half_precision):
    #         hint = batch['model_inputs'].to(f'cuda:{local_rank}')
    #         hint = hint.permute(0, 1, 3, 2, 4, 5) # (B, S, C, Tv, H, W)
    #         hint = video_encoder(hint, for_loop=False)[0] # if for_loop = True: Segment is not combined with batch
    #         B, S, tv, D = hint.shape
    #         hint = hint.view(B, S*tv, D)

    #     hint = phi(hint.float())
    #     hint = hint.squeeze(2)
        
    #     print("hint.shape: ", hint.shape) ##
       




def build_control_model(
    control_type,
    ckpt_path=None,
    config=None,
    model_name="audioldm-m-full",
    distribution=False,
    device = "cuda:0",
    local_rank = None
):
    print(f"Load AudioLDM: {model_name}")
    
    control  = False
    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
        control  = True
    else:
        config = default_audioldm_config(model_name)
        
    # Use text as condition instead of using waveform during training
    config["model"]["params"]["cond_stage_key"] = "text"
    
    latent_diffusion = ReWaS(**config["model"]["params"])

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)


    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    if distribution: 
        latent_diffusion = torch.nn.parallel.DistributedDataParallel(latent_diffusion, device_ids=[local_rank])
        dist.barrier() 
        latent_diffusion = latent_diffusion.module

    model_idx = latent_diffusion.cond_stage_model_metadata['film_clap_cond1']["model_idx"]
    latent_diffusion.cond_stage_models[model_idx].embed_mode = "text"
    latent_diffusion.control_key = "energy"

    return latent_diffusion

def get_input(
        video_path,
        synchformer_cfg,
        control_type,
        device: str = 'cuda',
        audio_sr: int = 16000,
        video_fps: int = 25, 
        in_size: int = 256,
        duration: int = 5,
        local_rank = None,
        re_encode=True,
        mode="train"
        ):
    
    if video_path is None:
        return None

    if control_type == "energy_video":

        # if re_encode:
        #     video_path = re_encode_video('.cache', video_path, video_fps, audio_sr, in_size)
        #print("duration: ", duration)
        # rgb, audio, meta, start_sec, end_sec = get_video_and_audio(
        #     video_path, get_meta=True, duration=duration, target_fps=25, random_start=True)
        start_time = time.time()
        rgb, audio, meta, start_index, end_index, start_sec, end_sec = process_video_with_frame(video_path, target_fps=25, mode=mode)
        end_time = time.time()
        #print("process_video time: ", end_time - start_time)
        item_temp = dict(
                video=rgb, audio=audio, meta=meta, path=video_path, split=mode, start_index=start_index, end_index=end_index, start_sec=start_sec, end_sec=end_sec,
                targets={'v_start_i_sec': 0.0, 'offset_sec': 0.0 }
            )
        #print(f"rgb: {rgb.shape}, start_index: {start_index}, end_index: {end_index}, start_sec: {start_sec}, end_sec: {end_sec}")
        #print("before transform rgb.shape: ", rgb.shape)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(get_transforms(synchformer_cfg, ['train'])['train'])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        if mode == "train":
            item_temp = get_transforms(synchformer_cfg, ['train'])['train'](item_temp) 
        else:
            item_temp = get_transforms(synchformer_cfg, ['test'])['test'](item_temp) 
        # print("#############################################################################################################################################3")
        #print(f"rgb: {item_temp['video'].shape}, attention mask: , {item_temp['mask'].shape},  vlen_frames: {item_temp['vlen_frames']}, start_index: {start_index}, end_index: {end_index}, start_sec: {start_sec}, end_sec: {end_sec}")
        # print("video: \n", item_temp['v_ranges'])
        # print("attention_mask: \n", item_temp['mask'])
        # print("#############################################################################################################################################3 \n\n\n")
        x = item_temp['video'].unsqueeze(0)

    else:
        print('Undefined control type')

    return x, item_temp