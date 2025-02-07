import os
from pathlib import Path
import subprocess
import random
import numpy as np
import soundfile as sf
import torch
import torchvision
#from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io import VideoFileClip
import os
import torch.nn.functional as F
from tqdm import tqdm
import math
import time
import ffmpeg

from encoder.encoder_utils import which_ffmpeg

def process_video_with_frame2(
    path,
    target_fps=25
):
    """
    비디오를 FPS만 변환한 후, 랜덤으로 start_index와 v_len에 따라 슬라이싱합니다.

    Args:
        path (str): 원본 비디오 경로.
        target_fps (int): 대상 FPS.

    Returns:
        torch.Tensor: 최종 비디오 텐서 (T, C, H, W).
        dict: 메타데이터 정보.
        int: 시작 인덱스.
        int: 종료 인덱스.
    """
    # FFmpeg 명령으로 FPS 변환 및 메모리 로드
    try:
        probe = ffmpeg.probe(path)
        video_info = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])

        out, _ = (
            ffmpeg
            .input(path)
            .filter('fps', fps=target_fps)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )

        # 비디오 데이터를 numpy로 변환
        # 버퍼를 numpy 배열로 변환하면서 복사
        video_np = np.frombuffer(out, dtype=np.uint8).copy()

        # PyTorch 텐서로 변환
        video = torch.from_numpy(video_np)
        # video = torch.frombuffer(out, dtype=torch.uint8)
        total_frames = video.numel() // (3 * width * height)
        video = video.view(total_frames, height, width, 3).permute(0, 3, 1, 2)  # (T, C, H, W)

    except Exception as e:
        raise RuntimeError(f"FFmpeg processing failed: {e}")

    assert total_frames > 0, "비디오에 프레임이 없습니다."
    
    video, _, meta = torchvision.io.read_video(path, output_format="TCHW", pts_unit='sec')
    total_frames = video.shape[0]
    # 랜덤 start_index와 v_len 생성
    start_index = random.randint(0, max(0, total_frames - 3))
    v_len = random.randint(2, min(125, total_frames - start_index))
    
    end_index = start_index + v_len

    # 슬라이싱
    sliced_rgb = video[start_index:end_index]

    # 메타데이터 생성
    meta_out = {'video': {'fps': [target_fps]}}
    duration = total_frames / meta["video_fps"]
    start_sec = start_index / int(meta["video_fps"])
    end_sec = (end_index) / int(meta["video_fps"])

    # 오디오 처리 생략 또는 필요 시 ffmpeg로 추가 구현
    meta_out['audio'] = {'framerate': [16000]}
    audio = torch.tensor([])  # 더미 값

    return sliced_rgb, audio, meta_out, start_index, end_index, start_sec, end_sec


def process_video_with_frame(
    path,
    target_fps=25,
    temp_dir="./temp_video",
    mode="train"
):
    """
    비디오를 FPS만 변환한 후, 랜덤으로 start_index와 v_len에 따라 슬라이싱합니다.

    Args:
        path (str): 원본 비디오 경로.
        target_fps (int): 대상 FPS.
        temp_dir (str): 임시 비디오 파일을 저장할 디렉토리.

    Returns:
        torch.Tensor: 최종 비디오 텐서 (T, C, H, W).
        dict: 메타데이터 정보.
        int: 시작 인덱스.
        int: 종료 인덱스.
    """
    
    # if mode == "test":
    #     # temp_video 폴더 생성
    #     os.makedirs(temp_dir, exist_ok=True)

    #     # 임시 파일 경로
    #     file_name = os.path.basename(path)
    #     temp_output = os.path.join(temp_dir, file_name)

    #     # ffmpeg 명령으로 FPS만 변환
    #     cmd = (
    #         f"ffmpeg -y -i {path} -vf fps={target_fps} {temp_output} -loglevel error"
    #     )

    #     ret_code = subprocess.call(cmd, shell=True)
    #     if ret_code != 0:
    #         raise RuntimeError(f"FFmpeg command failed with code {ret_code}: {cmd}")

    # torchvision으로 비디오 로드
    rgb, audio, meta = torchvision.io.read_video(path, output_format="TCHW", pts_unit='sec')
    assert meta["video_fps"] == target_fps, f"Target FPS {target_fps}와 다릅니다."

    total_frames = rgb.shape[0]
    assert total_frames > 0, "비디오에 프레임이 없습니다."

    if mode =="train":
        # 랜덤 start_index와 v_len 생성
        start_index = random.randint(0, max(0, total_frames - 16))
        #v_len = random.randint(4,40)
        v_len = random.randint(16, min(128, total_frames - start_index))
    
        end_index = start_index + v_len
        # 슬라이싱
        sliced_rgb = rgb[start_index:end_index]
    
    else:
        start_index = 0
        v_len = min(total_frames, 128)
        end_index = start_index + v_len
        sliced_rgb = rgb[start_index:end_index]

    # os.remove(temp_output)
    
    # Handle metadata
    meta_out = {'video': {'fps': [meta['video_fps']]}}
    duration = total_frames / meta["video_fps"]
    start_sec = start_index / int(meta["video_fps"])
    end_sec = (end_index) / int(meta["video_fps"])  # end_index 포함
    if meta.get('audio_fps'):
        alen = int(duration * meta['audio_fps'])
        audio = audio.mean(dim=0)
        audio = audio[int(start_sec*meta['audio_fps']):(int(start_sec*meta['audio_fps'])+alen)]
        meta_out['audio'] =  {'framerate': [meta['audio_fps']]}
    else:
        meta_out['audio'] =  {'framerate': [16000]}

    return sliced_rgb, audio, meta_out, start_index, end_index, start_sec, end_sec





def process_video_and_audio(
    path,
    duration=5,
    target_fps=25,
    random_start=True, 
    total_duration_sec=10,
    temp_dir="./temp_video"
):
    """
    비디오를 5초로 자르고, 25fps로 변환하며 오디오와 메타데이터도 처리합니다.

    Args:
        path (str): 원본 비디오 경로.
        target_fps (int): 대상 FPS.
        duration (int): 자를 비디오 길이 (초 단위).
        total_duration_sec (int): 원본 비디오 총 길이 (초 단위).
        temp_dir (str): 임시 비디오 파일을 저장할 디렉토리.

    Returns:
        torch.Tensor: 최종 비디오 텐서 (T, C, H, W).
        torch.Tensor: 오디오 텐서.
        dict: 메타데이터 정보.
    """
    # 랜덤 시작 시간 계산
    if random_start:
        start_sec = random.randint(0, max(0, math.floor(total_duration_sec - duration)))
    else:
        start_sec = 0
    end_sec = start_sec + duration
    # temp_video 폴더 생성
    os.makedirs(temp_dir, exist_ok=True)

    # 임시 파일 경로
    file_name = os.path.basename(path)
    temp_output = os.path.join(temp_dir, file_name)

    # # ffmpeg 명령으로 FPS 변경 및 5초 자르기
    # cmd = (
    # f"ffmpeg -y -i {path} -ss {start_sec} -to {end_sec} "
    # f"-vf fps={target_fps} {temp_output} -loglevel error"
    # )
    
    cmd = (
    f"ffmpeg -y -i {path} -ss {start_sec} -to {end_sec} "
    f"-vf fps={target_fps} {temp_output} -loglevel error"
    )


    ret_code = subprocess.call(cmd, shell=True)
    if ret_code != 0:
        raise RuntimeError(f"FFmpeg command failed with code {ret_code}: {cmd}")
    # torchvision으로 비디오와 오디오 로드
    rgb, audio, meta = torchvision.io.read_video(temp_output, output_format="TCHW", pts_unit='sec')
    assert meta["video_fps"] == target_fps, f"Target FPS {target_fps}와 다릅니다."
    print("audio.shape: ", audio.shape)
    #print("rgb.shape: ", rgb.shape, meta['video_fps'], rgb.dtype)
    # 임시 파일 삭제
    os.remove(temp_output)
    
    # Handle metadata
    meta_out = {'video': {'fps': [meta['video_fps']]}}
    
    if meta.get('audio_fps'):
        alen = int(duration * meta['audio_fps'])
        audio = audio.mean(dim=0)
        audio = audio[start_sec*meta['audio_fps']:(start_sec*meta['audio_fps']+alen)]
        meta_out['audio'] =  {'framerate': [meta['audio_fps']]}
    else:
        meta_out['audio'] =  {'framerate': [16000]}
    # 결과 반환
    return rgb, audio, meta_out, start_sec, end_sec

def get_video_and_audio(
    path, 
    get_meta=False, 
    duration=5, 
    target_fps=25, 
    random_start=False, 
    audio_total_duration=10,
    start_sec = 0,
    end_sec = None
):
    """
    Read video and optionally audio from a file, adjust FPS, and crop to the desired duration.

    Args:
        path (str): Path to the video file.
        get_meta (bool): Whether to include metadata in the output.
        duration (float): Duration of the output video in seconds.
        target_fps (int): Target FPS for the output video.
        random_start (bool): Whether to crop from a random starting point.
        audio_cam_length (int): Total length of the audio CAM.
        audio_total_duration (int): Total duration of the audio CAM in seconds.

    Returns:
        torch.Tensor: Cropped video tensor of shape (T, C, H, W).
        torch.Tensor: Audio tensor (if audio exists), otherwise None.
        dict: Metadata containing video and audio FPS.
    """
    # Read the video (and optionally audio) with meta information
    rgb, audio, meta = torchvision.io.read_video(str(path), start_sec, end_sec, 'sec', output_format='TCHW')
    #rgb, audio, meta = torchvision.io.read_video(str(path), output_format='TCHW')
    #print("Audio here!!!!!!: ", audio.shape)
    assert meta['video_fps'], f"No video FPS found for {path}"
    
    # Original FPS from metadata
    original_fps = meta['video_fps']
    #print(f"Original FPS: {original_fps}, Target FPS: {target_fps}")

    # Resample video to target FPS
    if original_fps != target_fps:
        rgb_batch = rgb.unsqueeze(0)  # Shape: (1, T, C, H, W)
        B, T, C, H, W = rgb_batch.shape
        #print("rgb shape ->: ", rgb_batch.shape)
        
        rgb_batch = rgb_batch.float()  # Normalize if the values are 0-255

        # Resample videos to target FPS
        if original_fps != target_fps:
            #print(f"Resampling videos from {original_fps} fps to {target_fps} fps...")
            rgb_batch = rgb_batch.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)
            resampled_frames = int(T * target_fps / original_fps)
            rgb_batch = F.interpolate(rgb_batch, size=(resampled_frames, H, W), mode='trilinear', align_corners=False)
            rgb_batch = rgb_batch.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)

        #print("After resampling, RGB batch shape:", rgb_batch.shape)
        meta['video_fps'] = target_fps  # Update FPS in metadata
    
    rgb = rgb.squeeze(0)
    # Calculate time-based cropping
    total_duration_sec = 10  # Total video duration in seconds
    if random_start:
        start_sec = random.randint(0, max(0, math.floor(total_duration_sec - duration)))
    else:
        start_sec = 0
    end_sec = start_sec + duration
    #print("start_sec, end_sec: ", start_sec, end_sec)
    # Convert time to frame indices for video
    start_frame = int(start_sec * target_fps)
    end_frame = math.ceil(end_sec * target_fps)
    #print("start_frame, end_frame: ", start_frame, end_frame)

    rgb = rgb[start_frame:end_frame]
    print("rgb.shape: ", rgb.shape, meta['video_fps'], rgb.dtype)

    # Handle metadata
    meta_out = {'video': {'fps': [meta['video_fps']]}}
    
    if meta.get('audio_fps'):
        alen = int(duration * meta['audio_fps'])
        audio = audio.mean(dim=0)
        audio = audio[start_sec*meta['audio_fps']:(start_sec*meta['audio_fps']+alen)]
        meta_out['audio'] =  {'framerate': [meta['audio_fps']]}
    else:
        meta_out['audio'] =  {'framerate': [16000]}

    #print(f"Video cropped from {start_frame} to {end_frame} (frames)")

    
   

    return rgb, audio, meta_out, start_sec, end_sec



def save_wave(waveform, savepath, name="outwav"):
    if type(name) is not list:
        name = [name] * waveform.shape[0]

    paths = []
    for i in range(waveform.shape[0]):
        path = os.path.join(
            savepath,
            "%s_%s.wav"
            % (
                os.path.basename(name[i])
                if (not ".wav" in name[i])
                else os.path.basename(name[i]).split(".")[0],
                i,
            ),
        )
        paths.append(path)
        print("Save audio to %s" % path)
        sf.write(path, waveform[i, 0], samplerate=16000)
        
    return paths

def save_video(audio_path, video_path):

    video_clip = VideoFileClip(video_path)
    video_clip = video_clip.subclip(0, 5) # generated audio duration is 5 seconds.

    audio_clip = AudioFileClip(audio_path)
    video_clip = video_clip.set_audio(audio_clip)
    
    # Output file path for the final video with audio
    out_video_path = audio_path.replace('.wav', '.mp4')

    # Write the video clip with the audio to a new file
    video_clip.write_videofile(out_video_path, audio_codec='aac')

    # Close the clips
    video_clip.close()
    audio_clip.close()

    return
        
def re_encode_video(new_path, path, vfps=25, afps=16000, in_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    
    os.makedirs(new_path, exist_ok=True)

    new_path += f"/{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4"
    new_path = str(new_path)
    cmd = f"{which_ffmpeg()}"
    # no info/error printing
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -y -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf minterpolate='fps={vfps}',scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    #cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f" {new_path}"
    subprocess.call(cmd.split())
    return new_path

