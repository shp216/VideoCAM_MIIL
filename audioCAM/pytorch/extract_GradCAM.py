import os
import sys
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import cv2

from models import *
from pytorch_utils import move_data_to_device
import config
from models import ResNet38, Wavegram_Logmel_Cnn14

class ResNet38_GradCAM(ResNet38):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, device):
        super(ResNet38_GradCAM, self).__init__(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num)
        self.device = device
        self.feature_map = None  # 마지막 Feature Map
        self.gradients = None    # Feature Map에 대한 그래디언트
        self.output=None 
        
    def save_gradients(self, grad):
        """Backward에서 계산된 그래디언트를 저장."""
        self.gradients = grad

    def forward(self, input, mixup_lambda=None):
        """
        Forward: 입력 데이터를 처리하고 Feature Map을 저장.
        """
        # Spectrogram 추출
        print("#########################################################################################3")
        print("input shape: ", input.shape)
        x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)            # (batch_size, 1, time_steps, mel_bins)
        
        x = x.transpose(1, 3)
        x = self.bn0(x) #BatchNorm
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)
        print("Start mel spectrogram shape: ", x.shape)
        print("#########################################################################################3")

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        print("After conv_block1 shape: ", x.shape)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        print("After resnet shape: ", x.shape)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(2, 2))
        print("After avgpool2d shape: ", x.shape)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        print("After conv_block_after1 shape: ", x.shape)
        x = torch.nn.functional.dropout(x, p=0.2, training=self.training, inplace=True)
        
        # 마지막 Feature Map
        x.requires_grad_(True)
        x.register_hook(self.save_gradients)  # Hook을 등록하여 그래디언트 저장
        self.feature_map = x

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        print("x1.shape, x2.shape, x.shape: ", x1.shape, x2.shape, x.shape)
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.nn.functional.relu_(self.fc1(x))
        print("After fc1, relu shape: ", x.shape)
        embedding = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        output = self.fc_audioset(x)
        print("After fc_audioset layer shape: ", output.shape)
        self.output = output
        clipwise_output = torch.sigmoid(output)
        print("clipwise_output.shape: ", clipwise_output.shape)
        
        output_dict = {'clipwise_output': clipwise_output, 'output': output, 'embedding': embedding}

        return output_dict


    def generate_Gradcam_All(self, class_idx, output):
        """
        Grad-CAM 생성 메서드 (반복문 활용).
        Args:
            class_idx (int or None): Grad-CAM을 생성할 클래스의 인덱스. None이면 모든 클래스에 대해 생성.
            output (torch.Tensor): 모델의 출력 (batch_size, num_classes).

        Returns:
            cams (numpy array): Grad-CAM 결과 맵 (num_classes, height, width) 또는 (height, width) (단일 클래스).
        """
        # Gradients 초기화
        self.zero_grad()

        # 모든 클래스 처리
        if class_idx is None:
            num_classes = output.shape[1]  # Number of classes
            all_gradients = []

            for i in range(num_classes):
                self.zero_grad()  # 각 클래스의 backward 전에 gradient 초기화

                # Calculate gradients for the i-th class
                class_score = output[0, i]
                class_score.backward(retain_graph=True)

                # Gradients와 Feature Map 가져오기
                gradients = self.gradients[0].cpu().data.numpy()  # Shape: (channels, height, width)
                all_gradients.append(gradients)

            # Gradients를 NumPy 배열로 변환
            all_gradients = np.stack(all_gradients, axis=0)  # Shape: (num_classes, channels, height, width)
            feature_map = self.feature_map[0].cpu().data.numpy()  # Shape: (channels, height, width)
            print("all_gradients.shape: ", all_gradients.shape)
            print("feature_map.shape: ", feature_map.shape)

            # Gradients의 Global Average Pooling (채널별 중요도 계산)
            weights = np.mean(all_gradients, axis=(2, 3))  # Shape: (num_classes, channels)

            # Weighted sum으로 Grad-CAM 계산 (벡터화)
            cams = np.tensordot(weights, feature_map, axes=([1], [0]))  # Shape: (num_classes, height, width)
            print("output cams.shape: ", cams.shape)
            # ReLU 적용 및 정규화
            cams = np.maximum(cams, 0)  # ReLU
            cams = (cams - np.min(cams, axis=(1, 2), keepdims=True)) / (np.max(cams, axis=(1, 2), keepdims=True) - np.min(cams, axis=(1, 2), keepdims=True) + 1e-10)
            cams = (cams * 255).astype(np.uint8)  # Normalize to [0, 255]
        else:
            # 단일 클래스 처리
            self.zero_grad()

            class_score = output[0, class_idx]  # 단일 클래스의 점수
            class_score.backward(retain_graph=True)

            # Gradients와 Feature Map 가져오기
            gradients = self.gradients[0].cpu().data.numpy()  # Shape: (channels, height, width)
            feature_map = self.feature_map[0].cpu().data.numpy()  # Shape: (channels, height, width)

            # Gradients의 Global Average Pooling (채널별 중요도 계산)
            weights = np.mean(gradients, axis=(1, 2))  # Shape: (channels,)
            cam = np.tensordot(weights, feature_map, axes=([0], [0]))  # Shape: (height, width)

            # ReLU 적용 및 정규화
            cam = np.maximum(cam, 0)
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)
            cams = (cam * 255).astype(np.uint8)  # Shape: (height, width)

        return cams


def save_all_class_time_cam_image(aggregated_cams_array, class_names, save_dir, audio_file):
    """
    모든 클래스(y축)와 시간(x축)을 갖는 CAM 값을 저장하는 함수
    """
    audio_name = audio_file.split(".")[0]
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 2 + len(class_names) * 0.5))
    
    # CAM 데이터를 시각화
    im = ax.imshow(
        aggregated_cams_array,
        aspect='auto', 
        origin='upper', 
        cmap='jet', 
        interpolation='nearest'
    )

    # 클래스별 경계선 추가
    for idx in range(1, len(class_names)):
        ax.hlines(
            y=idx - 0.5, 
            xmin=0, 
            xmax=aggregated_cams_array.shape[1] - 1, 
            colors='white', 
            linestyles='-', 
            linewidth=1.0
        )

    # y축: 클래스 이름 설정
    ax.set_yticks(np.arange(len(class_names))) 
    ax.set_yticklabels(class_names)

    # x축: 시간 프레임 설정
    ax.set_xlabel('Time Frames')
    ax.set_title('Aggregated CAM over Time for All Classes')

    # 색상바 추가
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.02)

    # 시각화 저장
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{audio_name}_all_classes_combined.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Aggregated CAM image saved at {save_path}")

def audio_tagging_GradCAM_All(args):
    """오디오 클립의 태깅 및 CAM 생성 결과를 추론합니다."""
    # 인자 설정
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_folder_path = args.audio_folder_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    print("classes_num:", classes_num)

    # 모델 초기화
    Model = ResNet38_GradCAM  # 직접 참조
    model = Model(sample_rate=sample_rate, window_size=window_size, hop_size=hop_size, 
                  mel_bins=mel_bins, fmin=fmin, fmax=fmax, classes_num=classes_num, device=device)

    # 사전 학습된 모델 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)  # 모델을 디바이스로 이동
    model.eval()  # 평가 모드로 전환

    # 오디오 파일 처리
    audio_folder_list = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
    print(f"Found {len(audio_folder_list)} audio files in {audio_folder_path}")
    
    for audio_file in audio_folder_list:
        # 오디오 로드
        audio_path = os.path.join(audio_folder_path, audio_file)
        waveform, _ = librosa.core.load(audio_path, sr=sample_rate, mono=True)
        waveform = waveform[None, :]  # (1, audio_length)
        waveform = move_data_to_device(waveform, device)
        print("waveform.shape", waveform.shape)

        # 추론
        with torch.set_grad_enabled(True):
            output = model(waveform)

        # Grad-CAM 계산에 필요한 모델 출력 추출
        clipwise_output = output['clipwise_output']
        output = output['output']  # Grad-CAM에 필요한 모델 출력

        # Spectrogram 추출 (CAM 시각화를 위해)
        spectrogram = model.logmel_extractor(model.spectrogram_extractor(waveform)).detach().cpu().numpy()[0][0]
        print(f"Spectrogram shape: {spectrogram.shape}") 

        # Grad-CAM 생성 (모든 클래스에 대해 한 번에 계산)
        cams = model.generate_Gradcam_All(None, output) # Shape: (num_classes, height, width)
        cams = cams.transpose(0,2,1)
        print(f"Generated CAMs shape: {cams.shape}")
        
        spectrogram_scaled = spectrogram - np.min(spectrogram)
        spectrogram_scaled = spectrogram_scaled / np.max(spectrogram_scaled)
        spectrogram_scaled = np.uint8(255 * spectrogram_scaled)
        spectrogram_scaled = spectrogram_scaled.T
        print("spectrogram_scaled.shape: ", spectrogram_scaled.shape)
        
        # CAM을 Spectrogram 크기로 리사이즈
        resized_cams = []
        for cam in cams:
            resized_cam = cv2.resize(cam, (spectrogram.shape[0], spectrogram.shape[1]))
            resized_cams.append(resized_cam)
        resized_cams = np.stack(resized_cams, axis=0)  # Shape: (num_classes, spectrogram_height, spectrogram_width)

         # 클래스 확률 가져오기
        clipwise_output = clipwise_output.detach().cpu().numpy()[0]

        # CAM과 클래스 확률 곱하기
        
        scaled_cams = resized_cams * clipwise_output[:, None, None]  # Shape: (num_classes, spectrogram_height, spectrogram_width)
        
        # 주파수 축으로 CAM 압축
        aggregated_cams = np.mean(scaled_cams, axis=1)  # Shape: (num_classes, time_steps)
        print(f"Aggregated CAMs shape: {aggregated_cams.shape}")

        # CAM 이미지를 저장할 폴더 경로
        cam_save_dir = 'GradCAM_images'
        os.makedirs(cam_save_dir, exist_ok=True)

        # 클래스-시간 CAM 이미지를 저장하는 함수 호출
        save_all_class_time_cam_image(aggregated_cams, labels, cam_save_dir, audio_file)



        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Tagging and CAM Generation')

    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=160)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default="ResNet38_CAM")
    #parser.add_argument('--checkpoint_path', type=str, default="C:/Users/Noah/Desktop/MMG/Codes/MMG/CAM/audioset_tagging_cnn/pytorch/Wavegram_Logmel_Cnn14_mAP=0.439.pth")
    parser.add_argument('--checkpoint_path', type=str, default="./ResNet38_mAP=0.434.pth")
    parser.add_argument('--audio_folder_path', type=str, default="./Audiosync_trimmed")
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    audio_tagging_GradCAM_All(args)
