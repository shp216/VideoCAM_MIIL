import os
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import cv2
import torch.nn.functional as F

from models import *
from models import ResNet38, Wavegram_Logmel_Cnn14
from pytorch_utils import move_data_to_device
import config

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

    # def generate_Gradcam_All(self, output):
    #     """Generate Grad-CAM for all classes and batches."""
    #     self.zero_grad()
    #     batch_size, num_classes = output.shape

    #     # Flatten output for batched processing
    #     class_scores = output.view(-1)  # Shape: (batch_size * num_classes)

    #     # Generate grad_outputs matching class_scores' shape
    #     grad_outputs = torch.ones_like(class_scores, device=output.device)  # Shape: (batch_size * num_classes)

    #     # Compute gradients for all classes and batches
    #     gradients = torch.autograd.grad(
    #         outputs=class_scores,
    #         inputs=self.feature_maps,
    #         grad_outputs=grad_outputs,
    #         retain_graph=True,
    #         create_graph=False
    #     )[0]  # Shape: (batch_size, channels, height, width)

    #     # Global Average Pooling for weights
    #     weights = gradients.mean(dim=(2, 3))  # Shape: (batch_size, channels)

    #     # Compute weighted sum of feature maps for CAM
    #     cams = torch.einsum("bchw,bc->bhw", self.feature_maps, weights)  # Shape: (batch_size, height, width)

    #     # Apply ReLU and normalize CAMs
    #     cams = torch.relu(cams)  # ReLU
    #     cams_min = cams.view(batch_size, -1).min(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    #     cams_max = cams.view(batch_size, -1).max(dim=1, keepdim=True)[0].view(batch_size, 1, 1)
    #     cams = (cams - cams_min) / (cams_max - cams_min + 1e-10)  # Normalize
    #     cams = (cams * 255).to(torch.uint8)  # Scale to [0, 255]

    #     return cams  # Shape: (batch_size, height, width)

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


def audio_tagging_GradCAM_All(args):
    """Perform audio tagging and generate Grad-CAMs."""
    sample_rate = args.sample_rate
    audio_folder_path = args.audio_folder_path
    batch_size = args.batch_size
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    cam_save_dir = args.cam_save_dir
    classes_num = config.classes_num
    labels = config.labels
    image_save_dir = "./MMG_test/Curated_Vggsound_GradCAM_images"
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
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
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
    
    print("######################################################################################################")
    print(f"global_min: {global_min}, global_max: {global_max}")
    print("######################################################################################################")

        
        # for i in range(normalized_aggregated_cams.shape[0]):
        #     audio_file = audio_files[batch_idx * batch_size + i]
            
        #     # CAM 이미지를 저장할 폴더 경로
        #     cam_save_dir = 'CAM_images'
        #     os.makedirs(cam_save_dir, exist_ok=True)

        #     # 클래스-시간 CAM 이미지를 저장하는 함수 호출
        #     save_all_class_time_cam_image(normalized_aggregated_cams[i].detach().cpu().numpy(), labels, cam_save_dir, audio_file)

        # # Scale values to [0, 255] and convert to byte tensor
        # #cams_resized = (cams_resized * 255).clamp(0, 255).byte()

        # for i in range(waveforms.shape[0]):
        #     audio_file = audio_files[batch_idx * batch_size + i]

        #     # Get the CAM for the current waveform
        #     cams_single = cams_resized[i].cpu()  # Shape: (num_classes, 321, 64)
        #     print("cams_single.shape: ", cams_single.shape)

        #     # CAM과 클래스 확률 곱하기
        #     print("clipwise_output.shape: ", clipwise_output.shape)
        #     clipwise_output_single = clipwise_output[i]  # 해당 배치의 클래스 확률
        #     print("clipwise_output_single.shape: ", clipwise_output_single.shape)
        #     sorted_indexes = np.argsort(clipwise_output_single)[::-1]
        #     topk_name = []
        #     for k in range(5):
        #         class_idx = sorted_indexes[k]
        #         class_name = np.array(labels)[class_idx]
        #         topk_name.append(class_name)
        #     print("topk_name: ", topk_name)
        #     print(clipwise_output_single.shape)
        #     scaled_cams = cams_single.float()
        #     scaled_cams = cams_single.float() * clipwise_output_single[:, None, None]  # Shape: (num_classes, 321, 64)

        #     # Frequency 축으로 CAM 압축 (64 -> mean)
        #     aggregated_cams = scaled_cams.mean(dim=2)  # Shape: (num_classes, 321)
        #     print("####################################################################################################")
        #     print(f"Cam max: {aggregated_cams.max()}, Cam min: {aggregated_cams.min()}")
        #     print("####################################################################################################")
        #     print(f"Aggregated CAMs shape: {aggregated_cams.shape}")

        #     # CAM 이미지를 저장할 폴더 경로
        #     cam_save_dir = 'CAM_images'
        #     os.makedirs(cam_save_dir, exist_ok=True)

        #     # 클래스-시간 CAM 이미지를 저장하는 함수 호출
        #     save_all_class_time_cam_image(aggregated_cams.detach().cpu().numpy(), labels, cam_save_dir, audio_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Tagging and CAM Generation')

    parser.add_argument('--sample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)
    parser.add_argument('--model_type', type=str, default="ResNet38_CAM")
    parser.add_argument('--checkpoint_path', type=str, default="./ResNet38_mAP=0.434.pth")
    parser.add_argument('--audio_folder_path', type=str, default="./audio_trimmed")
    parser.add_argument('--cam_save_dir', type=str, default="./GradCAM_images")
    parser.add_argument('--cam_image_save_dir', type=str, default="./MMG_test/Curated_Vggsound_GradCAM_images")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    audio_tagging_GradCAM_All(args)
