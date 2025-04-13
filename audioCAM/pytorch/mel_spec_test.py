import torch
import torchaudio
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
import matplotlib.pyplot as plt

# 1. Load an audio file using torchaudio
waveform, sample_rate = torchaudio.load("CAT.wav", normalize=True)  # "CAT.wav" 파일 로드

# 2. Convert to mono if it's stereo
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)  # (1, audio_length) 형태로 변환

# 3. Define the STFT and Mel FilterBank parameters
n_fft = 1024
hop_length = 320
n_mels = 64
target_sample_rate = 32000

# 4. Resample if necessary
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)
    sample_rate = target_sample_rate

# 5. Create Spectrogram and LogmelFilterBank instances
spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)

# 6. Compute spectrogram and log mel-spectrogram
with torch.no_grad():
    # Spectrogram
    spectrogram = spectrogram_extractor(waveform)  # shape: (batch_size, 1, time, freq)
    
    # Log mel-spectrogram
    logmel_spectrogram = logmel_extractor(spectrogram)  # shape: (batch_size, 1, time, n_mels)

# 7. Squeeze the dimensions to get a 2D array for visualization
logmel_spectrogram_np = logmel_spectrogram.squeeze().numpy()  # shape: (time, n_mels)

# 8. Mel-spectrogram visualization
plt.figure(figsize=(10, 4))
plt.imshow(logmel_spectrogram_np.T, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(format='%+2.0f dB')
plt.title('Log Mel-Spectrogram')
plt.xlabel('Time Frames')
plt.ylabel('Mel Bands')
plt.tight_layout()
plt.show()
