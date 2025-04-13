import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

def wav_to_mel_spectrogram(wav_path, save_path, n_mels=128, hop_length=512, sr=None):
    """
    Converts a WAV file to a Mel spectrogram and saves it as an image.

    Args:
        wav_path (str): Path to the input WAV file.
        save_path (str): Path to save the Mel spectrogram image.
        n_mels (int): Number of Mel bands to generate.
        hop_length (int): Number of samples between successive frames.
        sr (int or None): Target sampling rate. If None, uses the native rate.

    Returns:
        None
    """
    # Load the WAV file
    y, sr = librosa.load(wav_path, sr=sr)

    # Compute the Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)

    # Convert power spectrogram to dB (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create a directory for saving the image if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Plot and save the Mel spectrogram as an image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Mel spectrogram saved to {save_path}")

# Example usage
wav_path = "./ours_test/vggsound_sparse__52ntwwQyv4_70_dog_barking_train_fixed.wav"  # Replace with your WAV file path
save_path = "./ours_test/mel_spectrogram_dog.png"  # Replace with the desired output image path
wav_to_mel_spectrogram(wav_path, save_path)
