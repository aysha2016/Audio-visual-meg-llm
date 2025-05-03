
import cv2
import torchaudio
import matplotlib.pyplot as plt

def display_inputs(image_path, audio_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Visual Input (Image)")
    plt.axis("off")
    plt.show()

    waveform, sample_rate = torchaudio.load(audio_path)
    plt.plot(waveform.t().numpy())
    plt.title("Audio Input (Waveform)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
