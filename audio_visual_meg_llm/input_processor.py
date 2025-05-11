import cv2
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display

class InputProcessor:
    def __init__(self):
        self.image_data = None
        self.audio_data = None
        self.sample_rate = None

    def load_image(self, image_path):
        """Load and process image file."""
        try:
            self.image_data = cv2.imread(image_path)
            if self.image_data is None:
                raise ValueError(f"Could not load image from {image_path}")
            return True
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return False

    def load_audio(self, audio_path):
        """Load and process audio file."""
        try:
            self.audio_data, self.sample_rate = torchaudio.load(audio_path)
            return True
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            return False

    def display_inputs(self, show_audio=True):
        """Display the loaded image and audio waveform."""
        if self.image_data is not None:
            # Display image
            img_rgb = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 6))
            plt.imshow(img_rgb)
            plt.title("Visual Input (Image)")
            plt.axis("off")
            plt.show()

        if self.audio_data is not None and show_audio:
            # Display audio waveform
            plt.figure(figsize=(10, 4))
            plt.plot(self.audio_data.t().numpy())
            plt.title("Audio Input (Waveform)")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude")
            plt.show()

            # Play audio
            display(Audio(self.audio_data.numpy(), rate=self.sample_rate))

    def get_audio_info(self):
        """Get audio information for MEG simulation."""
        if self.audio_data is not None:
            return {
                'waveform': self.audio_data,
                'sample_rate': self.sample_rate,
                'n_samples': len(self.audio_data[0])
            }
        return None 