import numpy as np
import matplotlib.pyplot as plt

class MEGSimulator:
    def __init__(self):
        self.meg_data = None
        self.time = None

    def simulate_responses(self, audio_info):
        """Simulate MEG responses based on audio timing information."""
        if audio_info is None:
            raise ValueError("Audio information is required for simulation")

        n_samples = audio_info['n_samples']
        sample_rate = audio_info['sample_rate']
        self.time = np.linspace(0, n_samples / sample_rate, n_samples)

        # Simulate visual response (occipital cortex)
        visual_response = np.random.normal(0, 0.5, n_samples)
        visual_activation = np.exp(-((self.time - 0.2) ** 2) / (2 * 0.05**2))
        visual_data = visual_response + visual_activation

        # Simulate auditory response (auditory cortex)
        auditory_response = np.random.normal(0, 0.5, n_samples)
        auditory_activation = np.exp(-((self.time - 0.5) ** 2) / (2 * 0.05**2))
        auditory_data = auditory_response + auditory_activation

        # Combined response
        combined_data = visual_data + auditory_data

        self.meg_data = {
            'visual': visual_data,
            'auditory': auditory_data,
            'combined': combined_data,
            'time': self.time
        }

        return self.meg_data

    def plot_responses(self):
        """Plot the simulated MEG responses."""
        if self.meg_data is None:
            raise ValueError("No MEG data available. Run simulate_responses first.")

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(self.time, self.meg_data['visual'])
        plt.title('Simulated Visual MEG Response (Occipital Cortex)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 2)
        plt.plot(self.time, self.meg_data['auditory'])
        plt.title('Simulated Auditory MEG Response (Auditory Cortex)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.subplot(3, 1, 3)
        plt.plot(self.time, self.meg_data['combined'])
        plt.title('Combined Audio-Visual MEG Response')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    def generate_description(self):
        """Generate a natural language description of the MEG responses."""
        if self.meg_data is None:
            raise ValueError("No MEG data available. Run simulate_responses first.")

        visual_peak_time = self.time[np.argmax(self.meg_data['visual'])]
        auditory_peak_time = self.time[np.argmax(self.meg_data['auditory'])]

        return f'''
Visual input detected in occipital cortex (sharp spike ~{visual_peak_time:.2f}s).
Audio input detected in auditory cortex (broadband activity ~{auditory_peak_time:.2f}s).
Combined AV response suggests object recognition and semantic retrieval.
''' 