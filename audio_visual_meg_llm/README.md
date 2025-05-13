# 🧠 Audio-Visual MEG + LLM Interpretation System

This project simulates human brain responses (MEG signals) when processing audio-visual inputs and generates natural language interpretations using GPT-2. It can be used both as a standalone Python package and as a Google Colab notebook.

## 📦 Project Structure

```
audio_visual_meg_llm/
├── requirements.txt           # Project dependencies
├── input_processor.py        # Handles image and audio input processing
├── meg_simulator.py         # Simulates MEG brain responses
├── llm_processor.py         # Manages LLM-based interpretation
├── main.py                  # Standalone script version
├── colab_notebook.py        # Colab-specific version
└── README.md               # This file
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/aysha2016/audio_visual_meg_llm.git
cd audio_visual_meg_llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Standalone Version

Run the main script:
```bash
python main.py
```

The script will:
1. Prompt for image and audio file uploads
2. Display the inputs
3. Simulate MEG responses
4. Generate interpretations
5. Allow interactive Q&A

### Google Colab Version

1. Open [Google Colab](https://colab.research.google.com)
2. Upload all project files
3. Open `colab_notebook.py`
4. Run the notebook

The Colab version provides the same functionality with a more interactive interface.

## 🔧 Components

### InputProcessor (`input_processor.py`)
- Handles image and audio file loading
- Displays visualizations
- Provides audio playback
- Manages input data for MEG simulation

### MEGSimulator (`meg_simulator.py`)
- Simulates brain responses to audio-visual input
- Generates MEG-like signals
- Creates visualizations of neural activity
- Produces natural language descriptions of the responses

### LLMProcessor (`llm_processor.py`)
- Manages GPT-2 model for interpretation
- Generates natural language responses
- Handles interactive Q&A
- Provides context-aware answers

## 📝 Example Usage

```python
from input_processor import InputProcessor
from meg_simulator import MEGSimulator
from llm_processor import LLMProcessor

# Initialize components
input_processor = InputProcessor()
meg_simulator = MEGSimulator()
llm_processor = LLMProcessor()

# Load and process inputs
input_processor.load_image("image.jpg")
input_processor.load_audio("audio.wav")

# Display inputs
input_processor.display_inputs()

# Simulate MEG responses
meg_data = meg_simulator.simulate_responses(input_processor.get_audio_info())
meg_simulator.plot_responses()

# Generate interpretation
interpretation = llm_processor.generate_interpretation(
    meg_simulator.generate_description()
)
```
  
