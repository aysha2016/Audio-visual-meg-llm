# 🧠 Audio-Visual MEG + LLM Interpretation Toolkit

This project simulates human brain responses (MEG signals) when a person sees and hears objects (e.g., seeing a cat and hearing it meow). It uses visual and audio input, simulates MEG-like responses, and generates natural language interpretations using a Large Language Model (LLM, GPT-2).

## 📦 Contents

```
audio_visual_meg_llm/
├── 1_image_audio_loader.py
├── 2_meg_simulation.py
├── 3_llm_interpretation.py
├── sample_image.jpg (optional)
├── sample_audio.wav (optional)
└── README.md
```

## 🚀 How to Use

### 1. Install dependencies

```bash
pip install transformers torchaudio opencv-python matplotlib
```

### 2. Run the scripts

```bash
python 1_image_audio_loader.py
python 2_meg_simulation.py
python 3_llm_interpretation.py
```

## 💡 Output

- 🎞️ Visual input from an image
- 🔊 Audio waveform display
- 🧠 Simulated MEG description
- 💬 LLM output such as: "I saw and heard a cat."

Replace input files as needed!
