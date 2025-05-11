#@title 🧠 Audio-Visual MEG + LLM Interpretation System
#@markdown This notebook simulates human brain responses (MEG signals) when processing audio-visual inputs and generates natural language interpretations using GPT-2.

#@markdown ## 1. Install Required Packages
!pip install transformers torchaudio opencv-python matplotlib ffmpeg-python

#@markdown ## 2. Import Libraries
import torch
import torchaudio
import cv2
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from google.colab import files
import io
from IPython.display import Audio, Image, display

#@markdown ## 3. Upload Files
#@markdown Upload your image and audio files. Supported formats:
#@markdown - Image: JPG, PNG
#@markdown - Audio: WAV, MP3

# Upload image file
print("Upload an image file (JPG/PNG):")
uploaded_image = files.upload()
image_filename = list(uploaded_image.keys())[0]

# Upload audio file
print("\nUpload an audio file (WAV/MP3):")
uploaded_audio = files.upload()
audio_filename = list(uploaded_audio.keys())[0]

#@markdown ## 4. Display Inputs
def display_inputs(image_path, audio_path):
    # Display image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title("Visual Input (Image)")
    plt.axis("off")
    plt.show()
    
    # Display audio waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title("Audio Input (Waveform)")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()
    
    # Play audio
    display(Audio(audio_path))

# Display the uploaded files
display_inputs(image_filename, audio_filename)

#@markdown ## 5. Simulate MEG Responses
def simulate_meg_responses(audio_path):
    # Load audio to get timing information
    waveform, sample_rate = torchaudio.load(audio_path)
    n_samples = len(waveform[0])
    time = np.linspace(0, n_samples / sample_rate, n_samples)
    
    # Simulate visual response (occipital cortex)
    visual_response = np.random.normal(0, 0.5, n_samples)
    visual_activation = np.exp(-((time - 0.2) ** 2) / (2 * 0.05**2))  # Visual activation at 0.2s
    visual_data = visual_response + visual_activation
    
    # Simulate auditory response (auditory cortex)
    auditory_response = np.random.normal(0, 0.5, n_samples)
    auditory_activation = np.exp(-((time - 0.5) ** 2) / (2 * 0.05**2))  # Auditory activation at 0.5s
    auditory_data = auditory_response + auditory_activation
    
    # Combined response
    combined_data = visual_data + auditory_data
    
    # Plot MEG responses
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time, visual_data)
    plt.title('Simulated Visual MEG Response (Occipital Cortex)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 2)
    plt.plot(time, auditory_data)
    plt.title('Simulated Auditory MEG Response (Auditory Cortex)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(3, 1, 3)
    plt.plot(time, combined_data)
    plt.title('Combined Audio-Visual MEG Response')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'visual': visual_data,
        'auditory': auditory_data,
        'combined': combined_data,
        'time': time
    }

def generate_meg_description(meg_data):
    return f'''
Visual input detected in occipital cortex (sharp spike ~{meg_data['time'][np.argmax(meg_data['visual'])]:.2f}s).
Audio input detected in auditory cortex (broadband activity ~{meg_data['time'][np.argmax(meg_data['auditory'])]:.2f}s).
Combined AV response suggests object recognition and semantic retrieval.
'''

# Simulate and display MEG responses
meg_data = simulate_meg_responses(audio_filename)
meg_description = generate_meg_description(meg_data)

#@markdown ## 6. Generate LLM Interpretation
def generate_llm_response(meg_description):
    # Load model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Create prompt with examples and current MEG data
    prompt = f'''
MEG: Visual spike in occipital cortex followed by auditory burst in STG.
Response: I saw and heard a cat.

MEG: Visual and audio co-activation with frontal synchronization.
Response: I recognized a cat making a sound.

MEG: {meg_description}
Response:'''
    
    # Generate response
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode and extract response
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("Response:")[-1].strip()
    return response

# Generate and display interpretation
print("🧠 Generating LLM interpretation...")
interpretation = generate_llm_response(meg_description)
print("\n💭 LLM Interpretation:", interpretation)

#@markdown ## 7. Interactive Question Answering
def answer_question(question, meg_description, previous_interpretation):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    prompt = f'''
MEG Data: {meg_description}
Previous Interpretation: {previous_interpretation}

Question: {question}
Answer:'''
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Answer:")[-1].strip()
    return answer

# Interactive Q&A loop
print("\nAsk questions about the audio-visual input (type 'quit' to exit):")
while True:
    question = input("\nYour question: ")
    if question.lower() == 'quit':
        break
        
    answer = answer_question(question, meg_description, interpretation)
    print("\n🤖 Answer:", answer) 