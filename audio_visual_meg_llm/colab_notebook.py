#@title 🧠 Audio-Visual MEG + LLM Interpretation System
#@markdown This notebook simulates human brain responses (MEG signals) when processing audio-visual inputs and generates natural language interpretations using GPT-2.

#@markdown ## 1. Install Required Packages
!pip install -r requirements.txt

#@markdown ## 2. Import All Components
from input_processor import InputProcessor
from meg_simulator import MEGSimulator
from llm_processor import LLMProcessor
from google.colab import files
import os

#@markdown ## 3. Initialize System
# Initialize all components
input_processor = InputProcessor()
meg_simulator = MEGSimulator()
llm_processor = LLMProcessor()

#@markdown ## 4. Upload and Process Files
# Upload image file
print("Upload an image file (JPG/PNG):")
uploaded_image = files.upload()
image_filename = list(uploaded_image.keys())[0]

# Upload audio file
print("\nUpload an audio file (WAV/MP3):")
uploaded_audio = files.upload()
audio_filename = list(uploaded_audio.keys())[0]

# Process inputs
if not input_processor.load_image(image_filename):
    print("Failed to load image. Exiting...")
    raise SystemExit

if not input_processor.load_audio(audio_filename):
    print("Failed to load audio. Exiting...")
    raise SystemExit

#@markdown ## 5. Display Inputs
print("\nDisplaying inputs...")
input_processor.display_inputs()

#@markdown ## 6. Simulate MEG Responses
print("\nSimulating MEG responses...")
audio_info = input_processor.get_audio_info()
meg_data = meg_simulator.simulate_responses(audio_info)
meg_simulator.plot_responses()

#@markdown ## 7. Generate Interpretations
# Generate MEG description
meg_description = meg_simulator.generate_description()
print("\nMEG Description:", meg_description)

# Generate LLM interpretation
print("\nGenerating LLM interpretation...")
interpretation = llm_processor.generate_interpretation(meg_description)
print("\n💭 LLM Interpretation:", interpretation)

#@markdown ## 8. Interactive Q&A
print("\nAsk questions about the audio-visual input (type 'quit' to exit):")
while True:
    question = input("\nYour question: ")
    if question.lower() == 'quit':
        break

    answer = llm_processor.answer_question(question, meg_description, interpretation)
    print("\n🤖 Answer:", answer)

#@markdown ## 9. Cleanup
# Clean up uploaded files
try:
    os.remove(image_filename)
    os.remove(audio_filename)
except:
    pass 