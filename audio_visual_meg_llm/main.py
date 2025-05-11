from input_processor import InputProcessor
from meg_simulator import MEGSimulator
from llm_processor import LLMProcessor
from google.colab import files
import os

def main():
    # Initialize components
    input_processor = InputProcessor()
    meg_simulator = MEGSimulator()
    llm_processor = LLMProcessor()

    # Upload files
    print("Upload an image file (JPG/PNG):")
    uploaded_image = files.upload()
    image_filename = list(uploaded_image.keys())[0]

    print("\nUpload an audio file (WAV/MP3):")
    uploaded_audio = files.upload()
    audio_filename = list(uploaded_audio.keys())[0]

    # Process inputs
    if not input_processor.load_image(image_filename):
        print("Failed to load image. Exiting...")
        return

    if not input_processor.load_audio(audio_filename):
        print("Failed to load audio. Exiting...")
        return

    # Display inputs
    print("\nDisplaying inputs...")
    input_processor.display_inputs()

    # Simulate MEG responses
    print("\nSimulating MEG responses...")
    audio_info = input_processor.get_audio_info()
    meg_data = meg_simulator.simulate_responses(audio_info)
    meg_simulator.plot_responses()

    # Generate MEG description
    meg_description = meg_simulator.generate_description()
    print("\nMEG Description:", meg_description)

    # Generate LLM interpretation
    print("\nGenerating LLM interpretation...")
    interpretation = llm_processor.generate_interpretation(meg_description)
    print("\n💭 LLM Interpretation:", interpretation)

    # Interactive Q&A
    print("\nAsk questions about the audio-visual input (type 'quit' to exit):")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'quit':
            break

        answer = llm_processor.answer_question(question, meg_description, interpretation)
        print("\n🤖 Answer:", answer)

    # Cleanup
    try:
        os.remove(image_filename)
        os.remove(audio_filename)
    except:
        pass

if __name__ == "__main__":
    main()
