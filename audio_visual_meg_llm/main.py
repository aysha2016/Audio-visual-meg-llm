
from audio_visual_input import display_inputs
from simulate_meg import simulate_meg_description
from generate_llm_response import generate_response

image_path = "cat.jpg"
audio_path = "cat-meow.wav"

display_inputs(image_path, audio_path)

meg_description = simulate_meg_description()

prompt = f'''
MEG: Visual spike in occipital cortex followed by auditory burst in STG.
Response: I saw and heard a cat.

MEG: Visual and audio co-activation with frontal synchronization.
Response: I recognized a cat making a sound.

MEG: {meg_description}
Response:'''

response = generate_response(prompt)
print("🧠 LLM Interpretation:", response)
