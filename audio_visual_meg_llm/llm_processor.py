from transformers import GPT2LMHeadModel, GPT2Tokenizer

class LLMProcessor:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.generation_config = {
            'max_new_tokens': 50,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': None  # Will be set to eos_token_id
        }
        self.generation_config['pad_token_id'] = self.tokenizer.eos_token_id

    def generate_interpretation(self, meg_description):
        """Generate an interpretation of the MEG data."""
        prompt = f'''
MEG: Visual spike in occipital cortex followed by auditory burst in STG.
Response: I saw and heard a cat.

MEG: Visual and audio co-activation with frontal synchronization.
Response: I recognized a cat making a sound.

MEG: {meg_description}
Response:'''

        return self._generate_response(prompt)

    def answer_question(self, question, meg_description, previous_interpretation):
        """Generate an answer to a question about the input."""
        prompt = f'''
MEG Data: {meg_description}
Previous Interpretation: {previous_interpretation}

Question: {question}
Answer:'''

        # Use longer response for questions
        self.generation_config['max_new_tokens'] = 100
        response = self._generate_response(prompt)
        # Reset to default
        self.generation_config['max_new_tokens'] = 50
        return response

    def _generate_response(self, prompt):
        """Internal method to generate responses from the model."""
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
            output = self.model.generate(
                input_ids,
                **self.generation_config
            )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract the response part
            if "Response:" in decoded:
                response = decoded.split("Response:")[-1].strip()
            elif "Answer:" in decoded:
                response = decoded.split("Answer:")[-1].strip()
            else:
                response = decoded.strip()
                
            return response
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating the response." 