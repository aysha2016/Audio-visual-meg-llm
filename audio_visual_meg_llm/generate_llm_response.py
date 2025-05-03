
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(prompt):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=50, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded.split("Response:")[-1].strip()
    return response
