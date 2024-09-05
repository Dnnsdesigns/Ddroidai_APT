from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("yourusername/ddroidai-agentic-transformer")
model = AutoModelForCausalLM.from_pretrained("yourusername/ddroidai-agentic-transformer")

def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Enter your text here: "
    print(generate_text(prompt))
