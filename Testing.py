import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the trained model and tokenizer
trained_model_path = "/content/drive/MyDrive/llama-finetuned-model/trained_model"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForCausalLM.from_pretrained(trained_model_path)

# Load the LoRA adapter
model = PeftModel.from_pretrained(model, trained_model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("Model and tokenizer loaded successfully!")

# Start interactive loop for real-time testing
print("\nThe model is ready for testing! Type your question below:")
while True:
    try:
        # Get user input
        input_text = input("\nEnter your question (type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if input_text.lower() == 'exit':
            print("Exiting the testing loop. Goodbye!")
            break

        # Tokenize input and generate response
        print("Generating response...")
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            inputs['input_ids'],
            max_length=50,  # Maximum length of generated text
            num_return_sequences=1,
            do_sample=True,  # Enable sampling
            temperature=0.7,  # Adjust randomness
            top_k=50,         # Top-k sampling
            top_p=0.95        # Nucleus sampling
        )

        # Decode and print the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\nResponse:", response)
    except KeyboardInterrupt:
        print("\nExiting the testing loop. Goodbye!")
        break
    except Exception as e:
        print("\nAn error occurred:", str(e))
