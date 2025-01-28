import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,pipeline
from huggingface_hub import login
from rag import setup_rag_pipeline
from fine_tuning import fine_tune_model
from peft import PeftModel

# Set Hugging Face API token
offload_folder = "freespace"
os.makedirs(offload_folder, exist_ok=True)

token = os.getenv("HF_AUTH_TOKEN")
if not token:
    raise ValueError("HF_AUTH_TOKEN is not set. Please set it in the environment variables.")
os.environ["HF_HOME"] = "C:/Users/Lenovo/.cache/huggingface"
login(token)
print("Login successful.")

print("cuda" if torch.cuda.is_available() else "cpu")

# Load Mistral-7B with 4-bit quantization
model_name = "mistralai/Mistral-7B-v0.1"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="balanced_low_0",  # Better memory management
    offload_folder="freespace"
)

if __name__ == "__main__":
    try:
        # Fine-tune the model
        fine_tuned_model = fine_tune_model(tokenizer, model, train_new=False)
        if fine_tuned_model is None:
            raise ValueError("Fine-tuning failed.")
        
        # Load the fine-tuned adapter
        model = PeftModel.from_pretrained(model, "mistral-7b-fine-tuned", is_trainable=False)
        # ========== CORRECTED MODEL TEST SECTION ==========
        print("\n=== Testing fine-tuned model ===")
        test_prompt = input("Enter a test question: ")

# Merge and unload properly
        model = model.merge_and_unload()

# Create pipeline WITHOUT device specification
        test_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.2
)

# Generate response
        try:
            raw_response = test_pipe(test_prompt)[0]['generated_text']
        # Clean response display
            response = raw_response.replace(test_prompt, "").strip()
            print("\nModel response:", response.split("\n")[0])
        except Exception as e:
            print(f"Generation error: {e}")
# ========== END OF SECTION ==========        
        # Proceed to RAG test
        print("\n=== Now testing RAG pipeline ===")
        rag_question = input("Enter your RAG question: ")
        rag_pipeline = setup_rag_pipeline(
            question=rag_question,
            model=model,  # <-- Pass merged model
            tokenizer=tokenizer  # <-- Pass existing tokenizer
        )
        print("\nRAG response:")
        print(rag_pipeline)
        
    except Exception as e:
        print(f"An error occurred: {e}")