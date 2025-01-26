import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from rag import setup_rag_pipeline
from fine_tuning import fine_tune_model

# Set Hugging Face API token
token = os.getenv("HF_AUTH_TOKEN")
if not token:
    raise ValueError("HF_AUTH_TOKEN is not set. Please set it in the environment variables.")
os.environ["HF_HOME"] = "C:/Users/Lenovo/.cache/huggingface"
login(token)
print("Login successful.")

print("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    low_cpu_mem_usage=True
)

if __name__ == "__main__":
    try:
        fine_tuned_model = fine_tune_model(tokenizer, model, train_new=False)
        if fine_tuned_model is None:
            raise ValueError("Fine-tuning failed.")
        
        question = input("Enter your question: ")  # Dynamic input
        rag_pipeline = setup_rag_pipeline(question=question)
        print(rag_pipeline)
    except Exception as e:
        print(f"An error occurred: {e}")