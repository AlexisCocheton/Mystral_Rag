import os
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

model_name = "gpt2"  # Use GPT-2 or a fine-tuned version
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)

if __name__ == "__main__":
    # Fine-tune the model (if needed)
    fine_tuned_model = fine_tune_model(tokenizer, model)
    
    # Set up the RAG pipeline
    rag_pipeline = setup_rag_pipeline(tokenizer)

    # Test the RAG pipeline
    user_query = "tell me the documents you got in the RAG pipeline"
    cleaned_answer = rag_pipeline(user_query)

    print(f"Question: {user_query}")
    print(f"Cleaned Answer: {cleaned_answer}")
