import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from huggingface_hub import login
from rag import setup_rag_pipeline
from fine_tuning import fine_tune_model

# Set Hugging Face API token
token = os.getenv("HF_AUTH_TOKEN")
print(f"User Query: {token}")
os.environ["HF_HOME"] = "C:/Users/Lenovo/.cache/huggingface"
login(token)
print(f"Login successful.")

model_name = "gpt2"  # Replace with mistralai/mistral-7b-v0.1 if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Tokenizer loaded.")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True
)
print(f"Model loaded.")

if __name__ == "__main__":
    # Fine-tune the model
    fine_tuned_model = fine_tune_model(tokenizer, model)
     
    # Set up RAG pipeline
    rag_pipeline = setup_rag_pipeline(tokenizer)

    # Test the RAG pipeline
    user_query = "What is Python?"
    result = rag_pipeline.invoke({"input": user_query})  # Pass as a dict with the expected key
    print(f"User Query: {user_query}")
    print(f"Generated Answer: {result}")

