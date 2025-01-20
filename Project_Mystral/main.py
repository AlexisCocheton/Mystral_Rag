import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login
from fine_tuning import fine_tune_model
from rag import setup_rag_pipeline


# Set Hugging Face API token
token = os.getenv("HF_AUTH_TOKEN")
os.environ["HF_HOME"] = "C:/Users/Lenovo/.cache/huggingface"
login(token)

model_name = "mistralai/mistral-7b-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# MAIN FUNCTION
if __name__ == "__main__":
    # Fine-tune the model
    fine_tune_model()

    # Set up RAG pipeline
    rag_pipeline = setup_rag_pipeline()

    # Test the RAG pipeline
    user_query = "What is Python?"
    result = rag_pipeline.run(query=user_query)
    print(f"User Query: {user_query}")
    print(f"Generated Answer: {result}")
