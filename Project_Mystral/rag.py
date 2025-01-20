import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login


# PART 2: Implementing Retrieval-Augmented Generation (RAG)
def setup_rag_pipeline():
    # Load fine-tuned model
    fine_tuned_model = AutoModelForCausalLM.from_pretrained("./mistral-fine-tuned")
    fine_tuned_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=fine_tuned_pipeline)

    # Load documents and create FAISS index
    documents = [
        {"text": "Python is a programming language.", "metadata": {"source": "doc1"}},
        {"text": "Transformers are deep learning models.", "metadata": {"source": "doc2"}},
    ]
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings)

    # Create a retrieval chain
    retriever = faiss_index.as_retriever()
    qa_chain = RetrievalQA(
        retriever=retriever,
        llm=llm,
        prompt=PromptTemplate(input_variables=["context", "query"], template="{context}\n\nQ: {query}\nA:"),
    )
    return qa_chain