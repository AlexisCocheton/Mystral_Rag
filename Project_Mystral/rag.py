from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch
import os

def setup_rag_pipeline(question, model, tokenizer, max_new_tokens=200):
    """
    Sets up and runs the RAG pipeline using a pre-loaded model and tokenizer.
    
    Args:
        question (str): The question to answer.
        model: Pre-loaded and fine-tuned model.
        tokenizer: Pre-loaded tokenizer.
        max_new_tokens (int): Maximum tokens to generate.
    
    Returns:
        str: The generated answer or None if an error occurs.
    """
    try:
        # Create generation pipeline with the provided model and tokenizer
        generation_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # Lower temperature for more focused answers
            do_sample=True,  # Enable sampling for diversity
            repetition_penalty=1.2,  # Reduce repetition
            # REMOVED: device=0 if torch.cuda.is_available() else -1  # Let accelerate handle device placement
        )
        
        # Wrap the pipeline in LangChain's HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=generation_pipe)

        # Load documents (replace with your document loading logic)
        with open("documents.txt", "r") as f:
            documents = [{"text": line.strip(), "metadata": {"source": f"doc{i}"}} for i, line in enumerate(f)]
        
        # Create embeddings using GPU if available
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        # Create FAISS index from documents
        texts = [doc["text"] for doc in documents]
        faiss_index = FAISS.from_texts(texts, embeddings)

        # Define the system prompt for RAG
        system_prompt = (
            "You are an AI assistant. Use the following context to answer the question. "
            "If the context doesn't contain the answer, say 'I don't know.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])

        # Create retriever to fetch relevant documents
        retriever = faiss_index.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 documents

        # Compose the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Invoke the RAG pipeline with the question
        response = rag_chain.invoke({"input": question})
        return response["answer"]

    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return None