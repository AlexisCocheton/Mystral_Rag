from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def setup_rag_pipeline(tokenizer):
    # Load fine-tuned GPT-2 model for generation
    fine_tuned_pipeline = pipeline(
        "text-generation",
        model="./gpt-fine-tuned",
        tokenizer=tokenizer,
        device=0,  # Use GPU if available
        max_new_tokens=50,
        temperature=0.7,
        repetition_penalty=1.2
    )
    llm = HuggingFacePipeline(pipeline=fine_tuned_pipeline)

    # Load pre-trained sentence-transformers model for embeddings
    embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

    # Create FAISS index
    documents = [
        {"text": "Python is a programming language.", "metadata": {"source": "doc1"}},
        {"text": "Transformers are deep learning models.", "metadata": {"source": "doc2"}},
    ]
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings)

    # Define the system prompt
    system_prompt = (
        "You are an AI assistant. Use the retrieved context to answer the question. "
        "Keep your answer concise, clear, and accurate. End your response with '<|endofanswer|>'. "
        "If the context does not have enough information, say 'I don't know.'\n\n{context}"
    )

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=system_prompt
    )
    
    # Create a retriever
    retriever = faiss_index.as_retriever()

    # Create a documents chain to combine retrieved documents
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever
    )

    # Function to run the pipeline and clean the response
    def clean_output(question):
    
        response = qa_chain.invoke({"input": question})

        # Extract the generated text from the response dictionary
        generated_text = response.get("answer", "")  # Use the correct key if "answer" is not available
        return generated_text.split("<|endofanswer|>")[-1].strip()



    return clean_output
