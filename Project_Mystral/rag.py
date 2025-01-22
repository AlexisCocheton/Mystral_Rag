from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def setup_rag_pipeline(tokenizer):
    # Load fine-tuned model
    fine_tuned_model = AutoModelForCausalLM.from_pretrained("./mistral-fine-tuned")
    fine_tuned_pipeline = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(pipeline=fine_tuned_pipeline)

    # Check if the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        fine_tuned_model.resize_token_embeddings(len(tokenizer))

    # Load documents and create FAISS index
    documents = [
        {"text": "Python is a programming language.", "metadata": {"source": "doc1"}},
        {"text": "Transformers are deep learning models.", "metadata": {"source": "doc2"}},
    ]
    embeddings = HuggingFaceEmbeddings(model_name="./mistral-fine-tuned")
    faiss_index = FAISS.from_texts([doc["text"] for doc in documents], embeddings)

    # Create a prompt template
    Prompt = PromptTemplate(
        input_variables=["context"],  # Use 'context' as required
        template="Given the context guiven, answer the question: {context}"
    )

    # Create a retriever
    retriever = faiss_index.as_retriever()

    # Create a documents chain
    combine_docs_chain = create_stuff_documents_chain(llm, Prompt)

    # Create the retrieval chain
    qa_chain = create_retrieval_chain(
        combine_docs_chain=combine_docs_chain,
        retriever=retriever
    )

    return qa_chain
