from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
    Set up a RAG pipeline for question answering.
    Args:
        question: The question to answer.
        model_path: The path to the fine-tuned model.
        max_new_tokens: The maximum number of tokens to generate.
    returns:
        The generated text response.
    """

def setup_rag_pipeline(question, model_path="GPT-wiki-fine-tuned", max_new_tokens=200):
    try:
        # Load fine-tuned model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        fine_tuned_pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=tokenizer,
            device=0,
            max_new_tokens=max_new_tokens,
            temperature=0.3,  # Lower temperature for more focused answers
            repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=fine_tuned_pipeline)

        # Load documents dynamically (e.g., from a file)
        with open("documents.txt", "r") as f:
            documents = [{"text": line.strip(), "metadata": {"source": f"doc{i}"}} for i, line in enumerate(f)]
        
        # Create FAISS index
        embeddings_model = "sentence-transformers/all-mpnet-base-v2"  # Better embeddings
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        texts = [doc["text"] for doc in documents]
        faiss_index = FAISS.from_texts(texts, embeddings)

        # Define the system prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
            ]
        )

        # Create retriever
        retriever = faiss_index.as_retriever(search_kwargs={"k": 5})  # Fetch top 5 documents

        # Compose the chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Debug: Print retrieved documents
        retrieved_docs = retriever.invoke(question)
        print("Retrieved Documents:", retrieved_docs)

        # Extract page_content from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Debug: Print input to model
        input_to_model = prompt.format(input=question, context=context)
        print("Input to Model:", input_to_model)

        # Invoke the pipeline
        response = rag_chain.invoke({"input": question})
        print("RAG Pipeline Response:", response)  # Debug: Print full response
        generated_text = response["answer"]
        
        return generated_text
    except Exception as e:
        print(f"Error setting up RAG pipeline: {e}")
        return None

s=setup_rag_pipeline("What is python?")
