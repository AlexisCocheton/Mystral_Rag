import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login


# PART 1: Fine-Tuning the Model
def fine_tune_model():
    # Load a sample dataset
    dataset = load_dataset("squad", split="train[:1%]")  # Use a small subset for testing
    dataset = dataset.map(
        lambda x: {"input_text": x["context"], "target_text": x["answers"]["text"][0]},
        remove_columns=dataset.column_names,
    )

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"], text_target=examples["target_text"], truncation=True
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./mistral-fine-tuned",
        evaluation_strategy="steps",
        logging_steps=10,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=10,
        save_total_limit=1,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()
    trainer.save_model("./mistral-fine-tuned")
    print("Model fine-tuned and saved to './mistral-fine-tuned'")