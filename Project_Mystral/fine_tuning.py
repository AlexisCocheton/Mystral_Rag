import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from huggingface_hub import login
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

def fine_tune_model(tokenizer, model):
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    # Load dataset
    dataset = load_dataset("squad", split="train[:1%]")
    eval_dataset = load_dataset("squad", split="validation[:1%]")

    # Tokenize the dataset
    def tokenize_function(examples):
        # Add the labels (input_ids are the same as labels for causal LM)
        tokenized = tokenizer(examples["context"], truncation=True, padding="max_length", max_length=512)
        tokenized["labels"] = tokenized["input_ids"].copy()  # Copy input_ids to labels
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-model",
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        save_steps=10,
        save_total_limit=1,
        push_to_hub=False,  # Disable auto-pushing to Hugging Face Hub
    )

    # Trainer automatically handles device management
    trainer = Trainer(
        model=model,  # No need to manually handle devices here
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,  # Make sure tokenizer is passed
    )

    # Fine-tune the model
    trainer.train()
    trainer.save_model("./mistral-fine-tuned")
    print("Model fine-tuned and saved!")
