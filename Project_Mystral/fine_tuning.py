import os
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def fine_tune_model(tokenizer, model):
    # Path to save/load the fine-tuned model
    fine_tuned_model_path = "./gpt-fine-tuned"
    

    # Check if the fine-tuned model already exists
    if os.path.exists(fine_tuned_model_path):
        print(f"Fine-tuned model found at {fine_tuned_model_path}. Loading the model...")
        return AutoModelForCausalLM.from_pretrained(fine_tuned_model_path)
    
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
    trainer.save_model(fine_tuned_model_path)
    print("Model fine-tuned and saved!")
