import os
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

"""
    Fine-tune a causal language model using LoRA or load an existing one.
    Args:
        tokenizer: Pretrained tokenizer compatible with the model.
        model: Pretrained causal language model to fine-tune.
        dataset_name: The dataset path or identifier (default: "wikitext").
        dataset_config: Specific configuration of the dataset (default: "wikitext-2-raw-v1").
        save_path: Path to save the fine-tuned model.
        train_new: If True, train a new model; otherwise, load the existing fine-tuned model if it exists.
        dataset_size: The size of the dataset to use for training (default: "train[:10%]").
        eval_size: The size of the dataset to use for evaluation (default: "validation[:10%]").
"""

def fine_tune_model(tokenizer, model, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", save_path="mistral-7b-fine-tuned", train_new=False):
    """
    Fine-tune Mistral-7B with LoRA for speed (not accuracy).
    """
    # Check if the fine-tuned model already exists
    if not train_new and os.path.exists(save_path):
        print(f"Fine-tuned model found at {save_path}. Loading the model...")
        return model

    # Ensure pad_token is set
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Load a tiny subset of the dataset
    try:
        dataset = load_dataset(dataset_name, dataset_config, split="train[:1%]")  # Use only 1% of the dataset
        eval_dataset = load_dataset(dataset_name, dataset_config, split="validation[:1%]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Tokenize the dataset with a shorter sequence length
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=128  # Short sequence length
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    # Add LoRA configuration with a very small rank
    lora_config = LoraConfig(
        r=2,  # Very small rank for speed
        lora_alpha=4,  # 2 * r
        target_modules=["q_proj", "v_proj"],  # Mistral-7B attention layers
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Debug: Verify LoRA setup

    # Training arguments optimized for speed
    training_args = TrainingArguments(
        output_dir=save_path,  # Directory to save the model
        per_device_train_batch_size=1,  # Small batch size for speed
        per_device_eval_batch_size=1,  # Same as train batch size
        gradient_accumulation_steps=1,  # No gradient accumulation for speed
        max_steps=100,  # Train for only 100 steps (fraction of an epoch)
        eval_strategy="no",  # Disable evaluation
        logging_steps=10,  # Log metrics every 10 steps
        learning_rate=5e-5,  # Learning rate for LoRA
        weight_decay=0.01,  # Regularization to prevent overfitting
        warmup_steps=10,  # Short warmup for learning rate scheduler
        save_steps=100,  # Save checkpoint every 100 steps
        save_total_limit=1,  # Keep only 1 checkpoint
        fp16=True,  # Use mixed precision (FP16)
        push_to_hub=False,  # Set to True if pushing to Hugging Face Hub
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
    )

    # Fine-tune the model
    try:
        print("Starting model fine-tuning...")
        trainer.train()
        trainer.model.save_pretrained(save_path)  # Save only the adapter
        print(f"Model fine-tuned and saved to {save_path}!")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None