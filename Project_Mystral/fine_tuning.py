import os
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset


def fine_tune_model(tokenizer, model, dataset_name="wikitext", dataset_config="wikitext-2-raw-v1", save_path="GPT-wiki-fine-tuned", train_new=False, dataset_size="train[:10%]", eval_size="validation[:10%]"):
    """
    Fine-tune a causal language model or load an existing one.
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
    test_context = """
    Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.

    Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It is dynamically typed and garbage-collected.

    Python's syntax is designed to be easy to read and write, making it a popular choice for beginners and experienced developers alike.

    Python is widely used in data science, machine learning, web development, automation, and scientific computing. Libraries like NumPy, Pandas, and TensorFlow are commonly used in these fields.

    Python has a large standard library that provides modules and functions for tasks like file I/O, regular expressions, and web development.
    """

    test_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question: What is Python?
    Answer:
    """.format(context=test_context)

    # Generate answer
    test_output = fine_tuned_pipeline(test_prompt)
    print("Test Output:", test_output)
    
    # Check if the fine-tuned model already exists
    if not train_new and os.path.exists(save_path):
        print(f"Fine-tuned model found at {save_path}. Loading the model...")
        return AutoModelForCausalLM.from_pretrained(save_path)

    # Ensure pad_token is set
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # Load dataset
    try:
        # Load dataset with configurable size
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_size)
        eval_dataset = load_dataset(dataset_name, dataset_config, split=eval_size)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=save_path,
        eval_strategy="steps", #FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
        eval_steps=100,
        logging_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        push_to_hub=False,
    )


    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        processing_class=tokenizer, 
    )

    # Fine-tune the model
    try:
        print("Starting model fine-tuning...")
        trainer.train()
        trainer.save_model(save_path)
        print(f"Model fine-tuned and saved to {save_path}!")
        return model
    except Exception as e:
        print(f"Error during training: {e}")
        return None
