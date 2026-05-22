import os
import torch
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback
)
from trl import SFTTrainer
from src.config import settings
from src.models.model_loader import get_tokenizer, load_quantized_model, setup_peft_model

def train():
    # 1. grab data
    dataset = load_dataset("json", data_files={
        "train": "data/processed/train.jsonl",
        "validation": "data/processed/validation.jsonl"
    })
    
    # 2. load model + tokenizer
    tokenizer = get_tokenizer()
    model = load_quantized_model()
    model, peft_config = setup_peft_model(model)
    
    # 3. setup hparams (mostly standard stuff)
    training_args = TrainingArguments(
        output_dir=settings.OUTPUT_DIR,
        per_device_train_batch_size=settings.BATCH_SIZE,
        gradient_accumulation_steps=settings.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=settings.LEARNING_RATE,
        logging_steps=settings.LOGGING_STEPS,
        num_train_epochs=settings.NUM_TRAIN_EPOCHS,
        evaluation_strategy="steps",
        eval_steps=settings.EVAL_STEPS,
        save_strategy="steps",
        save_steps=settings.SAVE_STEPS,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
        push_to_hub=False,
    )
    
    # 4. init trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=settings.MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 5. actually train it
    print("Starting training...")
    trainer.train()
    
    # 6. dump weights to disk
    trainer.save_model(os.path.join(settings.OUTPUT_DIR, "final"))
    print(f"Training complete. Model saved to {settings.OUTPUT_DIR}")

if __name__ == "__main__":
    train()
