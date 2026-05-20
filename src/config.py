import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Model Configuration
    BASE_MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    DEVICE: str = "cuda"
    
    # QLoRA Configuration
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    TARGET_MODULES: list[str] = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    # Training Configuration
    OUTPUT_DIR: str = "./models/mistral-domain-support-adapter"
    BATCH_SIZE: int = 4
    GRADIENT_ACCUMULATION_STEPS: int = 4
    LEARNING_RATE: float = 2e-4
    NUM_TRAIN_EPOCHS: int = 3
    LOGGING_STEPS: int = 10
    EVAL_STEPS: int = 50
    SAVE_STEPS: int = 50
    MAX_SEQ_LENGTH: int = 1024
    
    # Dataset Configuration
    DATASET_PATH: str = "data/processed/support_dataset.jsonl"
    TRAIN_TEST_SPLIT: float = 0.1
    
    # Inference Configuration
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    class Config:
        env_file = ".env"

settings = Settings()
