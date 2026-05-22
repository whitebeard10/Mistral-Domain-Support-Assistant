import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.config import settings

def get_tokenizer(model_name: str = settings.BASE_MODEL_NAME):
    """Loads and configures the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # gotta use right padding for llama models
    return tokenizer

def load_quantized_model(model_name: str = settings.BASE_MODEL_NAME):
    """Loads the base model with 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # prep for kbit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    return model

def setup_peft_model(model):
    """Configures and wraps the model with LoRA adapters."""
    peft_config = LoraConfig(
        r=settings.LORA_R,
        lora_alpha=settings.LORA_ALPHA,
        target_modules=settings.TARGET_MODULES,
        lora_dropout=settings.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, peft_config
