import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.config import settings

class InferenceEngine:
    def __init__(self, adapter_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 4-bit quant to save vram
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            settings.BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if adapter_path:
            print(f"Loading adapters from {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            print("Using base model for zero-shot inference")
            self.model = base_model
            
        self.model.eval()

    def generate(self, instruction: str, max_new_tokens: int = 512):
        system_prompt = "You are a helpful domain-specific support assistant."
        prompt = f"<system> {system_prompt} </system> <instruction> {instruction} </instruction> <response>"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=settings.TEMPERATURE,
                top_p=settings.TOP_P,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        end_time = time.time()
        
        new_tokens = output_tokens[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {
            "response": response.strip(),
            "latency": end_time - start_time,
            "tokens_generated": len(new_tokens)
        }

if __name__ == "__main__":
    # quick zero-shot sanity check
    engine = InferenceEngine()
    result = engine.generate("How do I reset my password?")
    print(f"Response: {result['response']}")
    print(f"Latency: {result['latency']:.2f}s")
