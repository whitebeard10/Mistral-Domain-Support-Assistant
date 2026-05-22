import os
import json
import torch
from src.inference.engine import InferenceEngine
from src.evaluation.evaluator import Evaluator

def check_env():
    print("--- Environment Check ---")
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("❌ No GPU detected. Mistral-7B will likely crash or be extremely slow.")
    print("-------------------------\n")

def run_sample_inference():
    print("--- Running Sample Inference ---")
    engine = InferenceEngine(adapter_path=None) # Zero-shot by default
    
    test_prompts = [
        "How do I reset my password?",
        "My account is locked, what should I do?",
        "How can I update my billing info?"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = engine.generate(prompt)
        print(f"Response: {result['response']}")
        print(f"Latency: {result['latency']:.2f}s")
    print("\n-------------------------------\n")

def run_colab_evaluation(num_samples=20):
    print(f"--- Running Evaluation ({num_samples} samples) ---")
    evaluator = Evaluator(adapter_path=None)
    test_data_path = "data/processed/validation.jsonl"
    
    if not os.path.exists(test_data_path):
        print(f"❌ Validation data not found at {test_data_path}. Run preprocessing first.")
        return

    results, rouge, bleu = evaluator.run_comparison(test_data_path, num_samples=num_samples)
    
    report = {
        "metrics": {
            "rouge": rouge,
            "bleu": bleu
        },
        "samples": results
    }
    
    output_path = "colab_evaluation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n✅ Evaluation complete. Report saved to: {output_path}")
    print("------------------------------------------\n")

if __name__ == "__main__":
    check_env()
    
    print("What would you like to do in Colab?")
    print("1. Run sample inference (3 prompts)")
    print("2. Run full evaluation and save JSON")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ")
    
    if choice in ['1', '3']:
        run_sample_inference()
    if choice in ['2', '3']:
        run_colab_evaluation(num_samples=20)
