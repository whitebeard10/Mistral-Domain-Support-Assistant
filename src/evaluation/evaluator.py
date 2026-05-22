import torch
import json
import pandas as pd
from evaluate import load
from tqdm import tqdm
from src.inference.engine import InferenceEngine
from src.config import settings

class Evaluator:
    def __init__(self, adapter_path: str = None):
        self.engine = InferenceEngine(adapter_path=adapter_path)
        self.rouge = load("rouge")
        self.bleu = load("bleu")

    def evaluate_dataset(self, test_data_path: str, num_samples: int = 50):
        """Evaluates the model on a test dataset."""
        test_samples = []
        with open(test_data_path, 'r') as f:
            for line in f:
                test_samples.append(json.loads(line))
        
        if num_samples < len(test_samples):
            test_samples = test_samples[:num_samples]
            
        predictions = []
        references = []
        
        print(f"Running evaluation on {len(test_samples)} samples...")
        for sample in tqdm(test_samples):
            # handle both raw and processed formats
            if 'instruction' in sample:
                instr = sample['instruction']
                ref = sample['response']
            elif 'text' in sample:
                # hacky regex to pull out the prompt parts
                import re
                instr_match = re.search(r'<instruction> (.*?) </instruction>', sample['text'])
                ref_match = re.search(r'<response> (.*?) </response>', sample['text'])
                instr = instr_match.group(1) if instr_match else ""
                ref = ref_match.group(1) if ref_match else ""
            else:
                continue

            pred_data = self.engine.generate(instr)
            predictions.append(pred_data['response'])
            references.append(ref)

        # calculate scores
        rouge_results = self.rouge.compute(predictions=predictions, references=references)
        # bleu needs nested lists for some reason
        bleu_results = self.bleu.compute(predictions=predictions, references=[[r] for r in references])
        
        print("\nEvaluation Results:")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"BLEU: {bleu_results['bleu']:.4f}")
        
        return {
            "rouge": rouge_results,
            "bleu": bleu_results,
            "num_samples": len(predictions)
        }

    def run_comparison(self, raw_data_path: str, num_samples: int = 20):
        """Compares zero-shot vs fine-tuned (if adapter exists)."""
        data = []
        with open(raw_data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        samples = data[:num_samples]
        results = []
        
        for sample in tqdm(samples):
            instr = sample['instruction']
            ref = sample['response']
            
            pred_data = self.engine.generate(instr)
            pred = pred_data['response']
            
            results.append({
                "instruction": instr,
                "reference": ref,
                "prediction": pred,
                "latency": pred_data['latency']
            })
            
        # Compute metrics
        preds = [r['prediction'] for r in results]
        refs = [[r['reference']] for r in results]
        
        rouge_results = self.rouge.compute(predictions=preds, references=[r['reference'] for r in results])
        bleu_results = self.bleu.compute(predictions=preds, references=refs)
        
        print("\nEvaluation Results:")
        print(f"ROUGE-L: {rouge_results['rougeL']:.4f}")
        print(f"BLEU: {bleu_results['bleu']:.4f}")
        
        return results, rouge_results, bleu_results

if __name__ == "__main__":
    # quick local test
    evaluator = Evaluator(adapter_path=None)
    evaluator.run_comparison("data/raw/raw_support_data.jsonl", num_samples=5)
