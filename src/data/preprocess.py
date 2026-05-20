import os
import json
from typing import List, Dict
from src.config import settings

def load_raw_data(file_path: str) -> List[Dict]:
    """Loads raw JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def clean_data(data: List[Dict]) -> List[Dict]:
    """Basic data cleaning and duplicate removal."""
    import pandas as pd
    df = pd.DataFrame(data)
    initial_len = len(df)
    df = df.drop_duplicates(subset=['instruction', 'response'])
    print(f"Removed {initial_len - len(df)} duplicates.")
    return df.to_dict('records')

def format_prompt(sample: Dict) -> str:
    """Formats sample into a structured instruction format."""
    # User requested format: <system> <instruction> <response>
    system_prompt = "You are a helpful domain-specific support assistant."
    instruction = sample['instruction']
    response = sample['response']
    
    prompt = f"<system> {system_prompt} </system> <instruction> {instruction} </instruction> <response> {response} </response>"
    return {"text": prompt}

def preprocess_pipeline(input_path: str, output_dir: str):
    """Full preprocessing pipeline."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load
    raw_data = load_raw_data(input_path)
    
    # 2. Clean
    cleaned_data = clean_data(raw_data)
    
    # 3. Format and Split
    df = pd.DataFrame(cleaned_data)
    train_df, val_df = train_test_split(df, test_size=settings.TRAIN_TEST_SPLIT, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Apply formatting
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_prompt, remove_columns=val_dataset.column_names)
    
    # Save processed data
    train_dataset.to_json(os.path.join(output_dir, "train.jsonl"))
    val_dataset.to_json(os.path.join(output_dir, "validation.jsonl"))
    
    print(f"Preprocessed {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    preprocess_pipeline("data/raw/raw_support_data.jsonl", "data/processed")
