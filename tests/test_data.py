import pytest
import os
import json
import shutil
from src.data.preprocess import clean_data, format_prompt, load_raw_data

def test_clean_data():
    sample_data = [
        {"instruction": "test 1", "response": "resp 1"},
        {"instruction": "test 1", "response": "resp 1"}, # Duplicate
        {"instruction": "test 2", "response": "resp 2"}
    ]
    cleaned = clean_data(sample_data)
    assert len(cleaned) == 2
    assert cleaned[0]["instruction"] == "test 1"
    assert cleaned[1]["instruction"] == "test 2"

def test_format_prompt():
    sample = {"instruction": "Hello", "response": "Hi there"}
    formatted = format_prompt(sample)
    assert "text" in formatted
    assert "<system>" in formatted["text"]
    assert "<instruction> Hello </instruction>" in formatted["text"]
    assert "<response> Hi there </response>" in formatted["text"]

def test_load_raw_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    p = d / "test.jsonl"
    content = {"instruction": "q1", "response": "a1"}
    p.write_text(json.dumps(content) + "\n")
    
    loaded = load_raw_data(str(p))
    assert len(loaded) == 1
    assert loaded[0]["instruction"] == "q1"
