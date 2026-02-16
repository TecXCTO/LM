# The_High-Performance_Data_Pipeline.py

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer # Using HuggingFace's fast BPE logic

class StreamingLLMDataset(IterableDataset):
    """Streams data directly from disk/web to keep memory footprint low."""
    def __init__(self, dataset_path, tokenizer_name="gpt2", max_length=2048):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Ensure we have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_length = max_length
        # In a real scenario, this would be a path to a massive JSONL or Parquet file
        self.data_source = dataset_path 

    def __iter__(self):
        # Simulated streaming logic
        with open(self.data_source, 'r', encoding='utf-8') as f:
            for line in f:
                tokenized = self.tokenizer(
                    line, 
                    truncation=True, 
                    max_length=self.max_length, 
                    padding="max_length",
                    return_tensors="pt"
                )
                # Shift for Causal LM (Input: 0 to N-1, Target: 1 to N)
                input_ids = tokenized['input_ids'].squeeze(0)
                yield input_ids[:-1], input_ids[1:]

def get_dataloader(file_path, batch_size=8):
    dataset = StreamingLLMDataset(file_path)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True)
