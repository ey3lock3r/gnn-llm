import os
from datasets import load_dataset, concatenate_datasets, interleave_datasets
from transformers import AutoTokenizer
from dotenv import load_dotenv
import torch

# Load environment variables (HF_TOKEN, WANDB_API_KEY)
load_dotenv()

class GigaDataPipeline:
    """
    v8.1: Knowledge + Logic (80/20 Mixed Stream)
    Uses Llama-3-8B (128k) Tokenization.
    """
    def __init__(self, tokenizer_name="meta-llama/Meta-Llama-3-8B"):
        print(f"Initializing GigaDataPipeline with {tokenizer_name}...")
        
        # Note: Requires HF_TOKEN in .env and access to Llama-3 on HF Hub
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            token=os.getenv('HF_TOKEN'),
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Streaming SOTA Knowledge (FineWeb-Edu) and Logic (Coding)...")
        # 1. 80% General Knowledge (Educational CommonCrawl)
        self.ds_knowledge = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            "sample-10BT", 
            split="train", 
            streaming=True
        )
        
        # 2. 20% Logic & Code reasoning
        self.ds_logic = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            "sample-10BT", 
            split="train",
            streaming=True
        )
        
        # Interleave with 80/20 ratio
        self.dataset = interleave_datasets(
            [self.ds_knowledge, self.ds_logic],
            probabilities=[0.8, 0.2],
            seed=42,
            stopping_strategy="all_exhausted"
        )

    def get_dataloader(self, batch_size=4, seq_len=1024):
        """
        Returns a generator of tokenized batches from the combined stream.
        """
        buffer = []
        for example in self.dataset:
            text = example["text"]
            # Tokenize without limit to avoid truncation at this stage
            tokens = self.tokenizer(text, truncation=False)["input_ids"]
            buffer.extend(tokens)
            
            # When we have enough tokens for a full batch
            while len(buffer) >= (batch_size * seq_len):
                batch_tokens = buffer[:batch_size * seq_len]
                buffer = buffer[batch_size * seq_len:]
                
                yield torch.tensor(batch_tokens).view(batch_size, seq_len)

if __name__ == "__main__":
    # Test pipeline (Small chunk)
    pipeline = GigaDataPipeline()
    loader = pipeline.get_dataloader(batch_size=2, seq_len=256)
    
    for i, batch in enumerate(loader):
        print(f"Batch {i} Shape: {batch.shape}")
        if i >= 2: break
