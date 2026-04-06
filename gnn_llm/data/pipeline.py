import os
import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

class GigaDataPipeline:
    """Unified Data Pipeline (v10.0). 80/20 FineWeb-Edu + Logic stream."""
    def __init__(self, tokenizer_name="meta-llama/Meta-Llama-3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, token=os.getenv('HF_TOKEN', ''), use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.ds_knowledge = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        self.ds_logic = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
        self.dataset = interleave_datasets(
            [self.ds_knowledge, self.ds_logic],
            probabilities=[0.8, 0.2], seed=42, stopping_strategy="all_exhausted"
        )

    def get_dataloader(self, batch_size=4, seq_len=1024, skip_steps=0):
        dataset = self.dataset.skip(skip_steps * batch_size) if skip_steps > 0 else self.dataset
        buffer = []
        for example in dataset:
            tokens = self.tokenizer(example["text"], truncation=False)["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= (batch_size * seq_len):
                yield torch.tensor(buffer[:batch_size * seq_len]).view(batch_size, seq_len)
                buffer = buffer[batch_size * seq_len:]
