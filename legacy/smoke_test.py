import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from aptp_gnn import GigaGraph_3B
from data_pipeline import GigaDataPipeline
from tqdm import tqdm

def run_giga_smoke_test():
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if hf_token: login(token=hf_token)
        
    print("🚀 Initializing GigaGraph 3.2B Smoke Test (v8.1)...")
    
    # 2. Pipeline (Small subset)
    pipeline = GigaDataPipeline()
    loader = pipeline.get_dataloader(batch_size=2, seq_len=128)
    
    # 3. Model (Scaled-down for local CPU check)
    # vocab_size, depth, d_model
    model = GigaGraph_3B(vocab_size=128256, depth=2, d_model=256)
    
    # 4. Single Step
    print("Running 1 Training Step on CPU...")
    batch = next(loader)
    
    labels = torch.roll(batch, -1, dims=1)
    loss = model.train_step(batch, labels)
    
    print(f"✅ GigaGraph Smoke Test Passed! Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_giga_smoke_test()
