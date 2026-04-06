import torch, os, wandb
from dotenv import load_dotenv
from huggingface_hub import login
from aptp_gnn import APTPGNN_LLM
from data_pipeline import GNNDataPipeline

def run_preflight_check():
    # 1. Load Local Configuration (.env)
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    wandb_key = os.getenv('WANDB_API_KEY')
    
    print("📋 [PREFLIGHT] Checking Credentials...")
    
    # HF Check
    if hf_token: 
        print("✅ HF_TOKEN found. Logging in...")
        login(token=hf_token)
    else:
        print("❌ HF_TOKEN MISSING in .env!")
        return

    # W&B Check
    if wandb_key:
        print("✅ WANDB_API_KEY found. Logging in...")
        wandb.login(key=wandb_key)
        print("🚀 Initializing W&B Run for verification...")
        wandb.init(project="apton-gnn-llm", mode="online")
        print("✅ WANDB initialized successfully.")
    else:
        print("❌ WANDB_API_KEY MISSING in .env!")
        return

    # 2. Pipeline & Architecture Step (Local CPU)
    print("🧬 Testing 1 Training Step on CPU...")
    pipeline = GNNDataPipeline()
    batch = pipeline.get_dataloader(bs=2, sl=64)[0]
    model = APTPGNN_LLM(vocab_size=50257, depth=2, d_model=128)
    
    loss = model.train_step(batch, torch.roll(batch, -1, 1))
    wandb.log({"smoke_test_loss": loss.item()})
    
    wandb.finish()
    print("✨ [ALL PASS] Your deployment is ready for Kaggle.")

if __name__ == "__main__":
    run_preflight_check()
