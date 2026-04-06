# Kaggle CLI Setup & Deployment

To deploy the **APTP-GNN v7.1** notebook directly from your terminal, follow these steps:

## 1. Authentication
1. Log in to [Kaggle.com](https://www.kaggle.com).
2. Go to your **Account Settings** (https://www.kaggle.com/settings/account).
3. Scroll to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded.

## 2. Local Configuration
Save the token to your secret directory:
```bash
mkdir -p ~/.kaggle
mv path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## 3. Deployment Command
Using **uv**, you can now push your notebook:
```bash
# Push the notebook to Kaggle Kernels
uv run kaggle kernels push -p .
```

## 4. Kaggle Secrets Configuration
Before running the kernel, add your credentials to the **Kaggle Add-ons -> Secrets** menu:
- `HF_TOKEN`: Your Hugging Face Hub Token.
- `WANDB_API_KEY`: Your Weights & Biases API Key.

## 5. Metadata Requirement
The `kaggle kernels push` command requires a `kernel-metadata.json` file. I have generated a template for you in the project root: [kernel-metadata.json](kernel-metadata.json).
