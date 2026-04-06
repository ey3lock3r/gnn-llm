"""GigaGraph GNN-LLM Package (v10.0)"""
from .models.aptp import APTPModel
from .models.noprop import NoPropModel
from .data.pipeline import GigaDataPipeline

MODEL_REGISTRY = {
    "aptp": APTPModel,
    "noprop": NoPropModel,
}

def build_model(algorithm, **kwargs):
    if algorithm not in MODEL_REGISTRY:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[algorithm](**kwargs)
