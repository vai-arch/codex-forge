import torch
from sentence_transformers import SentenceTransformer

from utils import statistics


def create_model(self, base_model):
    """
    Load base model for fine-tuning

    Args:
        base_model_name: HuggingFace model name

    Returns:
        SentenceTransformer: Model ready for fine-tuning
    """

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentenceTransformer(base_model, device=device, trust_remote_code=True)

    stats = {
        "name": "model_creation",
        "metrics": {
            "device": device,
            "base_model": base_model,
            "model_embedding_dimensions": model.get_sentence_embedding_dimension(),
            "model_device": model.device,
            "real_max_seq_length": model.max_seq_length,
            "first_module_device": next(model.parameters()).device,
            "torch_cuda_available": torch.cuda.is_available(),
            "tourch_cuda_device_count": torch.cuda.device_count(),
            "torch_cuda_current_device": torch.cuda.current_device(),
        },
    }

    statistics.print_results(stats)

    return model, stats
