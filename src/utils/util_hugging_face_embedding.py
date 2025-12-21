from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer

from utils import statistics


def create_model(base_model):
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
            "device_name": torch.cuda.get_device_name(0),
        },
    }

    statistics.print_results(stats)

    return model, stats


def get_token_stats(model_path, sentences: list[str]):
    model = SentenceTransformer(model_path, trust_remote_code=True)

    # fmt: off
    stats = {
        "name": "tokenization_stats",
        "metrics": {
            'total_samples': len(sentences),
            'truncated_count': 0,
            'token_lengths': [],
            'max_seq_length': model.max_seq_length
        }
    }
    # fmt: on
    for sentence in tqdm(sentences, desc="Tokenizing for stats"):
        # Tokenize without truncation to see original length
        tokens = model.tokenizer(
            sentence,
            add_special_tokens=True,
            truncation=False,  # Important: don't truncate here
            return_tensors=None,
        )
        original_length = len(tokens["input_ids"])
        stats["token_lengths"].append(original_length)

        if original_length > model.max_seq_length:
            stats["truncated_count"] += 1

    stats["max_tokens"] = max(stats["token_lengths"])
    stats["avg_tokens"] = np.mean(stats["token_lengths"])
    stats["min_tokens"] = min(stats["token_lengths"])
    stats["truncation_rate"] = stats["truncated_count"] / stats["total_samples"] * 100

    return stats


def embed_chunks(model_path, texts: List[str], batch_size: int = 16, show_progress: bool = True) -> Tuple[List[List[float]], float, int, float]:
    """
    Generate embeddings for text chunks

    Returns:
        embeddings, avg_tokens, max_tokens, total_time
    """
    start_time = datetime.now()

    model = SentenceTransformer(model_path, trust_remote_code=True)

    all_embeddings = []

    # Process in batches with progress bar
    pbar = tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="Embedding")

    for i in pbar:
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings.tolist())

    total_time = (datetime.now() - start_time).total_seconds()

    # Return format matching existing code
    return all_embeddings, 0, 0, total_time  # avg_tokens, max_tokens not tracked
