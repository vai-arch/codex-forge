"""
Generic Embedding Fine-Tuning Framework - Configuration
Extends Dragon's Codex configuration with fine-tuning specific settings.
"""

import os

from dotenv import load_dotenv

# Load base Dragon's Codex config
# This imports all existing settings (OLLAMA_BASE_URL, EMBEDDING_MODEL, etc.)
try:
    from src.config import Config as BaseConfig

    HAS_BASE_CONFIG = True
except ImportError:
    HAS_BASE_CONFIG = False
    BaseConfig = object


class FineTuningConfig(BaseConfig if HAS_BASE_CONFIG else object):
    """
    Fine-tuning configuration extending Dragon's Codex base config.

    If Dragon's Codex config exists, inherits all settings.
    Otherwise, provides standalone configuration.
    """

    def __init__(self, env_file=".env"):
        """Initialize fine-tuning configuration"""

        # Load base config if available
        if HAS_BASE_CONFIG:
            super().__init__(env_file)
        else:
            load_dotenv(override=True)

        # =================================================================
        # FINE-TUNING SETTINGS
        # =================================================================

        # Model settings
        self.BASE_MODEL = os.getenv("BASE_MODEL", "nomic-embed-text")
        self.MODEL_BACKEND = os.getenv("MODEL_BACKEND", "ollama")  # ollama, huggingface, openai
        self.EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

        # Training hyperparameters
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 2e-5))
        self.BATCH_SIZE_GPU = int(os.getenv("BATCH_SIZE_GPU", 32))
        self.BATCH_SIZE_CPU = int(os.getenv("BATCH_SIZE_CPU", 8))
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
        self.WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 500))
        self.MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 512))

        # Loss function settings
        self.LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "MultipleNegativesRankingLoss")
        self.SCALE = float(os.getenv("LOSS_SCALE", 20.0))  # Temperature scaling

        # Training pair generation
        self.NUM_HARD_NEGATIVES = int(os.getenv("NUM_HARD_NEGATIVES", 3))
        self.POSITIVES_PER_QUERY = int(os.getenv("POSITIVES_PER_QUERY", 3))
        self.SYNTHETIC_EXPANSION_FACTOR = int(os.getenv("SYNTHETIC_EXPANSION_FACTOR", 5))

        # Topic matching thresholds
        self.MIN_TOPIC_COVERAGE = float(os.getenv("MIN_TOPIC_COVERAGE", 0.5))  # 50% of expected topics
        self.ALIAS_MATCH_WEIGHT = float(os.getenv("ALIAS_MATCH_WEIGHT", 2.0))  # Boost alias matches

        # Checkpointing
        self.CHECKPOINT_STEPS = int(os.getenv("CHECKPOINT_STEPS", 500))
        self.SAVE_BEST_MODEL = os.getenv("SAVE_BEST_MODEL", "True").lower() == "true"

        # Evaluation settings
        self.VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.0))  # 0 = use all for training
        self.EVAL_STEPS = int(os.getenv("EVAL_STEPS", 1000))
        self.EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", 3))

        # Additional evaluation metrics
        self.RECALL_K_VALUES = [1, 3, 5, 10]  # Calculate Recall@1, @3, @5, @10

        # Device settings
        self.FORCE_CPU = os.getenv("FORCE_CPU", "False").lower() == "true"

        # =================================================================
        # FALLBACK SETTINGS (if no base config)
        # =================================================================

        if not HAS_BASE_CONFIG:
            # Ollama settings
            self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

            # Basic paths
            from pathlib import Path

            self.PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
            self.DATA_PATH = self.PROJECT_ROOT / "data"

    def get_batch_size(self, device: str) -> int:
        """Get appropriate batch size based on device"""
        if device == "cuda":
            return self.BATCH_SIZE_GPU
        return self.BATCH_SIZE_CPU

    def __repr__(self):
        """String representation"""
        return (
            f"FineTuningConfig(\n"
            f"  BASE_MODEL={self.BASE_MODEL},\n"
            f"  BACKEND={self.MODEL_BACKEND},\n"
            f"  LR={self.LEARNING_RATE},\n"
            f"  EPOCHS={self.NUM_EPOCHS},\n"
            f"  BATCH_SIZE={self.BATCH_SIZE_GPU}/{self.BATCH_SIZE_CPU}\n"
            f")"
        )


# Global instance
_finetuning_config = None


def get_finetuning_config():
    """Get the global fine-tuning configuration instance"""
    global _finetuning_config
    if _finetuning_config is None:
        _finetuning_config = FineTuningConfig()
    return _finetuning_config


def print_config():
    """Print current fine-tuning configuration"""
    config = get_finetuning_config()

    print("=" * 70)
    print("EMBEDDING FINE-TUNING FRAMEWORK - CONFIGURATION")
    print("=" * 70)

    print("\n🤖 MODEL SETTINGS:")
    print(f"  Base Model: {config.BASE_MODEL}")
    print(f"  Backend: {config.MODEL_BACKEND}")
    print(f"  Embedding Dimension: {config.EMBEDDING_DIM}")

    print("\n⚙️  TRAINING HYPERPARAMETERS:")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Batch Size (GPU): {config.BATCH_SIZE_GPU}")
    print(f"  Batch Size (CPU): {config.BATCH_SIZE_CPU}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Warmup Steps: {config.WARMUP_STEPS}")
    print(f"  Max Sequence Length: {config.MAX_SEQ_LENGTH}")

    print("\n📊 LOSS FUNCTION:")
    print(f"  Type: {config.LOSS_FUNCTION}")
    print(f"  Scale: {config.SCALE}")

    print("\n🎯 TRAINING PAIRS:")
    print(f"  Hard Negatives per Query: {config.NUM_HARD_NEGATIVES}")
    print(f"  Positives per Query: {config.POSITIVES_PER_QUERY}")
    print(f"  Synthetic Expansion Factor: {config.SYNTHETIC_EXPANSION_FACTOR}x")
    print(f"  Min Topic Coverage: {config.MIN_TOPIC_COVERAGE * 100}%")

    print("\n💾 CHECKPOINTING:")
    print(f"  Checkpoint Every: {config.CHECKPOINT_STEPS} steps")
    print(f"  Save Best Model: {config.SAVE_BEST_MODEL}")

    print("\n📈 EVALUATION:")
    print(f"  Validation Split: {config.VALIDATION_SPLIT * 100}%")
    print(f"  Recall@K: {config.RECALL_K_VALUES}")
    print(f"  Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE} evals")

    print("=" * 70)


if __name__ == "__main__":
    # Test configuration
    print_config()
