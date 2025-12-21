"""
Generic Embedding Fine-Tuning Framework - Configuration
Extends Dragon's Codex configuration with fine-tuning specific settings.
"""

import os

from dotenv import load_dotenv


class FineTuningConfig:
    """
    Fine-tuning configuration extending Dragon's Codex base config.

    If Dragon's Codex config exists, inherits all settings.
    Otherwise, provides standalone configuration.
    """

    def __init__(self, env_file=".env"):
        """Initialize fine-tuning configuration"""

        load_dotenv(override=True)
        # =================================================================
        # TRAINING PAIRS GENERATION SETTINGS
        # =================================================================

        self.MIN_POSITIVE_SCORE = 1.0  # ✅ Allows single text matches
        self.MIN_POSITIVE_SCORE_FALLBACK = 0.5  # ✅ Fallback even lower
        self.MIN_POSITIVES_REQUIRED = 2  # Need at least 2 positives

        # =================================================================
        # FINE-TUNING SETTINGS
        # =================================================================

        # Model settings
        self.BASE_MODEL = os.getenv("BASE_MODEL", "nomic-ai/nomic-embed-text-v1.5")

        # Training hyperparameters
        self.LEARNING_RATE = float(os.getenv("LEARNING_RATE", 5e-6))
        self.BATCH_SIZE_GPU = int(os.getenv("BATCH_SIZE_GPU", 16))
        self.NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 1))
        self.WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", 200))
        self.CHECKPOINT_STEPS = int(os.getenv("CHECKPOINT_STEPS", 50))
        self.SYNTHETIC_EXPANSION_FACTOR = int(os.getenv("SYNTHETIC_EXPANSION_FACTOR", 5))
        self.POSITIVES_PER_QUERY = int(os.getenv("POSITIVES_PER_QUERY", 3))
        self.NUM_HARD_NEGATIVES = int(os.getenv("NUM_HARD_NEGATIVES", 3))

        # first attempt went wrong witgh:
        # "LEARNING_RATE", 2e-5
        # "NUM_EPOCHS", 4
        # "WARMUP_STEPS", 100
        # second attempt:
        # self.learning_rate = 1e-5  # Was 2e-5 (cut in half)
        # self.num_epochs = 1        # Was 4 (reduce overfitting)
        # self.warmup_steps = 200    # Was 100 (more gradual start)

        # Evaluation settings
        self.VALIDATION_SPLIT = float(os.getenv("VALIDATION_SPLIT", 0.1))  # 0 = use all for training puede servir para hacer pruebas rapidas si lo pongo al 0.99

        # TODO We are not using this anywhere in the code. See plan.md phase 2A-1
        self.MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", 2048))

    def get_batch_size(self, device: str) -> int:
        """Get appropriate batch size based on device"""
        if device == "cuda":
            return self.BATCH_SIZE_GPU
        return self.BATCH_SIZE_CPU

    # fmt: off
    def __repr__(self):
        """String representation"""
        return (
            f"FineTuningConfig(\n"
            f"  BASE_MODEL={self.BASE_MODEL},\n"
            f"  LR={self.LEARNING_RATE},\n"
            f"  EPOCHS={self.NUM_EPOCHS},\n"
            f"  BATCH_SIZE={self.BATCH_SIZE_GPU}\n"
            f")"
        )
    # fmt: on


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

    print("\n⚙️  TRAINING HYPERPARAMETERS:")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Batch Size (GPU): {config.BATCH_SIZE_GPU}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Warmup Steps: {config.WARMUP_STEPS}")
    print(f"  Max Sequence Length: {config.MAX_SEQ_LENGTH}")

    print("\n💾 CHECKPOINTING:")
    print(f"  Checkpoint Every: {config.CHECKPOINT_STEPS} steps")

    print("\n📈 EVALUATION:")
    print(f"  Validation Split: {config.VALIDATION_SPLIT * 100}%")

    print("=" * 70)


if __name__ == "__main__":
    # Test configuration
    print_config()
