"""
CodexForge - Fine-Tune Embedding Model
Fine-tunes nomic-embed-text on domain-specific training pairs.

Based on sentence-transformers MultipleNegativesRankingLoss approach.
"""

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

from src.config import get_finetuning_config
from src.logger import get_logger
from src.paths import get_paths
from src.utils.util_hugging_face_embedding import create_model
from utils import statistics

logger = get_logger(__name__)


class EmbeddingFineTuner:
    """Fine-tunes embedding models on domain-specific data"""

    def __init__(self):
        """
        Initialize fine-tuner

        Args:
            config: Config object
            paths: Paths object
        """
        self.config = get_finetuning_config()

        # Training hyperparameters (can be moved to config later)
        self.batch_size = self.batch_size
        self.num_epochs = self.config.NUM_EPOCHS
        self.warmup_steps = self.config.WARMUP_STEPS
        self.learning_rate = self.config.LEARNING_RATE
        self.checkpoints_steps = self.config.CHECKPOINT_STEPS

        self.validation_split = self.config.VALIDATION_SPLIT

        self.base_model = self.config.BASE_MODEL

    def load_training_data(self, training_file: Path):
        """
        Load training pairs from JSONL file

        Args:
            training_file: Path to training_pairs.jsonl

        Returns:
            list: List of InputExample objects for sentence-transformers
        """
        logger.info(f"📂 Loading training data from: {training_file}")

        if not training_file.exists():
            raise FileNotFoundError(f"Training file not found: {training_file}")

        examples = []

        with open(training_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)

                    # Extract query and positive chunk
                    query = pair["query"]
                    positive = pair["positive_text"]  # Single positive text string

                    # Create InputExample (query, positive)
                    # MultipleNegativesRankingLoss uses in-batch negatives
                    example = InputExample(texts=[query, positive])
                    examples.append(example)

                except Exception as e:
                    logger.warning(f"⚠️  Skipping line {line_num}: {e}")
                    continue

        if len(examples) == 0:
            raise ValueError("❌ No training examples loaded!")

        logger.info(f"✅ Loaded {len(examples):,} training examples")
        return examples

    # TODO volver a entrenar el modelo con validation_split a 0 antes de irme a casa de maria
    def split_train_validation(self, examples):
        """
        Split examples into train and validation sets

        Args:
            examples: List of InputExample objects
            validation_split: Fraction for validation (default 0.1 = 10%)

        Returns:
            tuple: (train_examples, val_examples)
        """
        total = len(examples)
        val_size = int(total * self.validation_split)
        train_size = total - val_size

        train_examples = examples[:train_size]
        val_examples = examples[train_size:]

        logger.info(f"📊 Split: {train_size:,} train / {val_size:,} validation")

        return train_examples, val_examples

    def fine_tune(self, model, train_examples, models_checkpoints_path, model_finetuned_path):
        """
        Fine-tune the model

        Args:
            model: SentenceTransformer model
            train_examples: List of InputExample objects

        Returns:
            dict: Training statistics
        """
        # Create DataLoader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.batch_size, pin_memory=True)

        # Define loss function
        # MultipleNegativesRankingLoss: Uses in-batch negatives
        # More efficient than providing explicit negatives
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # Calculate total steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.num_epochs

        stats = {
            "name": "fine_tuning",
            "metrics": {
                "total_examples": len(train_examples),
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "total_steps": total_steps,
                "learning_rate": self.learning_rate,
                "warmup_steps": self.warmup_steps,
                "output_path": str(model_finetuned_path),
            },
        }

        statistics.print_results(stats)

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.num_epochs,
            warmup_steps=self.warmup_steps,
            optimizer_params={"lr": self.learning_rate},
            checkpoint_path=str(models_checkpoints_path),
            checkpoint_save_steps=self.checkpoints_steps,
            checkpoint_save_total_limit=3,
            show_progress_bar=True,
            use_amp=True,
        )

        model.save(str(model_finetuned_path))
        logger.info("\n✅ Fine-tuning complete! Model saved to: {model_finetuned_path}")

        return stats


def main():
    """Main execution function"""

    start_time = datetime.now()

    # Initialize
    config = get_finetuning_config()
    paths = get_paths()
    fine_tuner = EmbeddingFineTuner()

    logger.info("📋 STEP 1: Load Training Data")
    examples = fine_tuner.load_training_data(paths.FILE_TRAINING_PAIRS)

    logger.info("\n📋 STEP 2: Split Train/Validation")
    train_examples, val_examples = fine_tuner.split_train_validation(examples)

    logger.info("\n📋 STEP 3: Load Base Model")
    model, model_stats = create_model(config.BASE_MODEL)

    logger.info("\n📋 STEP 4: Fine-Tune Model")
    finetune_stats = fine_tuner.fine_tune(model, train_examples, paths.MODEL_CHECKPOINTS_PATH, paths.MODEL_FINETUNED_PATH)

    # Step 5: Summary
    total_time = datetime.now() - start_time

    # TODO combinar las 2 estadisticas creacion y finetune
    statistics.total_statistics_logging(finetune_stats, total_time, "FINE-TUNING STATISTICS", "02_finetune_embedding_model", False)


if __name__ == "__main__":
    try:
        main()
        exit_code = 0
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)
