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
        self.batch_size = self.config.BATCH_SIZE_GPU
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
        metadata_list = []  # Track source

        with open(training_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    pair = json.loads(line)

                    # Extract query and positive chunk
                    query = f"search_query: {pair['query']}"
                    positive = f"search_document: {pair['positive_text']}"

                    # Create InputExample (query, positive)
                    # MultipleNegativesRankingLoss uses in-batch negatives
                    example = InputExample(texts=[query, positive])
                    examples.append(example)
                    metadata_list.append(pair.get("source", "unknown"))  # Store source
                except Exception as e:
                    logger.warning(f"⚠️  Skipping line {line_num}: {e}")
                    continue

        if len(examples) == 0:
            raise ValueError("❌ No training examples loaded!")

        logger.info(f"✅ Loaded {len(examples):,} training examples")
        return examples, metadata_list

    def split_train_validation(self, examples, metadata_list):
        """Split with WoT-only validation, respecting validation_split config"""

        # Separate by source
        wot_examples = [examples[i] for i, src in enumerate(metadata_list) if src == "wot"]
        general_examples = [examples[i] for i, src in enumerate(metadata_list) if src == "general"]

        # Use validation_split percentage of WoT data for validation
        wot_val_size = int(len(wot_examples) * self.validation_split)

        # Split WoT
        wot_train = wot_examples[wot_val_size:]
        wot_val = wot_examples[:wot_val_size]

        # All general goes to training
        train_examples = general_examples + wot_train
        val_examples = wot_val

        logger.info(f"📊 Split: {len(train_examples):,} train ({len(general_examples):,} general + {len(wot_train):,} WoT) / {len(val_examples):,} validation (WoT only)")

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
        # MultipleNegativesRankingLoss: Uses in-batch negatives. More efficient than providing explicit negatives.
        # Trains model to work at multiple embedding dimensions simultaneously.
        # Benefits:

        # MatryoshkaLoss -> Preserves nomic's multi-dim capability
        # More robust fine-tuning
        # Can use lower dims later for speed
        base_loss = losses.MultipleNegativesRankingLoss(model)
        train_loss = losses.MatryoshkaLoss(model, base_loss, matryoshka_dims=[768, 512, 384, 256, 128, 64])

        # Calculate total steps
        steps_per_epoch = len(train_dataloader)
        total_steps = steps_per_epoch * self.num_epochs

        finetuning_parameters_stats = {
            "name": "fine_tuning_parameters",
            "metrics": {
                "total_examples": len(train_examples),
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "total_steps": total_steps,
                "learning_rate": self.learning_rate,
                "checkpoints_steps": self.checkpoints_steps,
                "warmup_steps": self.warmup_steps,
                "output_path": str(model_finetuned_path),
            },
        }

        statistics.print_results(finetuning_parameters_stats)

        from sentence_transformers.evaluation import InformationRetrievalEvaluator

        # Load 10% validation pairs
        val_queries = {f"q{i}": train_examples[i].texts[0] for i in range(len(train_examples))}
        val_corpus = {f"d{i}": train_examples[i].texts[1] for i in range(len(train_examples))}
        val_relevant = {f"q{i}": {f"d{i}"} for i in range(len(train_examples))}

        evaluator = InformationRetrievalEvaluator(val_queries, val_corpus, val_relevant)

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
            evaluator=evaluator,
            evaluation_steps=steps_per_epoch,  # evaluo una vez al final de cada epoch, es MUY costoso, mas de una hora por evaluación (1 step = total_examples / batch_size)
            use_amp=True,
        )

        model.save(str(model_finetuned_path))
        logger.info("\n✅ Fine-tuning complete! Model saved to: {model_finetuned_path}")

        return finetuning_parameters_stats


def main():
    """Main execution function"""

    start_time = datetime.now()

    # Initialize
    config = get_finetuning_config()
    paths = get_paths()
    fine_tuner = EmbeddingFineTuner()

    logger.info("📋 STEP 1: Load Training Data")
    examples, metadata_list = fine_tuner.load_training_data(paths.FILE_TRAINING_PAIRS_MIXED)

    logger.info("\n📋 STEP 2: Split Train/Validation")
    train_examples, val_examples = fine_tuner.split_train_validation(examples, metadata_list)

    print("\n" + "=" * 70)
    print("VALIDATION SET DEBUG")
    print("=" * 70)
    print(f"Total validation examples: {len(val_examples)}")
    print("\nFirst 5 validation examples:")
    for i in range(min(5, len(val_examples))):
        print(f"\n--- Example {i + 1} ---")
        print(f"Query: {val_examples[i].texts[0][:100]}...")
        print(f"Positive: {val_examples[i].texts[1][:100]}...")
    print("=" * 70 + "\n")

    logger.info("\n📋 STEP 3: Load Base Model")
    model, model_creation_stats = create_model(config.BASE_MODEL)

    logger.info("\n📋 STEP 4: Fine-Tune Model")
    finetuning_parameters_stats = fine_tuner.fine_tune(model, train_examples, paths.MODEL_CHECKPOINTS_PATH, paths.MODEL_FINETUNED_PATH)

    total_time = datetime.now() - start_time

    global_stats = []
    global_stats.append(model_creation_stats)
    global_stats.append(finetuning_parameters_stats)

    statistics.total_statistics_logging(global_stats, total_time, "FINE-TUNING STATISTICS", "02_finetune_embedding_model", False)


if __name__ == "__main__":
    try:
        main()
        exit_code = 0
    except Exception as e:
        logger.error(f"❌ An error occurred: {str(e)}")
        traceback.print_exc()
        exit_code = 1

    sys.exit(exit_code)
