"""
Generic Fine-Tuning Framework - Paths Configuration
Manages all file paths for fine-tuning embedding models on any corpus.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


class Paths:
    """
    Path configuration for fine-tuning project.
    Works with any domain corpus.
    """

    def __init__(self, env_file=".env"):
        """Initialize paths by loading .env file"""
        load_dotenv(override=True)

        # Project root
        self.PROJECT_ROOT_PATH = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
        self.DATA_PATH = self.PROJECT_ROOT_PATH / "data"

        # =====================================================================
        # INPUT: Source corpus
        # =====================================================================
        self.CORPUS_PATH = self.DATA_PATH / "corpus"
        self.RAW_CORPUS_PATH = self.CORPUS_PATH / "raw"  # Original text files
        self.INDEXES_PATH = self.CORPUS_PATH / "indexes"  # Domain indexes (entities, terms)

        # =====================================================================
        # PROCESSING: Intermediate files
        # =====================================================================
        self.PROCESSED_PATH = self.DATA_PATH / "processed"
        self.CHUNKS_PATH = self.PROCESSED_PATH / "chunks"  # Chunked text
        self.TRAINING_PAIRS_PATH = self.PROCESSED_PATH / "pairs"  # (query, doc) pairs

        # =====================================================================
        # EMBEDDINGS: Generated embeddings
        # =====================================================================
        self.EMBEDDINGS_PATH = self.DATA_PATH / "embeddings"
        self.BASE_EMBEDDINGS_PATH = self.EMBEDDINGS_PATH / "base"  # Base model embeddings
        self.FINETUNED_EMBEDDINGS_PATH = self.EMBEDDINGS_PATH / "finetuned"  # Fine-tuned embeddings

        # =====================================================================
        # MODEL: Fine-tuning artifacts
        # =====================================================================
        self.MODELS_PATH = self.PROJECT_ROOT_PATH / "models"
        self.CHECKPOINTS_PATH = self.MODELS_PATH / "checkpoints"  # Training checkpoints
        self.FINETUNED_MODEL_PATH = self.MODELS_PATH / "finetuned"  # Final model

        # =====================================================================
        # EVALUATION: Test sets and results
        # =====================================================================
        self.EVALUATION_PATH = self.DATA_PATH / "evaluation"
        self.TEST_SETS_PATH = self.EVALUATION_PATH / "test_sets"
        self.RESULTS_PATH = self.EVALUATION_PATH / "results"

        # =====================================================================
        # LOGS: Training and evaluation logs
        # =====================================================================
        self.LOG_PATH = self.PROJECT_ROOT_PATH / "logs"
        self.TRAINING_LOG_PATH = self.LOG_PATH / "training"
        self.EVALUATION_LOG_PATH = self.LOG_PATH / "evaluation"
        self.STATISTICS_PATH = self.LOG_PATH / "statistics"

        # =====================================================================
        # FILE PATHS: Specific files
        # =====================================================================

        # Corpus files
        self.FILE_CORPUS_CHUNKS = self.CHUNKS_PATH / "corpus_chunks.jsonl"
        self.FILE_ENTITY_INDEX = self.INDEXES_PATH / "entity_index.json"  # Domain entities
        self.FILE_TERM_INDEX = self.INDEXES_PATH / "term_index.json"  # Important terms
        self.FILE_CONCEPT_INDEX = self.INDEXES_PATH / "concept_index.json"  # Domain concepts

        # Training files
        self.FILE_TRAINING_PAIRS = self.TRAINING_PAIRS_PATH / "training_pairs.jsonl"
        self.FILE_VALIDATION_PAIRS = self.TRAINING_PAIRS_PATH / "validation_pairs.jsonl"
        self.FILE_TRAINING_CHECKPOINT = self.CHECKPOINTS_PATH / "checkpoint_latest.pt"
        self.FILE_TRAINING_CONFIG = self.MODELS_PATH / "training_config.json"

        # Embedding files
        self.FILE_BASE_EMBEDDINGS = self.BASE_EMBEDDINGS_PATH / "base_embeddings.pkl"
        self.FILE_FINETUNED_EMBEDDINGS = self.FINETUNED_EMBEDDINGS_PATH / "finetuned_embeddings.pkl"

        # Evaluation files
        self.FILE_TEST_QUESTIONS = self.TEST_SETS_PATH / "test_questions.json"
        self.FILE_BASELINE_RESULTS = self.RESULTS_PATH / "baseline_results.json"
        self.FILE_FINETUNED_RESULTS = self.RESULTS_PATH / "finetuned_results.json"
        self.FILE_COMPARISON_REPORT = self.RESULTS_PATH / "comparison_report.json"

        # Log files
        self.FILE_MAIN_LOG = self.LOG_PATH / "finetune.log"
        self.FILE_TRAINING_LOG = self.TRAINING_LOG_PATH / "training.log"
        self.FILE_TRAINING_STATS = self.STATISTICS_PATH / "training_stats.json"
        self.FILE_EVALUATION_STATS = self.STATISTICS_PATH / "evaluation_stats.json"

        # Create all directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        for name, value in self.__dict__.items():
            if name.endswith("_PATH") and isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        """String representation"""
        return f"Paths(PROJECT_ROOT={self.PROJECT_ROOT_PATH})"


# Global paths instance
_paths = None


def get_paths():
    """Get the global paths instance"""
    global _paths
    if _paths is None:
        _paths = Paths()
    return _paths


# Convenience function for testing
def print_paths():
    """Print current paths (useful for debugging)"""
    paths = get_paths()

    print("=" * 70)
    print("Fine-Tuning Framework - Paths & Files")
    print("=" * 70)

    print("\n📁 DIRECTORY PATHS:")
    for name, value in paths.__dict__.items():
        if name.endswith("_PATH") and isinstance(value, Path):
            exists = "✅" if value.exists() else "❌"
            print(f"  {exists} {name:30s}: {value}")

    print("\n📄 FILE PATHS:")
    for name, value in paths.__dict__.items():
        if name.startswith("FILE_") and isinstance(value, Path):
            exists = "✅" if value.exists() else "❌"
            print(f"  {exists} {name:30s}: {value}")


if __name__ == "__main__":
    print_paths()
