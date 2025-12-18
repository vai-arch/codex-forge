"""
Generic Embedding Fine-Tuning Framework
Script 01: Generate Training Pairs

This script generates training pairs (query, positive, negatives) from:
- Chunks (with entity mentions)
- Indexes (character, concept, magic, prophecy)
- Test questions (with expected topics)

Usage:
    python 01_generate_training_pairs.py
    python 01_generate_training_pairs.py --no-expand  # Don't synthesize queries
    python 01_generate_training_pairs.py --output custom_pairs.jsonl
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_finetuning_config, print_config
from src.paths import get_paths, print_paths
from src.training_pairs.training_pair_generator import TrainingPairGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate training pairs for embedding fine-tuning")

    parser.add_argument("--chunks-dir", type=Path, default=None, help="Directory containing chunk files (default: from paths.py)")

    parser.add_argument("--indexes-dir", type=Path, default=None, help="Directory containing index files (default: from paths.py)")

    parser.add_argument("--questions-file", type=Path, default=None, help="Test questions JSON file (default: from paths.py)")

    parser.add_argument("--output", "-o", type=Path, default=None, help="Output file for training pairs (default: from paths.py)")

    parser.add_argument("--no-expand", action="store_true", help="Don't synthesize additional queries (use only original test questions)")

    parser.add_argument("--show-config", action="store_true", help="Show configuration and exit")

    parser.add_argument("--show-paths", action="store_true", help="Show paths and exit")

    return parser.parse_args()


def main():
    """Main execution"""
    start_time = datetime.now()

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = get_finetuning_config()
    paths = get_paths()

    # Show config/paths if requested
    if args.show_config:
        print_config()
        return 0

    if args.show_paths:
        print_paths()
        return 0

    # Print header
    print("\n" + "=" * 70)
    print("EMBEDDING FINE-TUNING FRAMEWORK")
    print("Step 1: Generate Training Pairs")
    print("=" * 70)
    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Determine file paths
    chunks_dir = args.chunks_dir or paths.CHUNKS_PATH
    indexes_dir = args.indexes_dir or paths.INDEXES_PATH
    questions_file = args.questions_file or paths.FILE_TEST_QUESTIONS
    output_file = args.output or paths.FILE_TRAINING_PAIRS

    print("\n📂 Configuration:")
    print(f"   Chunks: {chunks_dir}")
    print(f"   Indexes: {indexes_dir}")
    print(f"   Questions: {questions_file}")
    print(f"   Output: {output_file}")
    print(f"   Expand queries: {not args.no_expand}")

    try:
        # Initialize generator
        print("\n🚀 Initializing training pair generator...")
        generator = TrainingPairGenerator(config, paths)

        # Load data
        generator.load_data(chunks_dir=chunks_dir, indexes_dir=indexes_dir, questions_file=questions_file)

        # Generate pairs
        training_pairs = generator.generate_pairs(output_file=output_file, expand_queries=not args.no_expand, save_intermediate=True)

        # Summary
        end_time = datetime.now()
        duration = end_time - start_time

        print("\n" + "=" * 70)
        print("✅ TRAINING PAIR GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated: {len(training_pairs):,} training pairs")
        print(f"Saved to: {output_file}")
        print(f"Duration: {duration}")
        print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n📊 Next step:")
        print("   python 02_finetune_model.py")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
