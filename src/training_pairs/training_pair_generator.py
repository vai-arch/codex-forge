"""
Generic Fine-Tuning Framework - Training Pair Generator
Orchestrates complete training pair generation: test questions → synthetic expansion → pairs
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import files  # noqa: F401

from src.training_pairs.data_loader import Chunk, DataLoader, TestQuestion
from src.training_pairs.negative_miner import NegativeMiner
from src.training_pairs.positive_finder import PositiveFinder


@dataclass
class TrainingPair:
    """Represents a single training triplet (query, positive, negative)"""

    query: str
    query_id: str
    positive_text: str
    positive_chunk_id: str
    negative_texts: List[str]
    negative_chunk_ids: List[str]
    metadata: Dict

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class QuerySynthesizer:
    """Synthesizes additional training queries from chunks"""

    def __init__(self, config=None):
        """Initialize query synthesizer"""
        if config is None:
            from config import get_finetuning_config

            config = get_finetuning_config()

        self.config = config

    def synthesize_from_chunk(self, chunk: Chunk, indexes: Dict) -> List[str]:
        """
        Generate synthetic queries from a chunk

        Args:
            chunk: Chunk to generate queries from
            indexes: Entity indexes for context

        Returns:
            List of synthetic query strings
        """
        queries = []

        # Character queries
        for char in chunk.character_mentions:
            queries.append(f"Who is {char}?")
            queries.append(f"Tell me about {char}.")
            queries.append(f"Describe {char}.")

        # Concept queries
        for concept in chunk.concept_mentions:
            queries.append(f"What is {concept}?")
            queries.append(f"Explain {concept}.")
            queries.append(f"Define {concept}.")

        # Magic queries
        for magic in chunk.magic_mentions:
            queries.append(f"What is {magic}?")
            queries.append(f"How does {magic} work?")
            queries.append(f"Explain {magic}.")

        # Prophecy queries
        for prophecy in chunk.prophecy_mentions:
            queries.append(f"What is {prophecy}?")
            queries.append(f"Tell me about {prophecy}.")

        return queries

    def expand_test_questions(self, questions: List[TestQuestion], expansion_factor: int = None) -> List[Tuple[str, TestQuestion]]:
        """
        Expand test questions with variations

        Args:
            questions: Original test questions
            expansion_factor: How many variations per question

        Returns:
            List of (query_text, original_question) tuples
        """
        if expansion_factor is None:
            expansion_factor = self.config.SYNTHETIC_EXPANSION_FACTOR

        expanded = []

        # Question templates
        templates = {
            "character": [
                "Who is {topic}?",
                "Tell me about {topic}.",
                "Describe {topic}.",
                "What do we know about {topic}?",
                "Explain {topic}.",
            ],
            "concept": [
                "What is {topic}?",
                "Explain {topic}.",
                "Define {topic}.",
                "Describe {topic}.",
                "What does {topic} mean?",
            ],
            "magic_system": [
                "What is {topic}?",
                "How does {topic} work?",
                "Explain {topic}.",
                "Describe {topic}.",
                "What is the purpose of {topic}?",
            ],
            "prophecy": [
                "What is {topic}?",
                "Tell me about {topic}.",
                "Explain {topic}.",
                "What does {topic} say?",
                "Describe {topic}.",
            ],
            "plot_event": [
                "What happens in {topic}?",
                "Describe {topic}.",
                "Tell me about {topic}.",
                "Explain {topic}.",
                "What occurred during {topic}?",
            ],
        }

        for question in questions:
            # Original question
            expanded.append((question.question, question))

            # Generate variations using templates
            category_templates = templates.get(question.category, templates["concept"])

            for topic in question.expected_topics[:2]:  # Use first 2 topics
                for template in category_templates[:expansion_factor]:
                    synthetic_query = template.format(topic=topic)
                    expanded.append((synthetic_query, question))

        return expanded


class TrainingPairGenerator:
    """Complete pipeline for generating training pairs"""

    def __init__(self, config=None, paths=None):
        """
        Initialize training pair generator

        Args:
            config: Configuration
            paths: Paths configuration
        """
        if config is None:
            from config import get_finetuning_config

            config = get_finetuning_config()

        if paths is None:
            from paths import get_paths

            paths = get_paths()

        self.config = config
        self.paths = paths

        # Initialize components
        self.data_loader = DataLoader(paths)
        self.positive_finder = PositiveFinder(config)
        self.negative_miner = NegativeMiner(config)
        self.query_synthesizer = QuerySynthesizer(config)

        # Data
        self.chunks = []
        self.indexes = {}
        self.test_questions = []

    def load_data(self, chunks_dir: Path = None, indexes_dir: Path = None, questions_file: Path = None):
        """
        Load all required data

        Args:
            chunks_dir: Directory with chunk files
            indexes_dir: Directory with index files
            questions_file: Test questions file
        """
        print("\n" + "=" * 70)
        print("LOADING DATA")
        print("=" * 70)

        # Load chunks
        self.chunks = self.data_loader.load_all_chunks(chunks_dir)

        # Load indexes
        self.indexes = self.data_loader.load_all_indexes(indexes_dir)

        # Load test questions
        self.test_questions = self.data_loader.load_test_questions(questions_file)

        # Initialize finders with data
        self.positive_finder.set_data(self.chunks, self.indexes)
        self.negative_miner.set_data(self.chunks, self.indexes)

        print("\n✅ Data loaded successfully")

    def generate_pairs(self, output_file: Path = None, expand_queries: bool = True, save_intermediate: bool = True) -> List[TrainingPair]:
        """
        Generate all training pairs

        Args:
            output_file: Where to save pairs
            expand_queries: Whether to synthesize additional queries
            save_intermediate: Save intermediate results

        Returns:
            List of TrainingPair objects
        """
        print("\n" + "=" * 70)
        print("GENERATING TRAINING PAIRS")
        print("=" * 70)

        # Step 1: Expand queries
        if expand_queries:
            print("\n📝 Expanding test questions...")
            expanded_queries = self.query_synthesizer.expand_test_questions(self.test_questions[:2])
            print(f"   Generated {len(expanded_queries)} queries from {len(self.test_questions)} questions")
        else:
            expanded_queries = [(q.question, q) for q in self.test_questions]

        # Step 2: Find positives for all unique questions
        print("\n🔍 Finding positive chunks...")
        unique_questions = {q.question_id: q for _, q in expanded_queries}
        unique_questions = list(unique_questions.values())

        positive_results = self.positive_finder.find_positives_batch(unique_questions)

        # Statistics
        total_positives = sum(len(p) for p in positive_results.values())
        avg_positives = total_positives / len(positive_results) if positive_results else 0
        print(f"   Found {total_positives:,} total positive chunks ({avg_positives:.1f} avg/question)")

        # Step 3: Mine hard negatives
        print("\n⛏️  Mining hard negatives...")
        negative_results = self.negative_miner.mine_negatives_batch(unique_questions, positive_results)

        total_negatives = sum(len(n) for n in negative_results.values())
        avg_negatives = total_negatives / len(negative_results) if negative_results else 0
        print(f"   Mined {total_negatives:,} hard negatives ({avg_negatives:.1f} avg/question)")

        # Step 4: Create training pairs
        print("\n🔗 Creating training pairs...")
        training_pairs = []

        for query_text, question in expanded_queries:
            positives = positive_results.get(question.question_id, [])
            negatives = negative_results.get(question.question_id, [])

            if not positives or not negatives:
                continue

            # Create pairs: each positive with all negatives for that query
            for pos_match in positives:
                pair = TrainingPair(
                    query=query_text,
                    query_id=f"q{question.question_id}_{len(training_pairs)}",
                    positive_text=pos_match.chunk.text,
                    positive_chunk_id=pos_match.chunk.chunk_id,
                    negative_texts=[n.chunk.text for n in negatives],
                    negative_chunk_ids=[n.chunk.chunk_id for n in negatives],
                    metadata={
                        "original_question_id": question.question_id,
                        "category": question.category,
                        "difficulty": question.difficulty,
                        "expected_topics": question.expected_topics,
                        "positive_score": pos_match.score,
                        "positive_matched_topics": pos_match.matched_topics,
                        "num_negatives": len(negatives),
                    },
                )
                training_pairs.append(pair)

        print(f"   Created {len(training_pairs):,} training pairs")

        # Step 5: Save pairs
        if output_file:
            self._save_pairs(training_pairs, output_file)

        # Statistics
        self._print_statistics(training_pairs)

        return training_pairs, self.compute_statistics(training_pairs)

    def _save_pairs(self, pairs: List[TrainingPair], output_file: Path):
        """Save training pairs to file"""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Saving training pairs to {output_file}...")

        with open(output_file, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")

        print(f"   ✅ Saved {len(pairs):,} pairs")

    def compute_statistics(self, pairs: List["TrainingPair"]) -> Dict[str, Any]:
        """
        Compute statistics for a list of TrainingPair objects.
        Returns a dictionary suitable for logging with log_results().
        """
        stats: Dict[str, Any] = {}

        # Overall
        stats["overall"] = {"total_pairs": len(pairs)}

        # By category
        by_category: Dict[str, int] = {}
        for pair in pairs:
            cat = pair.metadata.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
        stats["by_category"] = by_category

        # By difficulty
        by_difficulty: Dict[str, int] = {}
        for pair in pairs:
            diff = pair.metadata.get("difficulty", "unknown")
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1
        stats["by_difficulty"] = by_difficulty

        # Average negatives per pair
        avg_negs = sum(len(p.negative_texts) for p in pairs) / len(pairs) if pairs else 0
        stats["average_negatives_per_pair"] = avg_negs

        return stats

    def _print_statistics(self, pairs: List[TrainingPair]):
        """Print statistics about generated pairs"""
        print("\n" + "=" * 70)
        print("TRAINING PAIR STATISTICS")
        print("=" * 70)

        print("\n📊 Overall:")
        print(f"   Total pairs: {len(pairs):,}")

        # By category
        by_category = {}
        for pair in pairs:
            cat = pair.metadata["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        print("\n📂 By category:")
        for cat, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(pairs)) * 100
            print(f"   {cat:15s}: {count:5,} ({pct:5.1f}%)")

        # By difficulty
        by_difficulty = {}
        for pair in pairs:
            diff = pair.metadata["difficulty"]
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

        print("\n⭐ By difficulty:")
        for diff in ["easy", "medium", "hard"]:
            count = by_difficulty.get(diff, 0)
            pct = (count / len(pairs)) * 100
            print(f"   {diff:10s}: {count:5,} ({pct:5.1f}%)")

        # Average negatives per pair
        avg_negs = sum(len(p.negative_texts) for p in pairs) / len(pairs) if pairs else 0
        print(f"\n🎯 Average negatives per pair: {avg_negs:.1f}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    print("✅ TrainingPairGenerator module ready")
