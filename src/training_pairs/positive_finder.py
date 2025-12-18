"""
Generic Fine-Tuning Framework - Positive Chunk Finder
Finds chunks that match expected topics for a query.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from data_loader import Chunk, TestQuestion


@dataclass
class PositiveMatch:
    """Represents a positive chunk match with scoring"""

    chunk: Chunk
    score: float
    matched_topics: List[str]
    match_details: Dict[str, any]


class PositiveFinder:
    """Finds positive chunks for training queries"""

    def __init__(self, config=None):
        """
        Initialize positive finder

        Args:
            config: Fine-tuning configuration
        """
        if config is None:
            from config import get_finetuning_config

            config = get_finetuning_config()

        self.config = config
        self.filename_map = {}
        self.indexes = {}

    def set_data(self, chunks: List[Chunk], indexes: Dict[str, Dict]):
        """
        Set corpus data for searching

        Args:
            chunks: List of all chunks
            indexes: Dictionary of indexes (character, concept, magic, prophecy)
        """
        # Build filename lookup map
        self.filename_map = defaultdict(list)
        for chunk in chunks:
            self.filename_map[chunk.filename].append(chunk)

        self.indexes = indexes

        print("✅ PositiveFinder initialized:")
        print(f"   Chunks indexed: {len(chunks):,}")
        print(f"   Unique files: {len(self.filename_map):,}")
        print(f"   Indexes loaded: {list(indexes.keys())}")

    def find_entity_aliases(self, entity_name: str, entity_type: str = None) -> Set[str]:
        """
        Find all aliases for an entity using indexes

        Args:
            entity_name: Entity name to search for
            entity_type: Type hint ("character", "concept", etc.)

        Returns:
            Set of aliases including the entity name itself
        """
        aliases = {entity_name}

        # Search appropriate index
        if entity_type and entity_type in self.indexes:
            index = self.indexes[entity_type]
            if entity_name in index:
                entity_data = index[entity_name]
                if "aliases" in entity_data:
                    aliases.update(entity_data["aliases"])
                if "primary_name" in entity_data:
                    aliases.add(entity_data["primary_name"])
        else:
            # Search all indexes
            for index_type, index in self.indexes.items():
                if entity_name in index:
                    entity_data = index[entity_name]
                    if "aliases" in entity_data:
                        aliases.update(entity_data["aliases"])
                    if "primary_name" in entity_data:
                        aliases.add(entity_data["primary_name"])

        return aliases

    def get_entity_filename(self, entity_name: str, entity_type: str = None) -> str:
        """
        Get the filename associated with an entity from indexes

        Args:
            entity_name: Entity name
            entity_type: Type hint

        Returns:
            Filename or None
        """
        if entity_type and entity_type in self.indexes:
            index = self.indexes[entity_type]
            if entity_name in index:
                return index[entity_name].get("filename")
        else:
            # Search all indexes
            for index in self.indexes.values():
                if entity_name in index:
                    return index[entity_name].get("filename")

        return None

    def score_chunk_for_topics(self, chunk: Chunk, expected_topics: List[str], query_text: str = None) -> Tuple[float, List[str], Dict]:
        """
        Score how well a chunk matches expected topics

        Args:
            chunk: Chunk to score
            expected_topics: List of topics that should be present
            query_text: Optional query text for additional matching

        Returns:
            Tuple of (score, matched_topics, details)
        """
        matched_topics = []
        match_details = {"alias_matches": [], "text_matches": [], "mention_matches": [], "filename_matches": []}

        chunk_text_lower = chunk.text.lower()
        score = 0.0

        for topic in expected_topics:
            topic_lower = topic.lower()
            topic_matched = False

            # 1. Check if topic is in chunk text
            if topic_lower in chunk_text_lower:
                score += 1.0
                matched_topics.append(topic)
                match_details["text_matches"].append(topic)
                topic_matched = True

            # 2. Check if topic matches via aliases (higher weight)
            aliases = self.find_entity_aliases(topic)
            for alias in aliases:
                if alias.lower() in chunk_text_lower:
                    score += self.config.ALIAS_MATCH_WEIGHT
                    if not topic_matched:
                        matched_topics.append(topic)
                        topic_matched = True
                    match_details["alias_matches"].append(f"{topic} (via {alias})")

            # 3. Check entity mentions
            all_mentions = chunk.character_mentions + chunk.concept_mentions + chunk.magic_mentions + chunk.prophecy_mentions

            for mention in all_mentions:
                if topic_lower in mention.lower() or mention.lower() in topic_lower:
                    score += 0.5
                    if not topic_matched:
                        matched_topics.append(topic)
                        topic_matched = True
                    match_details["mention_matches"].append(f"{topic} (mention: {mention})")

            # 4. Check if topic has associated filename
            filename = self.get_entity_filename(topic)
            if filename and filename == chunk.filename:
                score += 2.0  # Strong signal!
                if not topic_matched:
                    matched_topics.append(topic)
                    topic_matched = True
                match_details["filename_matches"].append(f"{topic} (file: {filename})")

        # Normalize score by number of expected topics
        if expected_topics:
            normalized_score = score / len(expected_topics)
        else:
            normalized_score = 0.0

        return normalized_score, matched_topics, match_details

    def find_positives(self, question: TestQuestion, top_k: int = None) -> List[PositiveMatch]:
        """
        Find positive chunks for a test question

        Args:
            question: TestQuestion with expected topics
            top_k: Number of positives to return (default from config)

        Returns:
            List of PositiveMatch objects, sorted by score
        """
        if top_k is None:
            top_k = self.config.POSITIVES_PER_QUERY

        matches = []

        # Strategy 1: Try to find chunks by filename (if topics map to entities)
        for topic in question.expected_topics:
            filename = self.get_entity_filename(topic, question.category)
            if filename and filename in self.filename_map:
                for chunk in self.filename_map[filename]:
                    score, matched, details = self.score_chunk_for_topics(chunk, question.expected_topics, question.question)

                    if score > 0:
                        matches.append(PositiveMatch(chunk=chunk, score=score, matched_topics=matched, match_details=details))

        # Strategy 2: Search all chunks if we don't have enough matches
        if len(matches) < top_k:
            # Score all chunks (expensive but comprehensive)
            for chunks in self.filename_map.values():
                for chunk in chunks:
                    # Skip if already matched
                    if any(m.chunk.chunk_id == chunk.chunk_id for m in matches):
                        continue

                    score, matched, details = self.score_chunk_for_topics(chunk, question.expected_topics, question.question)

                    # Only consider if meets minimum coverage
                    topic_coverage = len(matched) / len(question.expected_topics) if question.expected_topics else 0
                    if topic_coverage >= self.config.MIN_TOPIC_COVERAGE:
                        matches.append(PositiveMatch(chunk=chunk, score=score, matched_topics=matched, match_details=details))

        # Sort by score and return top_k
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def find_positives_batch(self, questions: List[TestQuestion], top_k: int = None) -> Dict[int, List[PositiveMatch]]:
        """
        Find positives for multiple questions

        Args:
            questions: List of TestQuestion objects
            top_k: Number of positives per question

        Returns:
            Dictionary mapping question_id to list of PositiveMatch
        """
        results = {}

        print(f"\n🔍 Finding positive chunks for {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            positives = self.find_positives(question, top_k)
            results[question.question_id] = positives

            # Progress update
            if i % 10 == 0 or i == len(questions):
                avg_matches = sum(len(p) for p in results.values()) / len(results)
                print(f"   Processed {i}/{len(questions)} questions (avg {avg_matches:.1f} positives/question)")

        return results


if __name__ == "__main__":
    # Test positive finder
    print("Testing PositiveFinder...")

    from data_loader import DataLoader

    from src.paths import get_paths

    paths = get_paths()
    loader = DataLoader(paths)

    # This would load real data
    print("\n✅ PositiveFinder module ready")
