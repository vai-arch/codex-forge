"""
Generic Fine-Tuning Framework - Hard Negative Miner
Finds semantically similar but incorrect chunks for contrastive learning.
"""

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set

from data_loader import Chunk, TestQuestion
from positive_finder import PositiveMatch


@dataclass
class NegativeMatch:
    """Represents a hard negative chunk"""

    chunk: Chunk
    similarity_score: float
    reason: str  # Why this is a hard negative


class NegativeMiner:
    """Mines hard negative chunks for training"""

    def __init__(self, config=None):
        """
        Initialize negative miner

        Args:
            config: Fine-tuning configuration
        """
        if config is None:
            from config import get_finetuning_config

            config = get_finetuning_config()

        self.config = config
        self.chunks = []
        self.indexes = {}
        self.filename_map = {}
        self.entity_to_related = {}

    def set_data(self, chunks: List[Chunk], indexes: Dict[str, Dict]):
        """
        Set corpus data for mining

        Args:
            chunks: List of all chunks
            indexes: Dictionary of indexes
        """
        self.chunks = chunks
        self.indexes = indexes

        # Build filename map
        self.filename_map = defaultdict(list)
        for chunk in chunks:
            self.filename_map[chunk.filename].append(chunk)

        # Build entity relationships
        self._build_entity_relationships()

        print("✅ NegativeMiner initialized:")
        print(f"   Total chunks: {len(chunks):,}")
        print(f"   Unique files: {len(self.filename_map):,}")
        print(f"   Entity relationships: {len(self.entity_to_related):,}")

    def _build_entity_relationships(self):
        """
        Build map of entities to related entities based on shared attributes
        """
        # For characters: group by nationality, organization, abilities
        if "character" in self.indexes:
            char_index = self.indexes["character"]

            # Group by attributes
            by_nationality = defaultdict(list)
            by_organization = defaultdict(list)
            by_ability = defaultdict(list)
            by_ajah = defaultdict(list)

            for char_name, char_data in char_index.items():
                # Nationalities
                for nat in char_data.get("nationalities", []):
                    by_nationality[nat].append(char_name)

                # Organizations
                for org in char_data.get("organizations", []):
                    by_organization[org].append(char_name)

                # Special abilities
                for ability in char_data.get("special_abilities", []):
                    by_ability[ability].append(char_name)

                # Ajah
                if "ajah" in char_data and char_data["ajah"]:
                    by_ajah[char_data["ajah"]].append(char_name)

            # Build relationships
            for char_name in char_index.keys():
                related = set()
                char_data = char_index[char_name]

                # Add characters with same nationality
                for nat in char_data.get("nationalities", []):
                    related.update(by_nationality[nat])

                # Add characters in same organizations
                for org in char_data.get("organizations", []):
                    related.update(by_organization[org])

                # Add characters with same abilities
                for ability in char_data.get("special_abilities", []):
                    related.update(by_ability[ability])

                # Add characters in same Ajah
                if "ajah" in char_data and char_data["ajah"]:
                    related.update(by_ajah[char_data["ajah"]])

                # Remove self
                related.discard(char_name)

                self.entity_to_related[char_name] = list(related)

        # For concepts: group by category
        if "concept" in self.indexes:
            concept_index = self.indexes["concept"]

            by_category = defaultdict(list)
            for concept_name, concept_data in concept_index.items():
                for category in concept_data.get("categories", []):
                    by_category[category].append(concept_name)

            for concept_name, concept_data in concept_index.items():
                related = set()
                for category in concept_data.get("categories", []):
                    related.update(by_category[category])

                related.discard(concept_name)
                self.entity_to_related[concept_name] = list(related)

        # Similar for magic and prophecy indexes
        for index_type in ["magic", "prophecy"]:
            if index_type in self.indexes:
                index = self.indexes[index_type]
                by_category = defaultdict(list)

                for entity_name, entity_data in index.items():
                    for category in entity_data.get("categories", []):
                        by_category[category].append(entity_name)

                for entity_name, entity_data in index.items():
                    related = set()
                    for category in entity_data.get("categories", []):
                        related.update(by_category[category])

                    related.discard(entity_name)
                    self.entity_to_related[entity_name] = list(related)

    def find_related_entities(self, entity_name: str, max_related: int = 5) -> List[str]:
        """
        Find entities related to given entity

        Args:
            entity_name: Entity to find relations for
            max_related: Maximum number to return

        Returns:
            List of related entity names
        """
        related = self.entity_to_related.get(entity_name, [])
        if len(related) > max_related:
            related = random.sample(related, max_related)
        return related

    def get_chunks_with_overlapping_mentions(self, positive_chunks: List[Chunk], exclude_chunk_ids: Set[str], max_candidates: int = 50) -> List[Chunk]:
        """
        Find chunks with overlapping entity mentions but not exact matches

        Args:
            positive_chunks: Positive chunks to compare against
            exclude_chunk_ids: Chunk IDs to exclude
            max_candidates: Maximum candidates to return

        Returns:
            List of candidate negative chunks
        """
        # Collect all mentions from positives
        positive_mentions = set()
        for chunk in positive_chunks:
            positive_mentions.update(chunk.character_mentions)
            positive_mentions.update(chunk.concept_mentions)
            positive_mentions.update(chunk.magic_mentions)
            positive_mentions.update(chunk.prophecy_mentions)

        # Find chunks with partial overlap
        candidates = []
        for chunk in self.chunks:
            if chunk.chunk_id in exclude_chunk_ids:
                continue

            chunk_mentions = set(chunk.character_mentions + chunk.concept_mentions + chunk.magic_mentions + chunk.prophecy_mentions)

            overlap = positive_mentions & chunk_mentions

            # Hard negative: has SOME overlap but not exact match
            if 0 < len(overlap) < len(positive_mentions):
                similarity = len(overlap) / len(positive_mentions)
                candidates.append((chunk, similarity, f"Overlapping mentions: {', '.join(list(overlap)[:3])}"))

        # Sort by similarity (want moderate overlap, not too high or too low)
        candidates.sort(key=lambda x: abs(x[1] - 0.5))  # Prefer ~50% overlap

        return [c[0] for c in candidates[:max_candidates]], [c[1] for c in candidates[:max_candidates]], [c[2] for c in candidates[:max_candidates]]

    def mine_hard_negatives(self, question: TestQuestion, positive_matches: List[PositiveMatch], num_negatives: int = None) -> List[NegativeMatch]:
        """
        Mine hard negative chunks for a question

        Args:
            question: Test question
            positive_matches: Positive chunks found for this question
            num_negatives: Number of negatives to return

        Returns:
            List of NegativeMatch objects
        """
        if num_negatives is None:
            num_negatives = self.config.NUM_HARD_NEGATIVES

        # Collect positive chunk IDs to exclude
        positive_ids = {pm.chunk.chunk_id for pm in positive_matches}
        positive_chunks = [pm.chunk for pm in positive_matches]

        negatives = []

        # Strategy 1: Find chunks about related entities
        for topic in question.expected_topics:
            related_entities = self.find_related_entities(topic, max_related=5)

            for related_entity in related_entities:
                # Find filename for related entity
                for index in self.indexes.values():
                    if related_entity in index:
                        filename = index[related_entity].get("filename")
                        if filename and filename in self.filename_map:
                            for chunk in self.filename_map[filename]:
                                if chunk.chunk_id not in positive_ids:
                                    negatives.append(
                                        NegativeMatch(
                                            chunk=chunk,
                                            similarity_score=0.7,  # High similarity (same domain)
                                            reason=f"Related entity: {related_entity} (similar to {topic})",
                                        )
                                    )

        # Strategy 2: Find chunks with overlapping mentions
        overlap_chunks, similarities, reasons = self.get_chunks_with_overlapping_mentions(positive_chunks, positive_ids, max_candidates=30)

        for chunk, sim, reason in zip(overlap_chunks, similarities, reasons):
            negatives.append(NegativeMatch(chunk=chunk, similarity_score=sim, reason=reason))

        # Strategy 3: Same wiki_type but different entity (if wiki chunks)
        if positive_chunks and positive_chunks[0].source == "wiki":
            wiki_type = positive_chunks[0].wiki_type
            for chunk in self.chunks:
                if chunk.chunk_id not in positive_ids and chunk.source == "wiki" and chunk.wiki_type == wiki_type and chunk.filename != positive_chunks[0].filename:
                    negatives.append(NegativeMatch(chunk=chunk, similarity_score=0.5, reason=f"Same type ({wiki_type}) but different entity"))

        # Strategy 4: Random sampling from same source (easier negatives)
        same_source = [c for c in self.chunks if c.source == positive_chunks[0].source and c.chunk_id not in positive_ids]

        if same_source:
            random_sample = random.sample(same_source, min(10, len(same_source)))
            for chunk in random_sample:
                negatives.append(NegativeMatch(chunk=chunk, similarity_score=0.3, reason=f"Random from {chunk.source}"))

        # Sort by similarity (prefer harder negatives)
        negatives.sort(key=lambda n: n.similarity_score, reverse=True)

        # Return diverse set (mix of hard and medium difficulty)
        # Hard: top 60%, Medium: next 30%, Easy: 10%
        hard_count = int(num_negatives * 0.6)
        medium_count = int(num_negatives * 0.3)
        easy_count = num_negatives - hard_count - medium_count

        selected = []
        selected.extend(negatives[:hard_count])  # Hardest
        selected.extend(negatives[len(negatives) // 2 : len(negatives) // 2 + medium_count])  # Medium
        selected.extend(negatives[-easy_count:])  # Easiest

        return selected[:num_negatives]

    def mine_negatives_batch(self, questions: List[TestQuestion], positive_results: Dict[int, List[PositiveMatch]], num_negatives: int = None) -> Dict[int, List[NegativeMatch]]:
        """
        Mine negatives for multiple questions

        Args:
            questions: List of questions
            positive_results: Dict mapping question_id to positive matches
            num_negatives: Number of negatives per question

        Returns:
            Dictionary mapping question_id to negative matches
        """
        results = {}

        print(f"\n⛏️  Mining hard negatives for {len(questions)} questions...")

        for i, question in enumerate(questions, 1):
            positives = positive_results.get(question.question_id, [])

            if not positives:
                print(f"   ⚠️  Q{question.question_id}: No positives found, skipping negatives")
                results[question.question_id] = []
                continue

            negatives = self.mine_hard_negatives(question, positives, num_negatives)
            results[question.question_id] = negatives

            if i % 10 == 0 or i == len(questions):
                avg_negs = sum(len(n) for n in results.values()) / len(results)
                print(f"   Processed {i}/{len(questions)} questions (avg {avg_negs:.1f} negatives/question)")

        return results


if __name__ == "__main__":
    print("✅ NegativeMiner module ready")
