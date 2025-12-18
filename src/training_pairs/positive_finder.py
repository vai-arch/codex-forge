"""
Positive Chunk Finder
Finds positive training examples by matching expected topics to chunks.
"""

from typing import Dict, List, Tuple

from tqdm import tqdm

from src.config import get_finetuning_config  # ⬅️ CHANGED THIS
from src.training_pairs.data_loader import Chunk, DataLoader, TestQuestion


class PositiveFinder:
    """
    Finds positive chunks for training pairs.
    Uses multiple strategies:
    1. Filename-based (for wiki chunks with entity pages)
    2. Entity mention-based (for all chunks including books)
    3. Text matching with alias support
    """

    def __init__(self, data_loader: DataLoader, config=None):
        """
        Initialize PositiveFinder

        Args:
            data_loader: DataLoader instance with loaded chunks and indexes
            config: Config object (if None, loads from get_finetuning_config())
        """
        if config is None:
            config = get_finetuning_config()  # ⬅️ CHANGED THIS

        self.config = config
        # ... rest stays the same
        self.data_loader = data_loader
        self.chunks = data_loader.chunks
        self.filename_map = data_loader.filename_map

        # Indexes for alias lookup
        self.character_index = data_loader.character_index
        self.concept_index = data_loader.concept_index
        self.magic_index = data_loader.magic_index
        self.prophecy_index = data_loader.prophecy_index

        print(f"✅ PositiveFinder initialized with {len(self.chunks):,} chunks")

    def find_entity_aliases(self, entity_name: str) -> List[str]:
        """
        Find all aliases for an entity across all indexes

        Args:
            entity_name: Entity name to look up

        Returns:
            List of aliases (including the entity name itself)
        """
        aliases = [entity_name]  # Always include the entity itself

        # Check character index
        if entity_name in self.character_index:
            char_data = self.character_index[entity_name]
            if isinstance(char_data, dict) and "aliases" in char_data:
                aliases.extend(char_data["aliases"])

        # Check concept index
        if entity_name in self.concept_index:
            concept_data = self.concept_index[entity_name]
            if isinstance(concept_data, dict) and "aliases" in concept_data:
                aliases.extend(concept_data["aliases"])

        # Check magic index
        if entity_name in self.magic_index:
            magic_data = self.magic_index[entity_name]
            if isinstance(magic_data, dict) and "aliases" in magic_data:
                aliases.extend(magic_data["aliases"])

        # Check prophecy index
        if entity_name in self.prophecy_index:
            prophecy_data = self.prophecy_index[entity_name]
            if isinstance(prophecy_data, dict) and "aliases" in prophecy_data:
                aliases.extend(prophecy_data["aliases"])

        # Remove duplicates, preserve order
        seen = set()
        unique_aliases = []
        for alias in aliases:
            if alias.lower() not in seen:
                seen.add(alias.lower())
                unique_aliases.append(alias)

        return unique_aliases

    def get_entity_filename(self, entity_name: str, category: str = None) -> str:
        """
        Get the wiki filename for an entity

        Args:
            entity_name: Entity name
            category: Optional category hint (character, concept, magic, prophecy)

        Returns:
            Filename (e.g., "Rand_al'Thor.txt") or None if not found
        """
        # Normalize entity name to filename format
        # "Rand al'Thor" -> "Rand_al'Thor.txt"
        normalized = entity_name.replace(" ", "_") + ".txt"

        # Check if this filename exists in our filename_map
        if normalized in self.filename_map:
            return normalized

        # Try without apostrophes
        normalized_no_apos = entity_name.replace(" ", "_").replace("'", "") + ".txt"
        if normalized_no_apos in self.filename_map:
            return normalized_no_apos

        # Try all aliases
        aliases = self.find_entity_aliases(entity_name)
        for alias in aliases:
            alias_filename = alias.replace(" ", "_") + ".txt"
            if alias_filename in self.filename_map:
                return alias_filename

            alias_filename_no_apos = alias.replace(" ", "_").replace("'", "") + ".txt"
            if alias_filename_no_apos in self.filename_map:
                return alias_filename_no_apos

        return None

    def score_chunk_for_topics(self, chunk: Chunk, expected_topics: List[str], query_text: str = None) -> Tuple[float, List[str], Dict]:
        """
        Score a chunk for how well it matches expected topics

        Scoring weights:
        - Text match: 1.0
        - Alias match: 2.0 (critical for domain terms!)
        - Entity mention: 0.5
        - Filename partial match: 2.0
        - Filename EXACT match: +2.0 (prioritizes bio pages!)

        Returns:
            tuple: (score, matched_topics, details)
        """
        score = 0.0
        matched_topics = []
        details = {"text_matches": [], "alias_matches": [], "mention_matches": [], "filename_matches": []}

        chunk_text = chunk.text.lower()

        # Get all entity mentions from chunk
        all_mentions = chunk.character_mentions + chunk.concept_mentions + chunk.magic_mentions + chunk.prophecy_mentions
        all_mentions_lower = [m.lower() for m in all_mentions]

        # Score each expected topic
        for topic in expected_topics:
            topic_lower = topic.lower()
            topic_score = 0.0

            # 1. Direct text match (weight: 1.0)
            if topic_lower in chunk_text:
                topic_score += 1.0
                details["text_matches"].append(topic)

            # 2. Alias match (weight: 2.0) - CRITICAL!
            aliases = self.find_entity_aliases(topic)
            for alias in aliases:
                if alias.lower() in chunk_text:
                    topic_score += 2.0
                    details["alias_matches"].append(f"{topic} (alias: {alias})")
                    break  # Only count once per topic

            # 3. Entity mention match (weight: 0.5)
            if topic_lower in all_mentions_lower:
                topic_score += 0.5
                details["mention_matches"].append(topic)

            # 4. Filename partial match (weight: 2.0)
            if chunk.filename:
                filename_entity = chunk.filename.replace(".txt", "").replace("_", " ").lower()

                if topic_lower in filename_entity or filename_entity in topic_lower:
                    topic_score += 2.0
                    details["filename_matches"].append(topic)

            # 5. Filename EXACT match bonus (weight: +2.0)
            if chunk.filename:
                # Normalize for exact comparison
                topic_normalized = topic.lower().replace(" ", "_").replace("'", "")
                filename_normalized = chunk.filename.lower().replace(".txt", "").replace("'", "")

                # EXACT match = bio/definition page
                if topic_normalized == filename_normalized:
                    topic_score += 2.0
                    details["filename_matches"].append(f"{topic} (EXACT)")

                # Check aliases too
                aliases = self.find_entity_aliases(topic)
                for alias in aliases:
                    alias_normalized = alias.lower().replace(" ", "_").replace("'", "")
                    if alias_normalized == filename_normalized:
                        topic_score += 2.0
                        details["filename_matches"].append(f"{topic} via {alias} (EXACT)")
                        break

            # If any score for this topic, add to matched
            if topic_score > 0:
                score += topic_score
                matched_topics.append(topic)

        # Additional boost if query text provided
        if query_text:
            query_lower = query_text.lower()
            query_words = set(query_lower.split())
            chunk_words = set(chunk_text.split())

            # Bonus for query word overlap
            overlap = len(query_words & chunk_words)
            if overlap > 0:
                score += overlap * 0.1

        return score, matched_topics, details

    def find_positives(self, question: TestQuestion, top_k: int = None, min_score: float = None) -> List[Dict]:
        """
        Find positive chunks for question

        Args:
            question: TestQuestion object
            top_k: Number of results to return
            min_score: Minimum score threshold (uses config default if None)

        Returns:
            List of dicts with chunk, score, matched_topics, match_details
        """
        if min_score is None:
            min_score = self.config.MIN_POSITIVE_SCORE

        if top_k is None:
            top_k = self.config.POSITIVES_PER_QUERY

        matches = []

        # Strategy 1: Filename-based lookup (for wiki chunks)
        for topic in question.expected_topics:
            filename = self.get_entity_filename(topic, question.category)

            if filename and filename in self.filename_map:
                for chunk in self.filename_map[filename]:
                    score, matched, details = self.score_chunk_for_topics(chunk, question.expected_topics, question.question)

                    if score >= min_score:
                        matches.append({"chunk": chunk, "score": score, "matched_topics": matched, "match_details": details})

        # Strategy 2: Search all chunks if not enough matches
        if len(matches) < top_k:
            for chunks in self.filename_map.values():
                for chunk in chunks:
                    score, matched, details = self.score_chunk_for_topics(chunk, question.expected_topics, question.question)

                    if score >= min_score:
                        # Avoid duplicates
                        chunk_ids = [m["chunk"].chunk_id for m in matches]
                        if chunk.chunk_id not in chunk_ids:
                            matches.append({"chunk": chunk, "score": score, "matched_topics": matched, "match_details": details})

        # Sort by score (descending)
        matches.sort(key=lambda m: m["score"], reverse=True)

        # Return top_k results
        return matches[:top_k]

    def find_positives_batch(self, questions: List[TestQuestion], top_k: int = None) -> Dict[int, List[Dict]]:
        """
        Find positives for multiple questions in batch

        Args:
            questions: List of TestQuestion objects
            top_k: Number of results per question

        Returns:
            Dict mapping question_id to list of positive matches
        """
        results = {}

        for question in tqdm(questions, desc="Finding positives"):
            positives = self.find_positives(question, top_k)
            results[question.question_id] = positives

        return results

    def get_stats(self) -> Dict:
        """Get statistics about positive finding"""
        return {
            "total_chunks": len(self.chunks),
            "filenames_mapped": len(self.filename_map),
            "character_entities": len(self.character_index),
            "concept_entities": len(self.concept_index),
            "magic_entities": len(self.magic_index),
            "prophecy_entities": len(self.prophecy_index),
        }
