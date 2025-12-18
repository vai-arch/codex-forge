"""
Data Loader
Loads chunks, indexes, and test questions for training pair generation.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from utils import util_files_functions as files


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""

    chunk_id: str
    text: str
    source: str  # "book" or "wiki"
    wiki_type: Optional[str] = None  # "character", "concept", etc.
    filename: Optional[str] = None  # Wiki page filename
    temporal_order: Optional[int] = None  # Book order
    character_mentions: List[str] = None
    concept_mentions: List[str] = None
    magic_mentions: List[str] = None
    prophecy_mentions: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        """Initialize empty lists if None"""
        if self.character_mentions is None:
            self.character_mentions = []
        if self.concept_mentions is None:
            self.concept_mentions = []
        if self.magic_mentions is None:
            self.magic_mentions = []
        if self.prophecy_mentions is None:
            self.prophecy_mentions = []
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def from_dict(cls, data: Dict, chunk_id: str = None):
        """Create Chunk from dictionary"""
        return cls(
            chunk_id=chunk_id or data.get("chunk_id", ""),
            text=data["text"],
            source=data["source"],
            wiki_type=data.get("wiki_type"),
            filename=data.get("filename"),
            temporal_order=data.get("temporal_order"),
            character_mentions=data.get("character_mentions", []),
            concept_mentions=data.get("concept_mentions", []),
            magic_mentions=data.get("magic_mentions", []),
            prophecy_mentions=data.get("prophecy_mentions", []),
            metadata=data,
        )


@dataclass
class TestQuestion:
    """Represents a test question"""

    question_id: int
    question: str
    category: str
    difficulty: str
    expected_topics: List[str]
    temporal_limit: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict):
        """Create TestQuestion from dictionary"""
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            category=data["category"],
            difficulty=data["difficulty"],
            expected_topics=data.get("expected_topics", []),
            temporal_limit=data.get("temporal_limit"),
        )


class DataLoader:
    """Loads and manages corpus data"""

    def __init__(self, paths=None):
        """
        Initialize data loader

        Args:
            paths: Paths instance (if None, creates new one)
        """
        if paths is None:
            from src.paths import get_paths

            paths = get_paths()

        self.paths = paths
        self.chunks_cache = {}
        self.indexes_cache = {}

        # Will be populated by load_all()
        self.chunks = None
        self.character_index = None
        self.concept_index = None
        self.magic_index = None
        self.prophecy_index = None
        self.filename_map = None

    def load_chunks(self, chunk_file: Path):
        """Load chunks from a single JSONL file"""
        chunks = []
        with open(chunk_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)

                # Handle missing filename for book chunks
                if "filename" not in data:
                    data["filename"] = None  # Explicit None

                chunk = Chunk.from_dict(data, chunk_id=f"{chunk_file.stem}_{i}")
                chunks.append(chunk)

        return chunks

    def load_all_chunks(self, chunks_dir: Path = None) -> List[Chunk]:
        """
        Load all chunks from a directory

        Args:
            chunks_dir: Directory containing .jsonl chunk files

        Returns:
            Combined list of all chunks
        """
        all_chunks = []
        chunk_files = files.find_files_in_folder(path_folder=self.paths.CHUNKS_PATH, extension="jsonl")

        for chunk_file in chunk_files:
            chunks = self.load_chunks(chunk_file)
            all_chunks.extend(chunks)
            print(f"  ✅ {chunk_file.name}: {len(chunks):,} chunks")

        print(f"\n✅ Total chunks loaded: {len(all_chunks):,}")
        return all_chunks

    def load_index(self, index_file: Path, cache: bool = True) -> Dict:
        """
        Load an index file (character, concept, magic, prophecy)

        Args:
            index_file: Path to .json index file
            cache: Whether to cache loaded index

        Returns:
            Dictionary of entities
        """
        if cache and str(index_file) in self.indexes_cache:
            return self.indexes_cache[str(index_file)]

        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        if cache:
            self.indexes_cache[str(index_file)] = index

        return index

    def load_all_indexes(self, indexes_dir: Path = None) -> Dict[str, Dict]:
        """
        Load all index files from a directory

        Args:
            indexes_dir: Directory containing .json index files

        Returns:
            Dictionary mapping index type to index data
        """
        indexes = {}

        index_files = files.find_files_in_folder(path_folder=self.paths.INDEXES_PATH, extension=".json")

        for index_file in index_files:
            # Extract index type from filename (e.g., "character_index.json" -> "character")
            index_type = index_file.stem.replace("_index", "")
            index_data = self.load_index(index_file)
            indexes[index_type] = index_data
            print(f"  ✅ {index_type}: {len(index_data):,} entities")

        return indexes

    def load_test_questions(self, questions_file: Path = None) -> List[TestQuestion]:
        """
        Load test questions

        Args:
            questions_file: Path to test_questions.json

        Returns:
            List of TestQuestion objects
        """
        data = files.load_json_from_file(questions_file)
        return [TestQuestion.from_dict(q) for q in data.get("questions", [])]

    def get_chunks_by_filename(self, chunks: List[Chunk], filename: str) -> List[Chunk]:
        """
        Filter chunks by filename

        Args:
            chunks: List of all chunks
            filename: Filename to match

        Returns:
            List of matching chunks
        """
        return [c for c in chunks if c.filename == filename]

    def get_chunks_by_entity(self, chunks: List[Chunk], entity_name: str, entity_type: str = None) -> List[Chunk]:
        """
        Find chunks that mention an entity

        Args:
            chunks: List of all chunks
            entity_name: Entity name to search for
            entity_type: Type of entity ("character", "concept", "magic", "prophecy")

        Returns:
            List of matching chunks
        """
        matching_chunks = []

        for chunk in chunks:
            # Check appropriate mention list based on entity_type
            if entity_type == "character":
                if entity_name in chunk.character_mentions:
                    matching_chunks.append(chunk)
            elif entity_type == "concept":
                if entity_name in chunk.concept_mentions:
                    matching_chunks.append(chunk)
            elif entity_type == "magic":
                if entity_name in chunk.magic_mentions:
                    matching_chunks.append(chunk)
            elif entity_type == "prophecy":
                if entity_name in chunk.prophecy_mentions:
                    matching_chunks.append(chunk)
            else:
                # Search all mention types
                if entity_name in chunk.character_mentions or entity_name in chunk.concept_mentions or entity_name in chunk.magic_mentions or entity_name in chunk.prophecy_mentions:
                    matching_chunks.append(chunk)

        return matching_chunks

    def build_filename_to_chunks_map(self, chunks: List[Chunk]) -> Dict[str, List[Chunk]]:
        """
        Build a map from filename to chunks for fast lookup

        Args:
            chunks: List of all chunks

        Returns:
            Dictionary mapping filename to list of chunks
        """
        filename_map = {}
        for chunk in chunks:
            if chunk.filename not in filename_map:
                filename_map[chunk.filename] = []
            filename_map[chunk.filename].append(chunk)

        return filename_map

    def get_statistics(self, chunks: List[Chunk]) -> Dict:
        """
        Get statistics about loaded chunks

        Args:
            chunks: List of chunks

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_chunks": len(chunks),
            "by_source": {},
            "by_wiki_type": {},
            "with_temporal_order": 0,
            "avg_text_length": 0,
            "total_character_mentions": 0,
            "total_concept_mentions": 0,
            "total_magic_mentions": 0,
            "total_prophecy_mentions": 0,
        }

        for chunk in chunks:
            # Count by source
            stats["by_source"][chunk.source] = stats["by_source"].get(chunk.source, 0) + 1

            # Count by wiki_type
            if chunk.wiki_type:
                stats["by_wiki_type"][chunk.wiki_type] = stats["by_wiki_type"].get(chunk.wiki_type, 0) + 1

            # Count temporal
            if chunk.temporal_order is not None:
                stats["with_temporal_order"] += 1

            # Sum text lengths
            stats["avg_text_length"] += len(chunk.text)

            # Count mentions
            stats["total_character_mentions"] += len(chunk.character_mentions)
            stats["total_concept_mentions"] += len(chunk.concept_mentions)
            stats["total_magic_mentions"] += len(chunk.magic_mentions)
            stats["total_prophecy_mentions"] += len(chunk.prophecy_mentions)

        # Calculate average
        if len(chunks) > 0:
            stats["avg_text_length"] = stats["avg_text_length"] / len(chunks)

        return stats

    def load_all(self):
        """
        Load all data and store as attributes
        This populates: chunks, indexes, filename_map
        """
        print("\n📂 Loading all data...")

        # Load chunks
        print("\n1️⃣ Loading chunks...")
        self.chunks = self.load_all_chunks()

        # Load indexes
        print("\n2️⃣ Loading indexes...")
        indexes = self.load_all_indexes()
        self.character_index = indexes.get("character", {})
        self.concept_index = indexes.get("concept", {})
        self.magic_index = indexes.get("magic", {})
        self.prophecy_index = indexes.get("prophecy", {})

        # Build filename map
        print("\n3️⃣ Building filename map...")
        self.filename_map = self.build_filename_to_chunks_map(self.chunks)
        print(f"  ✅ Mapped {len(self.filename_map):,} unique filenames")

        print("\n✅ All data loaded!\n")
