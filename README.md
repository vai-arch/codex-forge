# Generic Embedding Fine-Tuning Framework

A reusable framework for fine-tuning embedding models on domain-specific corpora, proven on the Dragon's Codex RAG system.

## 🎯 Philosophy

Based on Dragon's Codex development methodology:

- **Analysis first, code second** - Understand the problem before building
- **Data-driven decisions** - Measure everything, iterate systematically  
- **One problem at a time** - Fix issues systematically with validation
- **Reusable architecture** - Build once, use everywhere

## 🚀 Quick Start

### Prerequisites

```bash
# Required
Python 3.8+
pip install -r requirements.txt

# For GPU training (recommended)
CUDA-capable GPU with 4GB+ VRAM

# For CPU training (slower but works)
Nothing special needed
```

### Installation

```bash
# Clone or copy the framework
git clone <your-repo>
cd codex-forge

# Install dependencies
pip install -r requirements.txt

# Configure for your domain
cp .env.example .env
# Edit .env with your settings
```

### Basic Usage

```bash
# Step 1: Generate training pairs
python scripts/01_generate_training_pairs.py

# Step 2: Fine-tune model (TODO - next phase)
python scripts/02_finetune_model.py

# Step 3: Evaluate results (TODO - next phase)
python scripts/03_evaluate_model.py

# Step 4: Compare baseline vs fine-tuned (TODO - next phase)
python scripts/04_compare_results.py
```

## 📊 Proven Results (Dragon's Codex)

| Phase | Retrieval Score | Status |
|-------|----------------|--------|
| Baseline (Semantic Only) | 1.40/5 | ❌ Failed MVP |
| BM25 Hybrid | ~2.0-2.5/5 | ⚠️  Better but not enough |
| Fine-Tuned (Expected) | 3.5-4.0/5 | ✅ Target MVP |

**The Problem:** Generic embeddings don't understand domain-specific vocabulary:

- "saidin" → Not recognized as related to "One Power", "channeling"
- "Dragon Reborn" → Not linked to "Rand al'Thor", "Lews Therin"
- Proper nouns treated as random character sequences

**The Solution:** Fine-tune embeddings on domain corpus with contrastive learning.

## 🏗️ Architecture

```
Input Data:
├── chunks.jsonl        # Text chunks with entity mentions
├── indexes/            # Entity indexes (character, concept, etc.)
└── test_questions.json # Queries with expected topics

Process:
├── 01_generate_training_pairs.py
│   ├── Find positive chunks (match expected topics)
│   ├── Mine hard negatives (similar but wrong)
│   └── Synthesize additional queries
│
├── 02_finetune_model.py (TODO)
│   ├── Load base model
│   ├── Contrastive learning (pull positives, push negatives)
│   └── Save fine-tuned model
│
├── 03_evaluate_model.py (TODO)
│   ├── Re-embed chunks with new model
│   ├── Run test questions
│   └── Score results
│
└── 04_compare_results.py (TODO)
    └── Baseline vs Fine-tuned comparison

Output:
├── training_pairs.jsonl    # Generated training data
├── finetuned_model/        # Saved model weights
└── evaluation_results/     # Before/after metrics
```

## 📁 Directory Structure

```
embedding-finetuning-framework/
├── src/
│   ├── config.py                    # Configuration management
│   ├── paths.py                     # Path configuration
│   ├── data_loader.py               # Load chunks/indexes/questions
│   ├── positive_finder.py           # Find positive chunks
│   ├── negative_miner.py            # Mine hard negatives
│   └── training_pair_generator.py   # Orchestrate full pipeline
│
├── scripts/
│   ├── 01_generate_training_pairs.py ✅ COMPLETE
│   ├── 02_finetune_model.py         ⏳ TODO
│   ├── 03_evaluate_model.py         ⏳ TODO
│   └── 04_compare_results.py        ⏳ TODO
│
├── data/                            # Created automatically
│   ├── corpus/
│   │   ├── chunks/                  # Your chunked corpus
│   │   └── indexes/                 # Entity indexes
│   ├── processed/
│   │   └── pairs/                   # Generated training pairs
│   ├── embeddings/
│   │   ├── base/                    # Base model embeddings
│   │   └── finetuned/               # Fine-tuned embeddings
│   ├── models/
│   │   ├── checkpoints/             # Training checkpoints
│   │   └── finetuned/               # Final model
│   └── evaluation/
│       ├── test_sets/               # Test questions
│       └── results/                 # Evaluation results
│
├── .env.example                     # Example configuration
├── requirements.txt
└── README.md                        # This file
```

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Project paths
PROJECT_ROOT=/path/to/your/project

# Model settings
BASE_MODEL=nomic-embed-text
MODEL_BACKEND=ollama  # ollama, huggingface, openai
EMBEDDING_DIM=768

# Training hyperparameters
LEARNING_RATE=2e-5
BATCH_SIZE_GPU=32
BATCH_SIZE_CPU=8
NUM_EPOCHS=3
MAX_SEQ_LENGTH=512

# Training pair generation
NUM_HARD_NEGATIVES=3
POSITIVES_PER_QUERY=3
SYNTHETIC_EXPANSION_FACTOR=5
MIN_TOPIC_COVERAGE=0.5

# Evaluation
RECALL_K_VALUES=[1,3,5,10]
```

### Configuration via Python

```python
from src.config import get_finetuning_config

config = get_finetuning_config()
print(config.LEARNING_RATE)  # 2e-5
```

## 📝 Input Data Format

### 1. Chunks (chunks/*.jsonl)

```json
{
  "source": "wiki",
  "wiki_type": "character",
  "filename": "Rand_al'Thor.txt",
  "text": "Rand al'Thor is the Dragon Reborn...",
  "character_mentions": ["Rand al'Thor", "Dragon Reborn"],
  "concept_mentions": ["Two Rivers", "One Power"],
  "magic_mentions": ["channeling", "saidin"],
  "prophecy_mentions": []
}
```

**Required fields:**

- `text` - The chunk content
- `filename` - Links to index entries
- `*_mentions` - Entity mentions (character, concept, magic, prophecy)

**Optional fields:**

- `source` - "book" or "wiki"
- `wiki_type` - Type classification
- `temporal_order` - For temporal filtering

### 2. Indexes (indexes/*_index.json)

```json
{
  "Rand al'Thor": {
    "primary_name": "Rand al'Thor",
    "filename": "Rand_al'Thor.txt",
    "aliases": ["Dragon Reborn", "Lews Therin Reborn"],
    "nationalities": ["Aiel", "Two Rivers"],
    "special_abilities": ["ta_veren", "channeler"]
  }
}
```

**Required fields:**

- `filename` - Must match chunk filenames
- `aliases` - Alternative names for the entity

### 3. Test Questions (test_questions.json)

```json
{
  "questions": [
    {
      "question_id": 1,
      "question": "Who is Rand al'Thor?",
      "category": "character",
      "difficulty": "easy",
      "temporal_limit": null,
      "expected_topics": ["Dragon Reborn", "Two Rivers", "channeler"]
    }
  ]
}
```

**Required fields:**

- `question` - Query text
- `expected_topics` - Topics that should be in retrieved chunks
- `category` - Type of question (helps with negative mining)

## 🔍 How It Works

### Step 1: Generate Training Pairs

**Positive Finding:**

1. Match expected topics to chunk content
2. Use index aliases (e.g., "Dragon Reborn" → Rand al'Thor)
3. Check entity mentions
4. Score by topic coverage

**Hard Negative Mining:**

1. Find chunks about related entities (e.g., Perrin for Rand queries)
2. Find chunks with overlapping mentions
3. Find same type but different entity
4. Mix hard/medium/easy negatives

**Query Synthesis:**

- Expand test questions with templates
- Generate from chunk entity mentions
- 5x expansion by default (100 questions → 500 queries)

### Step 2: Fine-Tune Model (TODO - Next Phase)

Uses **Multiple Negatives Ranking Loss**:

- Pull positive chunks closer to queries in embedding space
- Push negative chunks away
- Learn domain-specific semantic relationships

### Step 3: Evaluate (TODO - Next Phase)

- Re-embed all chunks with fine-tuned model
- Run same 100 test questions
- Score using same rubric
- Compare to baseline

## 📈 Expected Improvements

Based on Dragon's Codex results:

**Before Fine-Tuning:**

- Query: "What is saidin?"
- Retrieved: Generic chunks, poor semantic matching
- Score: 1/5 (missing expected topics)

**After Fine-Tuning:**

- Query: "What is saidin?"
- Retrieved: Chunks about male One Power, taint, madness
- Score: 4/5 (all expected topics covered)

**Improvement Areas:**

- ✅ Domain vocabulary recognition
- ✅ Entity relationship understanding
- ✅ Semantic similarity for domain concepts
- ✅ Alias/synonym matching

## 🎓 Adapting to Your Domain

This framework is generic and reusable. To adapt:

1. **Prepare Your Corpus:**
   - Chunk your text
   - Extract entity mentions
   - Create indexes with aliases

2. **Create Test Questions:**
   - 100+ questions covering your domain
   - Include expected topics for each
   - Mix difficulties (easy/medium/hard)

3. **Configure:**
   - Set paths in `.env`
   - Adjust hyperparameters if needed
   - Set domain-specific thresholds

4. **Run Pipeline:**

   ```bash
   python scripts/01_generate_training_pairs.py
   python scripts/02_finetune_model.py
   python scripts/03_evaluate_model.py
   python scripts/04_compare_results.py
   ```

## 🔧 Development Status

- ✅ **Phase 1: Training Pair Generation** - COMPLETE
  - Data loader
  - Positive finder
  - Hard negative miner
  - Query synthesizer
  - Main script

- ⏳ **Phase 2: Model Fine-Tuning** - TODO
  - Model loader (model-agnostic)
  - Contrastive trainer
  - Checkpointing
  - Loss functions

- ⏳ **Phase 3: Evaluation** - TODO
  - Re-embedding
  - Retrieval with new model
  - Scoring
  - Comparison

- ⏳ **Phase 4: Analysis & Reporting** - TODO
  - Detailed metrics
  - Improvement analysis
  - Recommendations

## 📚 References

- **Dragon's Codex Methodology:** See `DEVELOPMENT_METHODOLOGY_v05.md`
- **Contrastive Learning:** Multiple Negatives Ranking Loss
- **Hard Negative Mining:** Based on entity relationships and co-occurrence

## 🤝 Contributing

This framework follows the Dragon's Codex development methodology:

1. **NO CODE until analysis is complete**
2. **Data-first approach** - Always examine real examples
3. **One problem at a time** - Fix systematically
4. **Measure everything** - Track improvements

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built from lessons learned developing Dragon's Codex, a RAG system for the Wheel of Time series.

---

**Status:** Phase 1 Complete ✅ | Next: Phase 2 (Fine-Tuning) ⏳
