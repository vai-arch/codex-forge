# Plan

## Phase 2A (Current - in progress)

Fine-tune model with SentenceTransformer
Complete training (~2 hours remaining)
Save fine-tuned model

## # TODO Phase 2A-1

tokenize the samples so im sure im not trunkating them.
do this in a generic way so i can reuse it before generating the embeddings in dragons code
in dragon put the tokenization check and use the batching now thats safe
in dragon adjust the size of chunks based on the testings

from tqdm import tqdm  # For progress bar

max_tokens = model.max_seq_length
truncated_count = 0
max_observed = 0

for example in tqdm(train_examples):  # Or your list of sentences/InputExamples
    # Tokenize one sentence (or both if pairs)
    sentence = example.texts[0]  # Adjust if pairs: example.texts[0] + example.texts[1]
    tokens = model.tokenizer(sentence, add_special_tokens=True)
    length = len(tokens['input_ids'])
    max_observed = max(max_observed, length)
    if length > max_tokens:
        truncated_count += 1

print(f"Max observed tokens: {max_observed}")
print(f"Examples that will be truncated: {truncated_count} ({truncated_count/len(train_examples)*100:.2f}%)")

## Phase 2B Only if Phase 2B shows improvement SEE [THRESHOLDS](#realistic-thresholds-without-bm25)

After training finishes:

Load fine-tuned model with SentenceTransformer
Replace VectorStoreManager.embed_chunks() in Dragon's Codex to use fine-tuned model

Re-run existing scripts:

emb_03_embed_all_chunks.py (generates new embeddings)
create_collections.py (new ChromaDB collections)

Test with same questions:

test_baselines_retrieval.py (same 100 questions)
evaluate_retrieval.py (compare scores)

We reuse 90% of Dragon's Codex code. Only change: how embeddings are generated (fine-tuned vs base model).

## Phase 2D FINE-TUNE THE HYPERT PARAMETERS

Run 1 (current): Baseline fine-tuning

batch_size=16, lr=2e-5, epochs=4
Test → Get score X.XX/5
IF X.XX ≥ 2.5 (shows promise):

Run 2-4: Hyperparameter search

Learning rate: [1e-5, 2e-5, 5e-5]
Epochs: [2, 4, 6]
Warmup steps: [50, 100, 200]
This takes:

3-4 more training runs × 2.4 hours = ~10 hours total
Could improve from 2.5 → 3.5+ (hit MVP)
Grid search the key params:
- Learning rate (biggest impact)
- Number of epochs
- Warmup steps

### Realistic thresholds without BM25

#### Success tiers

##### Minimal success: ≥2.0/5 (+43% improvement)

- Shows fine-tuning helps
- Not MVP yet, but validates approach
- Worth hyperparameter tuning

##### Good success: ≥2.5/5 (+79% improvement)

- Significant improvement
- Close to useful
- Definitely do hyperparameter tuning

##### MVP success: ≥3.0/5 (+114% improvement)

- Usable system
- May not need BM25
- Production-ready consideration

##### Excellent: ≥3.5/5 (+150% improvement)

- Clear win
- Exceeds MVP threshold
- Fine-tuning alone solved the problem

#### My prediction: 2.2-2.8/5

- Fine-tuning helps with domain terms
- But won't magically fix all retrieval issues
- BM25 hybrid likely still needed for 3.5+ final score

#### Set threshold

- ≥2.0 = success, proceed with optimization
- <2.0 = fine-tuning doesn't help enough, pivot strategy
