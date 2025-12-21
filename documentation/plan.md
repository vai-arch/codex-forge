# Plan

## # TODO Phase 2A-1

tokenize the samples so im sure im not trunkating them.
do this in a generic way so i can reuse it before generating the embeddings in dragons code
in dragon put the tokenization check and use the batching now thats safe
in dragon adjust the size of chunks based on the testings

from tqdm import tqdm  # For progress bar

(Creamos primero el model y de ahi cogemos su max_seq_length)
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

## Phase 2B

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

## Phase 2D FINE-TUNE THE HYPERT PARAMETERS Only if Phase 2B shows improvement SEE [THRESHOLDS](#realistic-thresholds-without-bm25)

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

## Upgrade to v3.0+ and Use the New Trainer ( IF EVERYTHING ELSE GOES RIGHT)

The new training system (v3.0 released ~2024) is much more powerful: supports gradient accumulation properly, better logging, multi-GPU, evaluators, etc.

Upgrade the library:Bashpip install -U sentence-transformers(Current latest as of 2025 is >v3.0)
Switch to the new style (example adaptation of your code):Pythonfrom sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers

### Your model, train_dataset (instead of examples + dataloader), loss already set up

args = SentenceTransformerTrainingArguments(
    output_dir=str(output_path / "checkpoints"),
    num_train_epochs=self.num_epochs,
    per_device_train_batch_size=16,  # Your real batch size
    gradient_accumulation_steps=4,   # ← Now supported! Effective batch=64
    warmup_steps=self.warmup_steps,
    learning_rate=self.learning_rate,
    fp16=True,  # Equivalent to use_amp=True (better name now)
    save_strategy="steps",
    save_steps=self.save_steps,
    save_total_limit=3,
    logging_steps=10,  # Optional: more feedback
    run_name="wheel-of-time-embedding",  # Optional
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # Good for in-batch negatives
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,  # A Hugging Face Dataset with your examples
    loss=train_loss,
)

trainer.train()
You'll need to convert your train_examples to a HF Dataset (easy: from datasets import Dataset; train_dataset = Dataset.from_list([ex.__dict__ for ex in train_examples]) or similar).
Full docs: <https://sbert.net/docs/sentence_transformer/training_overview.html>

This upgrade gives you gradient accumulation properly, plus many other improvements.

## Add callback for execution statistics

from transformers import TrainerCallback

class FinalStatsCallback(TrainerCallback):
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        print("\n🎉 TRAINING COMPLETED!")
        print(f"Final training loss: {state.log_history[-1].get('loss', 'N/A')}")
        # Access full history: state.log_history

### In your trainer

trainer = SentenceTransformerTrainer(
    ...,
    callbacks=[FinalStatsCallback()],  # Add here (list)
)

trainer.train()
