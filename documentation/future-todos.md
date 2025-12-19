# TO-DO for the future

## Upgrade to v3.0+ and Use the New Trainer

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
