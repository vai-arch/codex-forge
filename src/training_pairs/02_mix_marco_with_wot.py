import json
import random
from pathlib import Path

from datasets import disable_caching, load_dataset

from src.paths import get_paths
from utils import files

paths = get_paths()

files.remove_file(paths.FILE_TRAINING_PAIRS_MIXED)


print("Loading MS MARCO hard triplets...")

# Best options (large, high-quality with filtered hard negatives):
# ~11.6M triplets – excellent for retrieval

disable_caching()  # Put before load_dataset
dataset = load_dataset("sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1", "triplet-hard", split="train", streaming=True)

# Load WoT (your existing code)
wot_pairs = []
# fmt: off
with open(paths.FILE_TRAINING_PAIRS, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        wot_pairs.append({
            "query": item["query"], 
            "positive_text": item["positive_text"],
            "negative_texts": item["negative_texts"],  
            "source": "wot"
        })

# Calculate stats
total_negatives_wot = sum(len(p["negative_texts"]) for p in wot_pairs)
avg_negatives_per_pair = total_negatives_wot / len(wot_pairs)

print(f"WoT pairs: {len(wot_pairs):,}")
print(f"Avg negatives per WoT pair: {avg_negatives_per_pair:.2f}")

general_pairs = []
for i, item in enumerate(dataset):
    if i >= len(wot_pairs) * (1 + avg_negatives_per_pair):
        break
    general_pairs.append({
        "query": item["query"], 
        "positive_text": item["positive"], 
        "negative_texts": [item["negative"]],
        "source": "general"})
# fmt: on

print(f"Loaded {len(general_pairs)} general pairs")

# Merge and shuffle
merged = general_pairs + wot_pairs
random.shuffle(merged)

# Save
output = Path(paths.FILE_TRAINING_PAIRS_MIXED)
with open(output, "w", encoding="utf-8") as f:
    for item in merged:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(merged)} mixed pairs")
