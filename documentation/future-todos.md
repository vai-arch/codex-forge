# IMPROVE the find_positives with semantic search over the embeddings we already have ???

def find_positives_hybrid(question, all_chunks, indexes, top_k=10):
    """
    Hybrid approach: Semantic search + Keyword validation + Source weighting

    Best of both worlds:
    - Uses semantic similarity (matches production)
    - Validates keywords present (ensures relevance)
    - Weights by source (quality prioritization)
    """
    
    # STEP 1: Semantic search (top 100 candidates)
    # This matches what happens at inference time!
    query_embedding = embed(question.question)
    
    candidates = []
    for chunk_id, chunk in all_chunks.items():
        similarity = cosine_similarity(query_embedding, chunk.embedding)
        
        if similarity >= 0.5:  # Broad initial filter
            candidates.append((chunk_id, chunk, similarity))
    
    # Sort by similarity
    candidates.sort(key=lambda x: x[2], reverse=True)
    candidates = candidates[:100]  # Top 100
    
    # STEP 2: Score each candidate with hybrid scoring
    scored = []
    for chunk_id, chunk, base_sim in candidates:
        
        # A) Semantic similarity (0-1)
        semantic_score = base_sim
        
        # B) Keyword coverage (0-1)
        keywords_found = sum(
            1 for topic in question.expected_topics
            if topic.lower() in chunk.text.lower()
        )
        keyword_score = keywords_found / len(question.expected_topics)
        
        # C) Source weight (1-3)
        source_weight = {
            'wiki_character': 3.0,
            'wiki_concept': 3.0,
            'wiki_magic': 3.0,
            'wiki_prophecy': 3.0,
            'book': 1.0
        }.get(chunk.metadata.get('source'), 1.0)
        
        # D) Length quality (bonus for substantial chunks)
        length_bonus = 1.0
        if len(chunk.text) >= 200:  # Substantial
            length_bonus = 1.2
        elif len(chunk.text) < 100:  # Too short
            length_bonus = 0.8
        
        # FINAL SCORE: Weighted combination
        final_score = (
            semantic_score * 0.6 +      # 60% semantic
            keyword_score * 0.4          # 40% keyword coverage
        ) * source_weight * length_bonus
        
        scored.append({
            'chunk_id': chunk_id,
            'chunk': chunk,
            'final_score': final_score,
            'semantic_sim': semantic_score,
            'keyword_coverage': keyword_score,
            'source_weight': source_weight
        })
    
    # STEP 3: Sort and filter
    scored.sort(key=lambda x: x['final_score'], reverse=True)
    
    # STEP 4: Apply strict threshold
    MIN_THRESHOLD = 1.0  # After weighting
    positives = [
        s for s in scored 
        if s['final_score'] >= MIN_THRESHOLD
    ][:top_k]
    
    return positives

```

---

## 📊 Why Hybrid is BEST

| Benefit | How Hybrid Achieves It |
|---------|------------------------|
| **Matches production** | ✅ Uses semantic similarity as primary signal |
| **Ensures relevance** | ✅ Validates keywords present |
| **Prioritizes quality** | ✅ Weights by source type |
| **Filters junk** | ✅ Length bonus/penalty |
| **Explainable** | ✅ Can debug why chunk was selected |

---

## 🎯 Concrete Example

**Question:** "Who is Rand al'Thor?"

### **Candidate 1: Wiki Character Page**
```

Text: "Rand al'Thor is the main protagonist of the series,
       the Dragon Reborn prophesied to fight the Dark One..."
Source: wiki_character
Length: 250 chars

Semantic: 0.92
Keywords: 3/5 (60%)
Source weight: 3.0
Length bonus: 1.2

Final: (0.92 *0.6 + 0.60* 0.4) *3.0* 1.2 = 2.51 ✅ TOP MATCH

```

### **Candidate 2: Random Book Sentence**
```

Text: "Rand al'Thor nodded at Mat."
Source: book
Length: 30 chars

Semantic: 0.35
Keywords: 1/5 (20%)
Source weight: 1.0
Length bonus: 0.8

Final: (0.35 *0.6 + 0.20* 0.4) *1.0* 0.8 = 0.23 ❌ REJECTED

```

### **Candidate 3: Paraphrase (No Direct Name)**
```

Text: "The shepherd from the Two Rivers discovered his destiny
       as the prophesied savior who would channel the One Power..."
Source: wiki_chapter_summary
Length: 180 chars

Semantic: 0.78
Keywords: 2/5 (40%)
Source weight: 2.0
Length bonus: 1.0

Final: (0.78 *0.6 + 0.40* 0.4) *2.0* 1.0 = 1.25 ✅ GOOD MATCH
