# ADR-0002: Combined Similarity for Retrieval (0.4 Ingredient + 0.6 Name)

Date: 2026-01-05
Status: Accepted

## Context

Initial retrieval used ingredient-only Jaccard similarity. Results showed:
- Very low scores (0.10-0.17)
- Retrieved recipes semantically unrelated (e.g., "swedish meatballs" for "grilled chicken")

Investigation revealed RecipePairs has strong name correspondence:
- "grilled garlic chicken" → "grilled garlic cheese grits"
- "mushroom ravioli" → "mushroom ravioli"

## Decision

Use **combined similarity**:
```
score = 0.4 × ingredient_jaccard + 0.6 × name_jaccard
```

Where:
- `ingredient_jaccard`: Jaccard on ingredient sets
- `name_jaccard`: Jaccard on lowercased name word tokens

Weight 0.6 on names because RecipePairs explicitly pairs recipes with similar names.

## Consequences

**Positive:**
- Scores improved from 0.17 → 0.34+
- Retrieved recipes now semantically related
- Interpretable: can explain weighting to supervisor

**Negative:**
- May overweight name similarity for datasets without naming conventions
- Not using embeddings (semantic similarity)

## Alternatives Considered

- **Embedding-based (sentence-transformers)**: More semantic, but heavy (400MB model), slower
- **BM25/TF-IDF**: Better than raw Jaccard, but more setup
- **Name-only**: Considered but ingredients still matter for grounding
