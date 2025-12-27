# Graph-RAG Recipe Adaptation

PhD research project comparing text-based RAG vs graph-based RAG for recipe adaptation using Case-Based Reasoning (CBR).

## Research Question

Does graph-structured recipe representation reduce functional hallucinations in LLM-based recipe adaptation compared to text-only RAG?

## Approach

- **Text-RAG baseline**: Recipes stored as plain text, retrieved and adapted
- **Graph-RAG experiment**: Recipes as bipartite graphs (CookingCAKE-style), graph-aware retrieval and adaptation

## Dataset

Toy dataset (5 recipes): 2 meat/veg pairs (curry, tacos) for proof-of-concept.

Real experiments will use paired recipe datasets (e.g., RecipePairs).

## Schema

Bipartite graph following Bergmann's CookingCAKE paradigm:
- **Ingredient nodes**: `{id, name}`
- **Action nodes**: `{id, verb, step_index, text?, duration?, temperature?}`
- **Edges** (action → ingredient): `{role: input|output, quantity?, unit?, state?}`

See [`data/schema.json`](data/schema.json) for full specification.

## Project Structure

```
data/
  ├── schema.json          # Graph schema definition
  ├── recipes.json         # Recipe dataset (text + graph)
  └── validate_recipes.py  # Validation script
utils/
  └── recipe_utils.py      # Shared utilities
retrieval/
  └── simple_retrieval.py  # Ingredient-overlap retrieval (Jaccard)
generation/
  └── rag_generation.py    # Text-RAG and Graph-RAG pipelines
evaluation/                # Evaluation (TBD)
```

## Usage

```bash
# Validate recipes
python data/validate_recipes.py

# Test retrieval
uv run python -m retrieval.simple_retrieval

# Test generation (requires Ollama running)
uv run python -m generation.rag_generation
```

