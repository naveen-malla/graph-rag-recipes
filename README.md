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
  └── recipe_utils.py      # Shared utilities (load_recipes, get_ingredients)
retrieval/
  └── simple_retrieval.py  # Ingredient-overlap based retrieval (Jaccard)
generation/                # Adaptation implementations (TBD)
evaluation/                # Evaluation metrics (TBD)
```

## Setup

```bash
# Install UV (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate

# Validate recipes
python data/validate_recipes.py

# Test retrieval system
uv run python -m retrieval.simple_retrieval
```

## Retrieval System

Simple ingredient-overlap based retrieval using **Jaccard similarity**:
- Extract ingredient sets from recipe graphs
- Calculate similarity: `|intersection| / |union|`
- Return top-k most similar recipes

Example output:
```
Query: Chicken Curry
  → Chickpea Curry: 0.667 (66.7% ingredient overlap)
  → Beef Tacos: 0.222 (22.2% overlap)
```

## Validation

The validation script checks:
- ✅ Ingredient coverage: Graph ingredients appear in text
- ✅ Action coverage: Action verbs appear in text
- ✅ Graph structure: Valid bipartite graph (action → ingredient only)
- ✅ Edge validity: Correct roles (input/output), valid node references

