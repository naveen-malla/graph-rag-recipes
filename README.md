# Graph-RAG Recipe Adaptation

PhD research project comparing text-based RAG vs graph-based RAG for recipe adaptation using Case-Based Reasoning (CBR).

## Research Question

Does graph-structured recipe representation reduce functional hallucinations in LLM-based recipe adaptation compared to text-only RAG?

## Approach

### Text-RAG (Baseline)
Recipes represented as plain text. LLM receives unstructured context:
```
Recipe: Chickpea Curry
Text: 1. Chop 2 medium onions finely. 2. Heat 2 tbsp oil...
```

### Graph-RAG (Experiment)
Recipes as bipartite graphs (CookingCAKE-style). LLM receives structured context with explicit action→ingredient relations:
```
Recipe: Chickpea Curry
Structure:
  Step 1. chop → onion (2 medium raw) [input]
  Step 2. heat → oil (2 tbsp) [input]
  Step 3. sauté → onion (chopped) [input]
  Step 4. cook → chickpeas (400 g) [input]
  Step 5. add → tomato sauce (400 g) [input], curry powder (1 tbsp) [input]
```

**Hypothesis**: Explicit structure reduces hallucinations by making ingredient-action relationships unambiguous.

## Implementation

**Pipeline**: Retrieval → Generation → Evaluation

1. **Retrieval**: Ingredient-overlap similarity (Jaccard)
2. **Generation**: Ollama (llama3.2:3b) with formatted context
3. **Evaluation**: 
   - Constraint violation detection (forbidden ingredients)
   - Ingredient consistency check (with substitution allowlist)

## Dataset

Toy dataset: 5 recipes (2 meat/veg pairs) for proof-of-concept.

## Schema

Bipartite graph following Bergmann's CookingCAKE paradigm:
- **Ingredient nodes**: `{id, name}`
- **Action nodes**: `{id, verb, step_index}`
- **Edges** (action → ingredient): `{role: input|output, quantity?, unit?, state?}`

See [`data/schema.json`](data/schema.json) for specification.

## Project Structure

```
data/
  ├── schema.json              # Graph schema definition
  ├── recipes.json             # Recipe dataset (text + graph)
  └── validate_recipes.py      # Validation script
utils/
  └── recipe_utils.py          # Shared utilities (load_recipes, get_ingredients)
retrieval/
  └── simple_retrieval.py      # Ingredient-overlap retrieval (Jaccard)
generation/
  └── rag_generation.py        # Text-RAG and Graph-RAG pipelines
evaluation/
  └── hallucination_checker.py # Constraint violation & ingredient consistency
run_experiment.py              # Main experiment runner
results/                       # Experiment outputs (JSON)
```

## Usage

```bash
# Validate recipes
python data/validate_recipes.py

# Test retrieval
uv run python -m retrieval.simple_retrieval

# Test generation (requires Ollama running)
uv run python -m generation.rag_generation

# Run full experiment
uv run python run_experiment.py
```

## Evaluation Metrics

**Primary**: Constraint violation (forbidden ingredients in output)
**Secondary**: Ingredient consistency (uses source ingredients or valid substitutions)

Example substitution allowlist:
```python
{"chicken": ["chickpeas", "tofu"], "beef": ["black beans", "lentils"]}
```

