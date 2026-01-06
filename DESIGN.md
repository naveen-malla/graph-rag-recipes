# DESIGN

## Purpose
Experimental codebase for comparing **Text-RAG vs Graph-RAG** approaches to **recipe adaptation** in a **CBR (Case-Based Reasoning)** framing. Primary hypothesis: explicit grounding constraints reduce functional hallucinations during LLM-based adaptation.

## Non-goals
- General-purpose recipe generation
- Production-ready recipe system
- Complex ontologies or knowledge graphs (until baseline works)
- Performance optimization before correctness

## Architecture at a glance

```
data/                    # Datasets
  recipepairs_veg_eval.json   # 10,000 meat→vegetarian pairs

retrieval/               # Retrieval modules
  jaccard.py             # Shared: jaccard_similarity(), name_jaccard(), combined_similarity()
  cc_retrieval.py        # CC experiment: retrieve by recipe ID
  grounded_retrieval.py  # Grounded experiment: retrieve by recipe object

generation/              # LLM generation
  rag_generation.py      # CC experiment: graph-based generation
  cc_rag_generation.py   # CC experiment: text-based generation  
  grounded_generation.py # Grounded experiment: baseline_adapt(), grounded_adapt()

evaluation/              # Evaluation metrics
  hallucination_checker.py    # CC experiment: graph violation checks
  cc_hallucination_checker.py # CC experiment: text hallucination checks
  grounded_checker.py         # Grounded experiment: constraint + grounding violation checks

utils/                   # Shared utilities
  recipe_utils.py        # load_recipes(), get_ingredients()

scripts/                 # Data preparation
  fetch_recipepairs.py   # Download & filter RecipePairs dataset

run_exp_cc.py                  # CC experiment runner
run_exp_grounded.py            # Grounded experiment runner (single k)
run_exp_grounded_ablation.py   # Grounded experiment runner (k ablation)
```

## Key invariants (do not break)
- **CBR framing**: All experiments must be framed as case retrieval + adaptation
- **Bipartite graph structure**: Action↔Ingredient edges only (CookingCAKE-aligned)
- **Reproducibility**: All experiments use fixed random seeds, log configs
- **Grounding constraint**: Grounded generation must use ONLY retrieved ingredients

## Data contracts

### RecipePairs format (input)
```json
{
  "pairs": [
    {
      "base": {"id": int, "name": str, "ingredients": [str], "steps": str},
      "target": {"id": int, "name": str, "ingredients": [str], "steps": str},
      "constraint": "vegetarian"
    }
  ]
}
```

### Grounded experiment result (output)
```json
{
  "metadata": {"timestamp": str, "config": {...}},
  "summary": {
    "baseline": {"constraint_violation_rate": float, "avg_novel_ingredients": float},
    "grounded": {"constraint_violation_rate": float, "avg_novel_ingredients": float}
  },
  "results": [{...per-example details...}]
}
```

## Evaluation

### Metrics (Grounded Experiment)
1. **Constraint violation rate**: % of recipes containing meat keywords (vegetarian constraint)
2. **Grounding violation rate**: % of recipes with novel ingredients not in retrieved set
3. **Avg novel ingredients**: Mean count of ingredients not in allowed set

### Baselines
- **Baseline**: LLM sees retrieved recipes but no explicit ingredient restriction
- **Grounded**: LLM explicitly told "use ONLY these ingredients" with whitelist

### Key findings (preliminary, n=5)
- Grounded reduces avg novel ingredients by ~74% (6.2 → 1.6)
- Constraint violations reduced from 80% → 60%

## Commands (uv)
- Setup: `uv sync`
- Fetch data: `uv run python scripts/fetch_recipepairs.py`
- Run grounded experiment: `uv run python run_exp_grounded.py`
- Run ablation study: `uv run python run_exp_grounded_ablation.py`
- Run CC experiment: `uv run python run_exp_cc.py`
