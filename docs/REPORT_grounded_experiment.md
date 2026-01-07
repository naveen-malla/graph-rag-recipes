# Grounded RAG Experiment Report

**Author:** Naveen  
**Supervisor:** Prof. Ralph Bergmann  
**Date:** 7 January 2026  
**Status:** Complete (with k ablation)

---

## Executive Summary

This report documents a controlled experiment comparing **baseline RAG** vs **grounded RAG** for recipe adaptation. The key finding: **explicit grounding constraints reduce novel ingredient hallucinations by 87%** (from 6.29 to 0.81 per recipe, n=500). An ablation study with k=5 vs k=3 retrieval shows **k=3 is more effective**, suggesting additional retrieved recipes introduce noise rather than improve grounding.

---

## 1. Research Question

**Can explicit grounding constraints in LLM prompts reduce functional hallucinations during case-based recipe adaptation?**

### Hypothesis
When adapting a recipe using retrieved similar recipes as context, explicitly constraining the LLM to use *only* ingredients from retrieved recipes will reduce the introduction of novel (potentially incompatible) ingredients.

### CBR Framing
This experiment fits the CBR adaptation cycle:
1. **Retrieve**: Find similar vegetarian recipes
2. **Reuse**: Adapt the meat recipe using retrieved recipes as source material
3. **Revise**: (future work) Check and correct violations
4. **Retain**: (future work) Store successful adaptations

---

## 2. Experiment Design

### 2.1 Task
Adapt meat-based recipes to be vegetarian, using retrieved vegetarian recipes as context.

### 2.2 Conditions

| Condition | Description |
|-----------|-------------|
| **Baseline** | LLM sees retrieved recipes but no ingredient restriction |
| **Grounded** | LLM explicitly told: "Use ONLY these ingredients: [list from retrieved recipes]" |

### 2.3 Key Difference
Both conditions see the **same retrieved recipes**. The only difference is whether the prompt explicitly constrains ingredient usage.

### 2.4 Controlled Variables
- Same retrieval method (combined Jaccard similarity)
- Same LLM (llama3.2:3b via Ollama)
- Same random seed for reproducibility
- Same evaluation metrics

---

## 3. Dataset

### Source
**RecipePairs** (lishuyang/recipepairs on HuggingFace)
- 450,000+ recipe pairs with dietary adaptations

### Filtering
Extracted meat→vegetarian pairs where:
- Target recipe tagged "vegetarian"
- Base recipe contains meat keywords (chicken, beef, pork, etc.)

### Final Dataset
- **10,000 pairs** after filtering
- Max 5 pairs per base recipe (diversity)

---

## 4. Retrieval

### Method
**Combined Jaccard Similarity**:
```
score = 0.4 × ingredient_jaccard + 0.6 × name_jaccard
```

### Rationale
- RecipePairs has strong name correspondence (e.g., "grilled chicken" → "grilled tofu")
- Pure ingredient Jaccard gave low scores (0.10-0.17)
- Combined approach yields scores of 0.30-0.50

### Ablation
Testing with **k=3** and **k=5** retrieved recipes to determine optimal context size for grounding.

---

## 5. Evaluation Metrics

### 5.1 Constraint Violations
**Definition**: Generated recipe contains meat keywords (chicken, beef, pork, etc.)

**Measurement**: Keyword search in generated text

### 5.2 Grounding Violations  
**Definition**: Generated recipe uses ingredients not present in retrieved recipes

**Measurement**:
1. Extract ingredients from generated text (NER + section detection)
2. Compare against allowed set (union of retrieved recipe ingredients)
3. Count novel ingredients not in allowed set

### 5.3 Avg Novel Ingredients
Mean count of ingredients used that weren't in the allowed set.

---

## 6. Key Design Decisions

### 6.1 Allowed ingredients exclude base recipe
The base recipe (meat dish) is NOT included in the allowed ingredient set. Rationale: base contains meat, which would defeat the purpose of the vegetarian constraint.

### 6.2 Ingredient extraction via NER
We use the `ingredient-parser-nlp` library to extract ingredient names from generated text, with:
- Markdown stripping (LLM outputs `**Ingredients:**`)
- Section detection (parse only ingredient section, not instructions)
- Garbage filtering (skip headers like "Ingredients", equipment like "saucepan")

### 6.3 Simple retrieval (no embeddings)
We use Jaccard similarity rather than embedding-based retrieval because:
- Interpretable and fast
- Both conditions use same retrieval → it's a controlled variable
- Good enough for RecipePairs naming patterns

---

## 7. Results

### 7.1 Main Experiment (n=500, k=3)

| Metric | Baseline | Grounded | Δ |
|--------|----------|----------|---|
| Constraint violation rate | 73.8% | 61.0% | -12.8% |
| Grounding violation rate | 99.8% | 46.0% | **-53.8%** |
| **Avg novel ingredients** | **6.29** | **0.81** | **-87%** |

#### Key Findings

1. **87% reduction in novel ingredients**: Grounded prompting dramatically reduces hallucinated ingredients (6.29 → 0.81 per recipe).

2. **54% reduction in grounding violations**: Nearly half the grounded outputs use only allowed ingredients (vs ~0% baseline).

3. **Modest constraint improvement**: Both conditions still produce meat keywords (~61-74%), suggesting the LLM sometimes ignores the vegetarian constraint regardless of grounding.

4. **Grounding ≠ Task Compliance**: Grounded recipes stay within allowed ingredients but may still fail the vegetarian constraint (the LLM may select meat-like terms from allowed ingredients or hallucinate meat terms in instructions).

---

### 7.2 Ablation Study: k=3 vs k=5 (n=500 each)

| Metric | K=3 (Base) | K=3 (Ground) | K=5 (Base) | K=5 (Ground) |
|--------|------------|--------------|------------|--------------|
| Constraint violations | 369 (73.8%) | 305 (61.0%) | 397 (79.4%) | 339 (67.8%) |
| Grounding violations | 499 (99.8%) | 230 (46.0%) | 500 (100%) | 250 (50.0%) |
| Avg novel ingredients | 6.29 | 0.81 | 5.93 | 0.89 |

#### Improvements Over Baseline

| Metric | K=3 | K=5 | Winner |
|--------|-----|-----|--------|
| Constraint violation reduction | +12.8% | +11.6% | **K=3** |
| Grounding violation reduction | +53.8% | +50.0% | **K=3** |
| Novel ingredient reduction | 87.2% | 85.0% | **K=3** |

#### Ablation Findings

1. **K=3 outperforms K=5 across all metrics**: More retrieved recipes don't improve grounding effectiveness.

2. **Diminishing returns from additional context**: The 4th and 5th retrieved recipes appear to introduce noise rather than helpful constraints.

3. **Tighter constraints work better**: With k=3, the allowed ingredient set is smaller (more constrained), leading to fewer novel ingredients (0.81 vs 0.89).

4. **Recommendation**: Use **k=3** for grounded RAG. Additional retrieved recipes dilute the grounding effect without compensatory benefits.

---

## 8. Experiment Details

**Main Experiment (k=3):**
- 500 test examples (randomly sampled, seed=42)
- 2 conditions (baseline, grounded)
- Total: 1,000 LLM calls
- Runtime: ~5 hours
- Model: llama3.2:3b via Ollama (~30-40s per generation)
- Completed: 6 January 2026

**Ablation Study (k=5):**
- 500 test examples (same seed=42)
- 2 conditions (baseline, grounded)
- Total: 1,000 LLM calls
- Runtime: 3h 46min
- Model: llama3.2:3b via Ollama (~27s per generation)
- Completed: 7 January 2026

---

## 9. Limitations

1. **Sample size**: 500 examples per condition (10% of available data)
2. **Keyword-based constraint check**: May miss euphemisms for meat
3. **Ingredient matching**: Variants like "fresh parsley" vs "parsley" may cause false positives
4. **Single model**: Only tested with llama3.2:3b
5. **No human evaluation**: Quality of adaptations not assessed
6. **Binary k values**: Only tested k=3 and k=5; optimal k may lie elsewhere

---

## 10. Next Steps

1. ~~**k=5 ablation**: Test with 5 retrieved recipes~~ ✓ **Complete** — k=3 is optimal
2. **Analyze failure cases**: Why does grounded still introduce ~0.8 novel ingredients?
3. **Constraint violation analysis**: Why do both conditions fail the vegetarian constraint at high rates?
4. **Test k=1 and k=2**: Check if even tighter constraints improve grounding further
5. **Compare with graph-based approach**: Use CookingCAKE bipartite representation
6. **Human evaluation**: Assess adaptation quality, not just constraint satisfaction
7. **Larger model test**: Compare with llama3.2:7b or llama3.1:70b

---

## 11. Reproducibility

### Code
All experiment code in: `github.com/naveenkb/graph-rag-recipes`

### Key Files
- `run_exp_grounded.py`: Main experiment runner (k=3 or k=5)
- `generation/grounded_generation.py`: Baseline and grounded prompt templates
- `evaluation/grounded_checker.py`: Violation detection
- `retrieval/grounded_retrieval.py`: Combined similarity retrieval

### Results Files
- **k=3**: `results/grounded_exp_20260106_031203.json`
- **k=5**: `results/grounded_exp_k5_20260107_004924.json`

### Random Seed
`RANDOM_SEED = 42` for all experiments (ensures same test samples across k values)

---

## Appendix A: Prompt Templates

### Baseline Prompt
```
You are a recipe adaptation assistant.

ORIGINAL RECIPE TO ADAPT:
[base recipe]

SIMILAR RECIPES FOR INSPIRATION:
[retrieved recipes]

TASK: Adapt the original recipe to be vegetarian.

Generate the adapted recipe with:
1. A list of ingredients
2. Step-by-step instructions
```

### Grounded Prompt
```
You are a recipe adaptation assistant with STRICT ingredient constraints.

ORIGINAL RECIPE TO ADAPT:
[base recipe]

SIMILAR RECIPES FOR REFERENCE:
[retrieved recipes]

ALLOWED INGREDIENTS (use ONLY these):
[comma-separated list from retrieved recipes]

TASK: Adapt the original recipe to be vegetarian.

CRITICAL RULES:
- You MUST use only ingredients from the ALLOWED INGREDIENTS list above.
- Do NOT introduce any new ingredients not in the list.
- If an ingredient is needed but not in the list, find a substitute from the list or omit it.

Generate the adapted recipe with:
1. A list of ingredients (only from allowed list)
2. Step-by-step instructions
```

---

## Appendix B: Sample Output

### Example: Mushroom Rice (k=3)

**Base Recipe Ingredients**: butter, chicken broth, garlic, mushroom, onion, parsley, pepper, rice, salt

**Retrieved Recipes**:
1. Spanish rice
2. Mushroom ravioli
3. Mushroom ravioli (variant)

**Allowed Ingredients** (17 total): mushroom, chive, shallot, garlic, dried rosemary, flour, wonton wrapper, tomato, lowfat milk, butter, ...

**Baseline Output** (excerpt):
```
Ingredients:
- 2 tablespoons butter
- 2 cups vegetable broth ← NOVEL (not in allowed)
- 3 cloves garlic
- 8 oz mushrooms
- 1/4 cup fresh parsley ← NOVEL (allowed has "parsley" not "fresh parsley")
```
Novel ingredients: 11

**Grounded Output** (excerpt):
```
Ingredients:
- 2 tablespoons butter
- 1 tablespoon olive oil
- 1 medium onion
- 2 cups mushroom
- 2 cloves garlic
- 1 cup rice
- 2 teaspoons dried rosemary leaves
```
Novel ingredients: 3

---

*Report generated: 2026-01-07*  
*Main experiment: 500 examples, k=3 retrieval (6 Jan 2026)*  
*Ablation study: 500 examples, k=5 retrieval (7 Jan 2026)*
