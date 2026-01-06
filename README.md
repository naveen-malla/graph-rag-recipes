# Graph-RAG Recipe Adaptation

PhD research on grounded retrieval for constraint-aware recipe adaptation. Tests whether explicit ingredient constraints reduce hallucinations in LLM-based recipe generation.

## Key Results (n=500)

**Grounded RAG reduces novel ingredient hallucinations by 87%**

| Metric | Baseline RAG | Grounded RAG |
|--------|--------------|--------------|
| Avg novel ingredients | 6.29 | 0.81 |
| Grounding violations | 99.8% | 46.0% |

**Approach**: Baseline sees retrieved recipes; grounded is explicitly told "use ONLY these ingredients: [list]"

## Quick Start

```bash
# Setup
uv sync
ollama pull llama3.2:3b

# Fetch data (10k meat→vegetarian pairs)
uv run python scripts/fetch_recipepairs.py

# Run experiment
uv run python run_exp_grounded.py
```

Results saved to `results/grounded_exp_*.json`

## Project Structure

```
retrieval/          # Jaccard + embedding similarity
generation/         # Baseline vs grounded prompts
evaluation/         # Constraint + grounding violation checks
run_exp_*.py        # Experiment runners
data/               # RecipePairs dataset (10k pairs)
results/            # Experiment outputs
```

## Documentation

- [DESIGN.md](DESIGN.md) — Architecture, metrics, data contracts
- [docs/REPORT_grounded_experiment.md](docs/REPORT_grounded_experiment.md) — Full experiment writeup
- [docs/decisions/](docs/decisions/) — ADRs (dataset choice, retrieval method, etc.)

## References

- Dataset: [RecipePairs](https://huggingface.co/datasets/lishuyang/recipepairs)
- Model: [Ollama](https://ollama.ai/) llama3.2:3b
- NER: [ingredient-parser-nlp](https://github.com/strangetom/ingredient-parser)

---

**Contact**: Naveen (PhD, Universität Trier) | Supervisor: Prof. Ralph Bergmann

