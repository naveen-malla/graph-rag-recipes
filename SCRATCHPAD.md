# SCRATCHPAD

Temporary notes, open questions, and experiment tracking.

---

## Current Status (2026-01-05)

### Grounded Experiment V2 — In Progress
- ✅ RecipePairs dataset: 10,000 meat→vegetarian pairs fetched
- ✅ Retrieval: Combined similarity (0.4 ingredient + 0.6 name Jaccard)
- ✅ Generation: baseline_adapt() vs grounded_adapt() 
- ✅ Evaluation: constraint violations (meat keywords) + grounding violations (novel ingredients)
- ✅ Ingredient extractor: fixed markdown parsing, garbage filtering
- ⏳ Ablation study: k=3 and k=5, 1000 examples each — running overnight

### Preliminary Results (n=5)
| Metric | Baseline | Grounded |
|--------|----------|----------|
| Avg novel ingredients | 6.2 | 1.6 |
| Reduction | — | **74%** |

---

## Open Questions

1. **Why does grounded still introduce 1-2 novel ingredients?**
   - Lemmatization mismatch? ("fresh parsley" vs "parsley")
   - LLM ignoring constraint?
   - Allowed set too restrictive?

2. **Should we try larger k values (k=10)?**
   - More retrieved recipes → larger allowed ingredient set
   - But also more noise in context

3. **Constraint violations still high (60-80%)**
   - LLM mentions meat even when adapting
   - May need stronger prompt engineering or post-filtering

---

## Experiments TODO

- [ ] Run ablation study (k=3, k=5) with n=1000
- [ ] Analyze failure cases (where grounded still violates)
- [ ] Try embedding-based retrieval (if Jaccard proves insufficient)
- [ ] Compare with CC experiment (graph-based)

---

## Links to Validate

- RecipePairs dataset: https://huggingface.co/datasets/lishuyang/recipepairs
- ingredient-parser-nlp: https://github.com/strangetom/ingredient-parser
- spaCy en_core_web_sm: https://spacy.io/models/en

---

## Session Notes

### 2026-01-05 Session
- Started with CC experiment (V1), pivoted to Grounded experiment (V2)
- Key insight: Name similarity is crucial for retrieval (ingredient-only Jaccard too weak)
- Fixed ingredient extractor: was picking up headers, section names, equipment
- Decided on 0.4 ingredient + 0.6 name weighting for combined similarity
