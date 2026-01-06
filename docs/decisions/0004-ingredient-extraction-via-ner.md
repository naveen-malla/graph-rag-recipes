# ADR-0004: Ingredient Extraction via NER + Section Detection

Date: 2026-01-05
Status: Accepted

## Context

To measure grounding violations, we need to extract ingredients from LLM-generated recipe text. Challenges:
1. LLM outputs markdown (e.g., `**Ingredients:**`)
2. ingredient-parser-nlp parses ANY text as ingredient (even headers, equipment)
3. Variants like "fresh parsley" need to match "parsley"

## Decision

Use a multi-step extraction:

1. **Strip markdown**: Remove `**` and `__` formatting
2. **Section detection**: Regex to find text between "Ingredients:" and "Steps:/Instructions:"
3. **Line-by-line NER**: Use ingredient-parser-nlp on each line in section
4. **Garbage filtering**: Skip headers, equipment, instruction-starter words
5. **Lemmatization** (optional): spaCy to normalize "tomatoes" → "tomato"

## Consequences

**Positive:**
- Extracts only actual ingredients (not "Vegetarian Mushroom Rice" title)
- Handles LLM output format variability
- Lemmatization catches plural/singular variations

**Negative:**
- Regex-based section detection may fail on unusual formats
- Compound ingredients ("olive oil") may not match ("oil")
- spaCy adds ~100MB dependency

## Alternatives Considered

- **LLM-based extraction**: Ask another LLM to extract. Expensive, slow.
- **Parse entire text**: Too many false positives (equipment, headers).
- **Structured output format**: Force LLM to output JSON. May hurt generation quality.
