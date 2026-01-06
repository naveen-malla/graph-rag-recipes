# ADR-0001: Use RecipePairs Dataset for Grounded Experiment

Date: 2026-01-05
Status: Accepted

## Context

We need a dataset of recipe pairs for evaluating recipe adaptation. Options considered:
1. RecipeNLG (1M+ recipes, but no pairs)
2. RecipePairs (450k pairs with dietary constraints)
3. Manual curation (too slow)

## Decision

Use **lishuyang/recipepairs** from HuggingFace. Filter for meat→vegetarian pairs where:
- Target recipe has "vegetarian" in categories
- Base recipe contains meat keywords in ingredients

## Consequences

**Positive:**
- Large scale: 10,000+ pairs available
- Built-in constraint labels (vegetarian, vegan, etc.)
- Paired structure matches CBR adaptation framing

**Negative:**
- Pairs may be noisy (not always true adaptations)
- Some recipes are duplicates with minor variations

## Alternatives Considered

- **RecipeNLG**: No built-in pairs, would need custom pairing logic
- **CookingCAKE corpus**: Smaller, may not have dietary constraints
