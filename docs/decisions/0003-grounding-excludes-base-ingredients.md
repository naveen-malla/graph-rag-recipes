# ADR-0003: Grounding Constraint Uses Only Retrieved Ingredients

Date: 2026-01-05
Status: Accepted

## Context

For the "grounded" condition, we need to define what ingredients the LLM is allowed to use. Two options:
1. Retrieved ingredients + base recipe ingredients
2. Retrieved ingredients only

## Decision

**Retrieved ingredients only.** The base recipe ingredients are explicitly excluded from the allowed set.

Rationale:
- Base recipe contains meat (that's why we're adapting it)
- Including base ingredients would allow meat in the "allowed" set
- This makes the grounding constraint meaningless

## Consequences

**Positive:**
- Clean experimental design: grounded must use ONLY what's retrieved
- No accidental meat inclusion via base ingredients
- Measures true "closed-world reuse" from retrieved cases

**Negative:**
- Allowed set may be small (15-25 ingredients typically)
- Common ingredients (salt, pepper, oil) may be missing if not in retrieved recipes

## Alternatives Considered

- **Include base ingredients (filtered)**: Remove meat, keep others. More complex, harder to verify.
- **Use a pantry list**: Add common staples. Muddies the "grounded" concept.
