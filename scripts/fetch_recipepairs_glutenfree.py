"""
Fetch gluten-free recipes from RecipePairs dataset.

Downloads from HuggingFace, filters for targets where:
- Target recipe is tagged "gluten-free"

Outputs recipes to data/recipepairs_glutenfree_eval.json
"""

import json
from pathlib import Path
from glob import glob
import pandas as pd
import numpy as np

# --- Config ---
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "recipepairs_glutenfree_eval.json"
TARGET_COUNT = 10000
MAX_PER_TARGET = 5  # Max appearances per target recipe (limits duplicates)

# HuggingFace paths
CACHE_BASE = Path.home() / ".cache/huggingface/hub/datasets--lishuyang--recipepairs/snapshots"
HF_RECIPES_URL = "hf://datasets/lishuyang/recipepairs/recipes.parquet"
HF_PAIRS_URL = "hf://datasets/lishuyang/recipepairs/pairs.parquet"

def load_parquet(table_name: str) -> pd.DataFrame:
    """Load parquet from local cache if available, else download from HuggingFace."""
    cache_pattern = str(CACHE_BASE / f"*/{table_name}.parquet")
    matches = glob(cache_pattern)
    if matches:
        print(f"  (using local cache)")
        return pd.read_parquet(matches[0])
    print(f"  (downloading from HuggingFace...)")
    url = HF_RECIPES_URL if table_name == "recipes" else HF_PAIRS_URL
    return pd.read_parquet(url)

def main():
    print("Loading recipes table...")
    recipes_df = load_parquet("recipes")
    print(f"  → {len(recipes_df):,} recipes loaded")

    print("Loading pairs table...")
    pairs_df = load_parquet("pairs")
    print(f"  → {len(pairs_df):,} pairs loaded")

    # Index recipes by ID for O(1) lookup
    print("Building recipe lookup index...")
    recipe_lookup = {row["id"]: row.to_dict() for _, row in recipes_df.iterrows()}

    def normalize_categories(cats):
        if isinstance(cats, (list, tuple, np.ndarray)):
            return [str(c).strip().lower().replace("-", "_") for c in cats]
        if isinstance(cats, str):
            return [cats.strip().lower().replace("-", "_")]
        return []

    def to_plain_list(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value] if pd.notna(value) else []

    def is_glutenfree_target(pair):
        """Check target recipe categories for gluten_free/gluten-free tag."""
        target = recipe_lookup.get(pair["target"])
        if not target:
            return False
        norm = normalize_categories(target.get("categories", []))
        return "gluten_free" in norm

    # Collect unique gluten-free targets (from pairs). If none found, fall back to all recipes.
    print(f"Collecting gluten-free targets (goal: {TARGET_COUNT}, max {MAX_PER_TARGET}/target)...")
    gf_recipes = []
    target_counts = {}
    seen_targets = set()      # unique by target id
    seen_names = set()        # also dedupe by recipe name (case-insensitive)

    def add_target(target):
        tid = target["id"]
        tname = target["name"].strip()
        key = tname.lower()
        if target_counts.get(tid, 0) >= MAX_PER_TARGET:
            return False
        if tid in seen_targets or key in seen_names:
            target_counts[tid] = target_counts.get(tid, 0) + 1
            return False
        seen_targets.add(tid)
        seen_names.add(key)
        target_counts[tid] = target_counts.get(tid, 0) + 1
        gf_recipes.append({
            "id": int(target["id"]),
            "name": target["name"],
            "ingredients": to_plain_list(target.get("ingredients", [])),
            "steps": to_plain_list(target.get("steps", [])),
            "categories": normalize_categories(target.get("categories", [])),
            "constraint": "gluten-free"
        })
        print(f"  [{len(gf_recipes)}/{TARGET_COUNT}] {tname}")
        return True

    # Pass 1: from pairs
    for _, pair in pairs_df.iterrows():
        if len(gf_recipes) >= TARGET_COUNT:
            break
        if not is_glutenfree_target(pair):
            continue
        target = recipe_lookup.get(pair["target"])
        if not target:
            continue
        add_target(target)

    # Pass 2 (fallback): from all recipes if we didn't reach target count
    if len(gf_recipes) < TARGET_COUNT:
        print(f"Fallback: scanning all recipes (need {TARGET_COUNT - len(gf_recipes)} more)...")
        for _, row in recipes_df.iterrows():
            if len(gf_recipes) >= TARGET_COUNT:
                break
            norm = normalize_categories(row.get("categories", []))
            if "gluten_free" not in norm:
                continue
            add_target(row)

    # Save output
    output = {
        "metadata": {
            "source": "lishuyang/recipepairs",
            "filter": "gluten_free_targets",
            "count": len(gf_recipes)
        },
        "recipes": gf_recipes
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
