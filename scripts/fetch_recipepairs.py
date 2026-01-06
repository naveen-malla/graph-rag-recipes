"""
Fetch meat→vegetarian recipe pairs from RecipePairs dataset.

Downloads from HuggingFace, filters for pairs where:
- Target recipe is tagged "vegetarian"
- Base recipe contains a meat keyword in ingredients

Outputs pairs to data/recipepairs_veg_eval.json
"""

import json
from pathlib import Path
from glob import glob
import pandas as pd

# --- Config ---
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "recipepairs_veg_eval.json"
TARGET_COUNT = 10000
MAX_PER_BASE = 5    # Max pairs per base recipe (limits duplicates)
MAX_PER_TARGET = 5  # Max pairs per target recipe (limits duplicates)
MEAT_KEYWORDS = {"chicken", "beef", "pork", "lamb", "turkey", "bacon", "sausage", "ham", "meat", "steak", "fish", "shrimp", "salmon"}

# HuggingFace paths
CACHE_BASE = Path.home() / ".cache/huggingface/hub/datasets--lishuyang--recipepairs/snapshots"
HF_RECIPES_URL = "hf://datasets/lishuyang/recipepairs/recipes.parquet"
HF_PAIRS_URL = "hf://datasets/lishuyang/recipepairs/pairs.parquet"


def load_parquet(table_name: str) -> pd.DataFrame:
    """Load parquet from local cache if available, else download from HuggingFace."""
    # Try local cache first (fast)
    cache_pattern = str(CACHE_BASE / f"*/{table_name}.parquet")
    matches = glob(cache_pattern)
    if matches:
        print(f"  (using local cache)")
        return pd.read_parquet(matches[0])
    
    # Fall back to HuggingFace URL (portable, but slower)
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

    def is_meat_to_veg(pair):
        """Check if pair is meat→vegetarian."""
        # Target must be vegetarian
        if "vegetarian" not in pair["categories"]:
            return False
        
        # Get base recipe
        base = recipe_lookup.get(pair["base"])
        if not base:
            return False
        
        # Base must contain a meat keyword
        ingredients_text = " ".join(base["ingredients"]).lower()
        return any(meat in ingredients_text for meat in MEAT_KEYWORDS)

    # Filter pairs
    print(f"Filtering for meat→vegetarian pairs (target: {TARGET_COUNT}, max {MAX_PER_BASE}/base, max {MAX_PER_TARGET}/target)...")
    veg_pairs = []
    base_counts = {}    # Count how many times each base appears
    target_counts = {}  # Count how many times each target appears
    for _, pair in pairs_df.iterrows():
        base_id = pair["base"]
        target_id = pair["target"]
        
        # Check limits
        if base_counts.get(base_id, 0) >= MAX_PER_BASE:
            continue
        if target_counts.get(target_id, 0) >= MAX_PER_TARGET:
            continue
            
        if is_meat_to_veg(pair):
            base = recipe_lookup[base_id]
            target = recipe_lookup[target_id]
            base_counts[base_id] = base_counts.get(base_id, 0) + 1
            target_counts[target_id] = target_counts.get(target_id, 0) + 1
            veg_pairs.append({
                "base": {
                    "id": int(base["id"]),
                    "name": base["name"],
                    "ingredients": list(base["ingredients"]),
                    "steps": base["steps"]  # Keep as string (that's the original format)
                },
                "target": {
                    "id": int(target["id"]),
                    "name": target["name"],
                    "ingredients": list(target["ingredients"]),
                    "steps": target["steps"]  # Keep as string
                },
                "constraint": "vegetarian"
            })
            print(f"  [{len(veg_pairs)}/{TARGET_COUNT}] {base['name']} → {target['name']}")
        
        if len(veg_pairs) >= TARGET_COUNT:
            break

    # Save output
    output = {
        "metadata": {
            "source": "lishuyang/recipepairs",
            "filter": "meat_to_vegetarian",
            "count": len(veg_pairs)
        },
        "pairs": veg_pairs
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
