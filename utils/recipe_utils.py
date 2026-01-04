"""Shared utilities for working with recipes."""
import json
from pathlib import Path

# Default paths
TOY_RECIPES_PATH = Path(__file__).parent.parent / "data" / "recipes.json"
RECIPEPAIRS_PATH = Path(__file__).parent.parent / "data" / "recipepairs_veg_eval.json"


def load_recipes(dataset="toy"):
    """
    Load recipes from JSON file.
    
    Args:
        dataset: "toy" for demo recipes with graphs, 
                 "recipepairs" for eval pairs (text-only)
    
    Returns:
        For "toy": list of recipe dicts with 'graph' field
        For "recipepairs": dict with 'pairs' list, each pair has 'base' and 'target'
    """
    if dataset == "toy":
        with open(TOY_RECIPES_PATH) as f:
            data = json.load(f)
        return data["recipes"]
    
    elif dataset == "recipepairs":
        with open(RECIPEPAIRS_PATH) as f:
            data = json.load(f)
        # Return the full structure (metadata + pairs)
        return data
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'toy' or 'recipepairs'")


def get_ingredients(recipe):
    """
    Extract ingredient names from a recipe.
    
    Handles both formats:
    - Toy: recipe["graph"]["ingredients"] = [{"name": "chicken"}, ...]
    - RecipePairs: recipe["ingredients"] = ["chicken", ...]
    """
    if "graph" in recipe:
        # Toy format: graph with ingredient objects
        return {ing["name"].lower() for ing in recipe["graph"]["ingredients"]}
    elif "ingredients" in recipe:
        # RecipePairs format: flat list of strings
        return {ing.lower() for ing in recipe["ingredients"]}
    else:
        return set()
