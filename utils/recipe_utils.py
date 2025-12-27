"""Shared utilities for working with recipes."""
import json


def load_recipes(recipes_path="data/recipes.json"):
    """Load recipes from JSON file."""
    with open(recipes_path) as f:
        data = json.load(f)
    return data["recipes"]


def get_ingredients(recipe):
    """Extract ingredient names from a recipe's graph representation."""
    return {ing["name"].lower() for ing in recipe["graph"]["ingredients"]}
