"""Grounded Experiment: Retrieval by recipe object for RecipePairs dataset."""
from utils.recipe_utils import get_ingredients
from retrieval.jaccard import combined_similarity


# Weight for ingredient similarity (0.4 ingredient + 0.6 name)
ALPHA = 0.4


def retrieve_similar_recipes(query_recipe, corpus, top_k=3):
    """
    Find top-k most similar recipes from corpus based on combined similarity.
    
    Uses: alpha * ingredient_jaccard + (1 - alpha) * name_jaccard
    Default: 0.4 * ingredients + 0.6 * names
    
    Works with recipe objects directly (not IDs), suitable for RecipePairs format.
    
    Args:
        query_recipe: Recipe dict (must have 'ingredients' and 'name' fields)
        corpus: List of recipe dicts to search
        top_k: Number of similar recipes to return
    
    Returns:
        List of (recipe, score) tuples, sorted by score descending
    """
    query_ingredients = get_ingredients(query_recipe)
    
    scores = []
    for recipe in corpus:
        # Skip if same recipe (by name, since RecipePairs doesn't have consistent IDs)
        if recipe.get("name") == query_recipe.get("name"):
            continue
        
        recipe_ingredients = get_ingredients(recipe)
        score = combined_similarity(
            query_recipe, recipe, 
            query_ingredients, recipe_ingredients,
            alpha=ALPHA
        )
        
        scores.append((recipe, score))
    
    # Sort by score descending and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def extract_allowed_ingredients(retrieved_recipes):
    """
    Extract union of all ingredients from retrieved recipes.
    
    Args:
        retrieved_recipes: List of (recipe, score) tuples from retrieval
    
    Returns:
        Set of ingredient names (lowercase)
    """
    allowed = set()
    for recipe, _score in retrieved_recipes:
        ingredients = get_ingredients(recipe)
        allowed.update(ingredients)
    return allowed


if __name__ == "__main__":
    # Test the retrieval system
    from utils.recipe_utils import load_recipes
    
    print("=== Testing Grounded Retrieval ===\n")
    
    # Load RecipePairs data
    data = load_recipes(dataset="recipepairs")
    pairs = data["pairs"]
    
    # Build corpus from all targets
    corpus = [pair["target"] for pair in pairs]
    print(f"Corpus size: {len(corpus)} vegetarian recipes\n")
    
    # Test with first base recipe
    query = pairs[0]["base"]
    print(f"Query: {query['name']}")
    print(f"Query ingredients: {query['ingredients'][:5]}...\n")
    
    # Retrieve
    results = retrieve_similar_recipes(query, corpus, top_k=3)
    
    print("Top 3 similar recipes:")
    for recipe, score in results:
        print(f"  {recipe['name'][:40]:40} | score: {score:.3f}")
    
    # Extract allowed ingredients
    allowed = extract_allowed_ingredients(results)
    print(f"\nAllowed ingredients ({len(allowed)} total):")
    print(f"  {list(allowed)[:10]}...")
