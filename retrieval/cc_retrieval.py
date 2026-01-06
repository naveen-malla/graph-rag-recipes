"""CC Experiment: Retrieval by recipe ID for toy dataset."""
from utils.recipe_utils import get_ingredients
from retrieval.jaccard import jaccard_similarity


def retrieve_similar(query_recipe_id, recipes, top_k=2):
    """
    Find the top-k most similar recipes based on ingredient overlap.
    
    Args:
        query_recipe_id: ID of the query recipe
        recipes: List of all recipes
        top_k: Number of similar recipes to return
    
    Returns:
        List of (recipe_id, score) tuples, sorted by score descending
    """
    # Find query recipe
    query_recipe = None
    for r in recipes:
        if r["id"] == query_recipe_id:
            query_recipe = r
            break
    
    if not query_recipe:
        raise ValueError(f"Recipe {query_recipe_id} not found")
    
    # Get query ingredients
    query_ingredients = get_ingredients(query_recipe)
    
    # Calculate similarity with all other recipes
    scores = []
    for recipe in recipes:
        if recipe["id"] == query_recipe_id:
            continue  # Skip self
        
        recipe_ingredients = get_ingredients(recipe)
        
        # Use shared Jaccard function
        score = jaccard_similarity(query_ingredients, recipe_ingredients)
        
        scores.append((recipe["id"], score))
    
    # Sort by score descending and return top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    # Test the retrieval system
    recipes = load_recipes()
    
    print("=== Testing Retrieval System ===\n")
    
    # Test with Chicken Curry (should find similar recipes)
    query_id = "recipe_001"
    print(f"Query recipe: {query_id}")
    
    # Find query recipe name
    for r in recipes:
        if r["id"] == query_id:
            print(f"Name: {r['name']}\n")
            break
    
    results = retrieve_similar(query_id, recipes, top_k=2)
    
    print("Top 2 similar recipes:")
    for recipe_id, score in results:
        # Find recipe name
        for r in recipes:
            if r["id"] == recipe_id:
                print(f"  {recipe_id} ({r['name']}): {score:.3f}")
                break
