"""Shared Jaccard similarity computation for retrieval."""


def jaccard_similarity(set1, set2):
    """
    Compute Jaccard similarity between two sets.
    
    Jaccard = |intersection| / |union|
    
    Args:
        set1: First set of items
        set2: Second set of items
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    if not set1 and not set2:
        return 0.0
    
    intersection = set1 & set2
    union = set1 | set2
    
    return len(intersection) / len(union) if union else 0.0


def name_jaccard(name1, name2):
    """
    Compute Jaccard similarity between recipe names based on word tokens.
    
    Args:
        name1: First recipe name (string)
        name2: Second recipe name (string)
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    words1 = set(name1.lower().split())
    words2 = set(name2.lower().split())
    return jaccard_similarity(words1, words2)


def combined_similarity(recipe1, recipe2, ingredients1, ingredients2, alpha=0.4):
    """
    Compute combined similarity using ingredients and names.
    
    combined = alpha * ingredient_jaccard + (1 - alpha) * name_jaccard
    
    Args:
        recipe1: First recipe dict (must have 'name')
        recipe2: Second recipe dict (must have 'name')
        ingredients1: Set of ingredient names for recipe1
        ingredients2: Set of ingredient names for recipe2
        alpha: Weight for ingredient similarity (default 0.4)
    
    Returns:
        float: Combined similarity score between 0.0 and 1.0
    """
    ingr_sim = jaccard_similarity(ingredients1, ingredients2)
    name_sim = name_jaccard(recipe1.get("name", ""), recipe2.get("name", ""))
    
    return alpha * ingr_sim + (1 - alpha) * name_sim
