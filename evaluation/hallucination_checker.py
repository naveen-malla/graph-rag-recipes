"""Hallucination detection for recipe adaptations."""


def check_constraint_violation(adapted_text, constraint, forbidden_ingredients):
    """
    Check if adapted recipe violates constraint by using forbidden ingredients.
    
    Args:
        adapted_text: Generated recipe text (string)
        constraint: Constraint name (e.g., "vegetarian")
        forbidden_ingredients: List of forbidden ingredient names (lowercase)
    
    Returns:
        dict: {
            "violated": bool,
            "violations": list of found forbidden ingredients,
            "message": explanation string
        }
    """
    adapted_lower = adapted_text.lower()
    
    found_forbidden = []
    for ingredient in forbidden_ingredients:
        if ingredient in adapted_lower:
            found_forbidden.append(ingredient)
    
    if found_forbidden:
        return {
            "violated": True,
            "violations": found_forbidden,
            "message": f"Constraint '{constraint}' violated: found {', '.join(found_forbidden)}"
        }
    else:
        return {
            "violated": False,
            "violations": [],
            "message": f"Constraint '{constraint}' satisfied"
        }


def check_ingredient_consistency(adapted_text, source_ingredients, allowed_substitutions=None):
    """
    Check if adapted recipe uses reasonable ingredients (from source or valid substitutions).
    
    Args:
        adapted_text: Generated recipe text (string)
        source_ingredients: Set of ingredient names from retrieved recipes (lowercase)
        allowed_substitutions: Dict mapping forbidden → allowed (e.g., {"chicken": ["chickpeas", "tofu"]})
    
    Returns:
        dict: {
            "consistent": bool,
            "unexpected_ingredients": list of ingredients not in source or substitutions,
            "message": explanation string
        }
    """
    if allowed_substitutions is None:
        allowed_substitutions = {}
    
    adapted_lower = adapted_text.lower()
    
    # Build valid ingredient set: source + all allowed substitutions
    valid_ingredients = set(source_ingredients)
    for substitution_list in allowed_substitutions.values():
        valid_ingredients.update(substitution_list)
    
    # Simple heuristic: check if any source ingredients are mentioned
    # (Full NER would be better, but this is minimal PoC)
    mentioned = [ing for ing in valid_ingredients if ing in adapted_lower]
    
    # For PoC, we consider it consistent if at least some valid ingredients are mentioned
    # and no obviously wrong ingredients appear (hard to detect without NER)
    # This is a simplified check - in practice, you'd use NER or ingredient extraction
    
    if mentioned:
        return {
            "consistent": True,
            "unexpected_ingredients": [],
            "message": f"Recipe uses valid ingredients: {', '.join(mentioned[:5])}"
        }
    else:
        return {
            "consistent": False,
            "unexpected_ingredients": [],
            "message": "No recognized ingredients from source recipes found"
        }


def evaluate_hallucinations(adapted_text, constraint, source_recipes):
    """
    Evaluate hallucinations in adapted recipe.
    
    Args:
        adapted_text: Generated recipe text
        constraint: Constraint applied (e.g., "vegetarian")
        source_recipes: List of source recipe dicts used for adaptation
    
    Returns:
        dict: {
            "constraint_violation": dict from check_constraint_violation(),
            "ingredient_consistency": dict from check_ingredient_consistency(),
            "has_hallucination": bool (True if any check failed)
        }
    """
    # Define forbidden ingredients for vegetarian constraint
    forbidden = []
    allowed_subs = {}
    
    if constraint == "vegetarian":
        forbidden = ["chicken", "beef", "pork", "lamb", "fish", "shrimp", "meat"]
        allowed_subs = {
            "chicken": ["chickpeas", "tofu", "tempeh"],
            "beef": ["black beans", "lentils", "mushrooms"],
            "pork": ["tempeh", "jackfruit"],
            "lamb": ["lentils", "beans"],
            "fish": ["tofu", "tempeh"],
            "shrimp": ["tofu", "mushrooms"]
        }
    
    # Extract source ingredients
    source_ingredients = set()
    for recipe in source_recipes:
        source_ingredients.update(
            ing["name"].lower() for ing in recipe["graph"]["ingredients"]
        )
    
    # Run checks
    constraint_result = check_constraint_violation(adapted_text, constraint, forbidden)
    consistency_result = check_ingredient_consistency(
        adapted_text, source_ingredients, allowed_subs
    )
    
    # Determine if hallucination occurred
    has_hallucination = constraint_result["violated"] or not consistency_result["consistent"]
    
    return {
        "constraint_violation": constraint_result,
        "ingredient_consistency": consistency_result,
        "has_hallucination": has_hallucination
    }


if __name__ == "__main__":
    # Test hallucination detection
    
    print("=== Testing Hallucination Detection ===\n")
    
    # Mock source recipe
    source_recipes = [{
        "graph": {
            "ingredients": [
                {"name": "onion"},
                {"name": "oil"},
                {"name": "tomato sauce"},
                {"name": "curry powder"}
            ]
        }
    }]
    
    # Test 1: Valid vegetarian adaptation
    print("Test 1: Valid vegetarian adaptation")
    valid_text = "1. Chop onions. 2. Heat oil. 3. Add chickpeas and tomato sauce. 4. Add curry powder."
    result1 = evaluate_hallucinations(valid_text, "vegetarian", source_recipes)
    print(f"✅ Constraint: {result1['constraint_violation']['message']}")
    print(f"✅ Ingredients: {result1['ingredient_consistency']['message']}")
    print(f"Has hallucination: {result1['has_hallucination']}\n")
    
    # Test 2: Constraint violation (mentions chicken)
    print("Test 2: Constraint violation")
    invalid_text = "1. Chop onions. 2. Heat oil. 3. Add chicken and tomato sauce. 4. Add curry powder."
    result2 = evaluate_hallucinations(invalid_text, "vegetarian", source_recipes)
    print(f"❌ Constraint: {result2['constraint_violation']['message']}")
    print(f"✅ Ingredients: {result2['ingredient_consistency']['message']}")
    print(f"Has hallucination: {result2['has_hallucination']}\n")
