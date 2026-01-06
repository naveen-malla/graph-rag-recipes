"""Grounded generation for recipe adaptation: baseline vs constrained."""
import ollama


def extract_allowed_ingredients(retrieved_recipes):
    """
    Extract union of all ingredients from retrieved recipes.
    
    Args:
        retrieved_recipes: List of recipe dicts with 'ingredients' field
    
    Returns:
        Set of lowercase ingredient names
    """
    allowed = set()
    for recipe in retrieved_recipes:
        for ing in recipe.get("ingredients", []):
            allowed.add(ing.lower().strip())
    return allowed


def format_retrieved_context(retrieved_recipes):
    """
    Format retrieved recipes as readable context for prompt.
    
    Args:
        retrieved_recipes: List of recipe dicts with name, ingredients, steps
    
    Returns:
        Multi-line string with recipe details
    """
    parts = []
    for i, recipe in enumerate(retrieved_recipes, 1):
        parts.append(f"Recipe {i}: {recipe['name']}")
        parts.append(f"Ingredients: {', '.join(recipe['ingredients'])}")
        parts.append(f"Steps: {recipe['steps']}")
        parts.append("")
    return "\n".join(parts)


def format_query_recipe(query_recipe):
    """Format the query recipe for the prompt."""
    return (
        f"Name: {query_recipe['name']}\n"
        f"Ingredients: {', '.join(query_recipe['ingredients'])}\n"
        f"Steps: {query_recipe['steps']}"
    )


def baseline_adapt(query_recipe, retrieved_recipes, constraint, model="llama3.2:3b"):
    """
    Unconstrained adaptation — LLM sees retrieved recipes but no ingredient restriction.
    
    Args:
        query_recipe: Recipe dict to adapt (has name, ingredients, steps)
        retrieved_recipes: List of similar recipe dicts for context
        constraint: Adaptation constraint (e.g., "vegetarian")
        model: Ollama model name
    
    Returns:
        Generated adapted recipe text
    """
    retrieved_context = format_retrieved_context(retrieved_recipes)
    query_context = format_query_recipe(query_recipe)
    
    prompt = f"""You are a recipe adaptation assistant.

ORIGINAL RECIPE TO ADAPT:
{query_context}

SIMILAR RECIPES FOR INSPIRATION:
{retrieved_context}

TASK: Adapt the original recipe to be {constraint}.

Generate the adapted recipe with:
1. A list of ingredients
2. Step-by-step instructions

Adapted recipe:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]


def grounded_adapt(query_recipe, retrieved_recipes, constraint, model="llama3.2:3b"):
    """
    Constrained adaptation — explicit ingredient whitelist in prompt.
    
    Args:
        query_recipe: Recipe dict to adapt (has name, ingredients, steps)
        retrieved_recipes: List of similar recipe dicts for context
        constraint: Adaptation constraint (e.g., "vegetarian")
        model: Ollama model name
    
    Returns:
        Generated adapted recipe text
    """
    retrieved_context = format_retrieved_context(retrieved_recipes)
    query_context = format_query_recipe(query_recipe)
    allowed_ingredients = extract_allowed_ingredients(retrieved_recipes)
    allowed_list = ", ".join(sorted(allowed_ingredients))
    
    prompt = f"""You are a recipe adaptation assistant with STRICT ingredient constraints.

ORIGINAL RECIPE TO ADAPT:
{query_context}

SIMILAR RECIPES FOR REFERENCE:
{retrieved_context}

ALLOWED INGREDIENTS (use ONLY these):
{allowed_list}

TASK: Adapt the original recipe to be {constraint}.

CRITICAL RULES:
- You MUST use only ingredients from the ALLOWED INGREDIENTS list above.
- Do NOT introduce any new ingredients not in the list.
- If an ingredient is needed but not in the list, find a substitute from the list or omit it.

Generate the adapted recipe with:
1. A list of ingredients (only from allowed list)
2. Step-by-step instructions

Adapted recipe:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]
