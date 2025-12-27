"""Text-RAG and Graph-RAG generation for recipe adaptation."""
import ollama


def format_text_rag_context(similar_recipes):
    """Format similar recipes as plain text for Text-RAG."""
    context_parts = []
    for recipe in similar_recipes:
        context_parts.append(f"Recipe: {recipe['name']}")
        context_parts.append(f"Text: {recipe['text']}")
        context_parts.append("")  # blank line
    return "\n".join(context_parts)


def format_graph_rag_context(similar_recipes):
    """Format similar recipes as structured graph text for Graph-RAG."""
    context_parts = []
    for recipe in similar_recipes:
        context_parts.append(f"Recipe: {recipe['name']}")
        context_parts.append("Structure:")
        
        # Build action -> ingredient mapping
        for action in recipe["graph"]["actions"]:
            step_idx = action["step_index"]
            verb = action["verb"]
            
            # Find edges from this action
            action_edges = [
                e for e in recipe["graph"]["edges"] 
                if e["source"] == action["id"]
            ]
            
            # Find ingredient details for each edge
            ingredients_info = []
            for edge in action_edges:
                # Find the ingredient
                ing = next(
                    (i for i in recipe["graph"]["ingredients"] if i["id"] == edge["target"]),
                    None
                )
                if ing:
                    # Format: ingredient (quantity unit state) [role]
                    parts = [ing["name"]]
                    details = []
                    if edge.get("quantity"):
                        details.append(str(edge["quantity"]))
                    if edge.get("unit"):
                        details.append(edge["unit"])
                    if edge.get("state"):
                        details.append(edge["state"])
                    if details:
                        parts.append(f"({' '.join(details)})")
                    parts.append(f"[{edge['role']}]")
                    ingredients_info.append(" ".join(parts))
            
            # Format step line
            ing_str = ", ".join(ingredients_info) if ingredients_info else "—"
            context_parts.append(f"  Step {step_idx}. {verb} → {ing_str}")
        
        context_parts.append("")  # blank line
    return "\n".join(context_parts)


def generate_adaptation(context, constraint, model="llama3.2:3b"):
    """
    Generate adapted recipe using Ollama.
    
    Args:
        context: Formatted context (text or graph)
        constraint: Adaptation constraint (e.g., "vegetarian")
        model: Ollama model name
    
    Returns:
        Generated recipe text
    """
    prompt = f"""You are a recipe adaptation assistant. Given similar recipes and a constraint, generate an adapted recipe.

Similar recipes:
{context}

Constraint: {constraint}

Generate a step-by-step adapted recipe that satisfies the constraint. Be concise and practical.

Adapted recipe:"""

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]


def text_rag_adapt(query_recipe, similar_recipes, constraint, model="llama3.2:3b"):
    """Adapt recipe using Text-RAG approach."""
    context = format_text_rag_context(similar_recipes)
    return generate_adaptation(context, constraint, model)


def graph_rag_adapt(query_recipe, similar_recipes, constraint, model="llama3.2:3b"):
    """Adapt recipe using Graph-RAG approach."""
    context = format_graph_rag_context(similar_recipes)
    return generate_adaptation(context, constraint, model)


if __name__ == "__main__":
    from utils.recipe_utils import load_recipes
    from retrieval.simple_retrieval import retrieve_similar
    
    print("=== Testing Generation System ===\n")
    
    # Load recipes and retrieve similar ones
    recipes = load_recipes()
    query_id = "recipe_001"  # Chicken Curry
    
    # Get query recipe
    query_recipe = next(r for r in recipes if r["id"] == query_id)
    print(f"Query: {query_recipe['name']} (meat)")
    
    # Retrieve similar recipes
    similar_ids = retrieve_similar(query_id, recipes, top_k=2)
    similar_recipes = [
        next(r for r in recipes if r["id"] == recipe_id)
        for recipe_id, _ in similar_ids
    ]
    
    print(f"Retrieved: {', '.join(r['name'] for r in similar_recipes)}\n")
    
    # Test Text-RAG
    print("--- Text-RAG Adaptation ---")
    text_result = text_rag_adapt(query_recipe, similar_recipes, "vegetarian")
    print(text_result[:300] + "...\n")
    
    # Test Graph-RAG
    print("--- Graph-RAG Adaptation ---")
    graph_result = graph_rag_adapt(query_recipe, similar_recipes, "vegetarian")
    print(graph_result[:300] + "...")
