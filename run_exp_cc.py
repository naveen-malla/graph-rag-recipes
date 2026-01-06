"""Experiment CC: Text-RAG vs Graph-RAG comparison (CookingCAKE-inspired bipartite graphs)."""
import json
from datetime import datetime
from pathlib import Path

from utils.recipe_utils import load_recipes
from retrieval.cc_retrieval import retrieve_similar
from generation.cc_rag_generation import text_rag_adapt, graph_rag_adapt
from evaluation.cc_hallucination_checker import evaluate_hallucinations


def run_experiment(query_recipe_id, constraint="vegetarian", model="llama3.2:3b"):
    """
    Run full experiment: retrieve → generate (both RAG types) → evaluate.
    
    Args:
        query_recipe_id: ID of recipe to adapt
        constraint: Adaptation constraint (e.g., "vegetarian")
        model: LLM model name
    
    Returns:
        dict: Experiment results with adaptations and evaluations
    """
    print(f"\n{'='*60}")
    print(f"Experiment: {query_recipe_id} → {constraint}")
    print(f"{'='*60}\n")
    
    # Step 1: Load recipes
    recipes = load_recipes()
    query_recipe = next(r for r in recipes if r["id"] == query_recipe_id)
    print(f"Query: {query_recipe['name']}")
    
    # Step 2: Retrieve similar recipes
    print("\n[1/4] Retrieving similar recipes...")
    similar_ids = retrieve_similar(query_recipe_id, recipes, top_k=2)
    similar_recipes = [
        next(r for r in recipes if r["id"] == recipe_id)
        for recipe_id, _ in similar_ids
    ]
    print(f"Retrieved: {', '.join(r['name'] for r in similar_recipes)}")
    
    # Step 3: Generate adaptations
    print("\n[2/4] Generating Text-RAG adaptation...")
    text_rag_result = text_rag_adapt(query_recipe, similar_recipes, constraint, model)
    print(f"Generated ({len(text_rag_result)} chars)")
    
    print("\n[3/4] Generating Graph-RAG adaptation...")
    graph_rag_result = graph_rag_adapt(query_recipe, similar_recipes, constraint, model)
    print(f"Generated ({len(graph_rag_result)} chars)")
    
    # Step 4: Evaluate both adaptations
    print("\n[4/4] Evaluating hallucinations...\n")
    
    print("--- Text-RAG Evaluation ---")
    text_eval = evaluate_hallucinations(text_rag_result, constraint, similar_recipes)
    print(f"Constraint: {text_eval['constraint_violation']['message']}")
    print(f"Ingredients: {text_eval['ingredient_consistency']['message']}")
    print(f"Has hallucination: {text_eval['has_hallucination']}")
    
    print("\n--- Graph-RAG Evaluation ---")
    graph_eval = evaluate_hallucinations(graph_rag_result, constraint, similar_recipes)
    print(f"Constraint: {graph_eval['constraint_violation']['message']}")
    print(f"Ingredients: {graph_eval['ingredient_consistency']['message']}")
    print(f"Has hallucination: {graph_eval['has_hallucination']}")
    
    # Compile results
    return {
        "query_recipe_id": query_recipe_id,
        "query_recipe_name": query_recipe['name'],
        "constraint": constraint,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "retrieved_recipes": [r['name'] for r in similar_recipes],
        "text_rag": {
            "output": text_rag_result,
            "evaluation": text_eval
        },
        "graph_rag": {
            "output": graph_rag_result,
            "evaluation": graph_eval
        }
    }


def generate_summary(results_list):
    """
    Generate summary comparison of Text-RAG vs Graph-RAG.
    
    Args:
        results_list: List of experiment result dicts
    
    Returns:
        dict: Summary statistics
    """
    text_rag_hallucinations = sum(
        1 for r in results_list 
        if r["text_rag"]["evaluation"]["has_hallucination"]
    )
    graph_rag_hallucinations = sum(
        1 for r in results_list 
        if r["graph_rag"]["evaluation"]["has_hallucination"]
    )
    
    return {
        "total_experiments": len(results_list),
        "text_rag_hallucinations": text_rag_hallucinations,
        "graph_rag_hallucinations": graph_rag_hallucinations,
        "text_rag_success_rate": 1 - (text_rag_hallucinations / len(results_list)),
        "graph_rag_success_rate": 1 - (graph_rag_hallucinations / len(results_list))
    }


if __name__ == "__main__":
    print("="*60)
    print("Text-RAG vs Graph-RAG Experiment")
    print("="*60)
    
    # Run experiments on test cases
    test_cases = [
        "recipe_001",  # Chicken Curry → vegetarian
        "recipe_003"   # Beef Tacos → vegetarian
    ]
    
    all_results = []
    for recipe_id in test_cases:
        result = run_experiment(recipe_id, constraint="vegetarian")
        all_results.append(result)
    
    # Generate summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    summary = generate_summary(all_results)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"\nText-RAG:")
    print(f"  Hallucinations: {summary['text_rag_hallucinations']}")
    print(f"  Success rate: {summary['text_rag_success_rate']:.1%}")
    print(f"\nGraph-RAG:")
    print(f"  Hallucinations: {summary['graph_rag_hallucinations']}")
    print(f"  Success rate: {summary['graph_rag_success_rate']:.1%}")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "results": all_results,
            "summary": summary
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
