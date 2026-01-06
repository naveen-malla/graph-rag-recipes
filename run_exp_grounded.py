"""Grounded Experiment: Baseline vs Grounded adaptation comparison."""
import json
import random
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from utils.recipe_utils import load_recipes
from retrieval.grounded_retrieval import retrieve_similar_recipes, extract_allowed_ingredients
from generation.grounded_generation import baseline_adapt, grounded_adapt
from evaluation.grounded_checker import evaluate_grounded


# === CONFIG ===
NUM_EXAMPLES = 500        # ~4 hours runtime
TOP_K = 3                 # Number of recipes to retrieve
MODEL = "llama3.2:3b"     # LLM model
CONSTRAINT = "vegetarian"
RANDOM_SEED = 42          # For reproducibility


def run_experiment():
    """
    Run the grounded experiment: compare baseline vs grounded adaptation.
    
    For each test example:
    1. Take a base recipe (meat dish)
    2. Retrieve top-K similar vegetarian recipes
    3. Generate adaptations with both methods
    4. Evaluate both for constraint and grounding violations
    """
    print(f"\n{'='*60}")
    print("GROUNDED EXPERIMENT: Baseline vs Grounded Adaptation")
    print(f"{'='*60}")
    print(f"Config: {NUM_EXAMPLES} examples, top-{TOP_K} retrieval, model={MODEL}")
    print(f"{'='*60}\n")
    
    # Set seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Load data
    print("[1/5] Loading data...")
    data = load_recipes(dataset="recipepairs")
    pairs = data["pairs"]
    
    # Build corpus from all targets (vegetarian recipes)
    corpus = [pair["target"] for pair in pairs]
    print(f"  Corpus: {len(corpus)} vegetarian recipes")
    
    # Extract base recipes
    base_recipes = [pair["base"] for pair in pairs]
    print(f"  Base recipes: {len(base_recipes)} meat dishes")
    
    # Sample test examples
    print(f"\n[2/5] Sampling {NUM_EXAMPLES} test examples...")
    test_bases = random.sample(base_recipes, NUM_EXAMPLES)
    
    # Run experiment for each example
    print(f"\n[3/5] Running experiment...")
    results = []
    
    for base in tqdm(test_bases, desc="Adapting recipes", unit="recipe"):
        # Retrieve similar recipes
        retrieved = retrieve_similar_recipes(base, corpus, top_k=TOP_K)
        retrieved_recipes = [r for r, _score in retrieved]
        
        # Extract allowed ingredients
        allowed = extract_allowed_ingredients(retrieved)
        
        # Generate baseline adaptation
        output_baseline = baseline_adapt(base, retrieved_recipes, CONSTRAINT, MODEL)
        
        # Generate grounded adaptation
        output_grounded = grounded_adapt(base, retrieved_recipes, CONSTRAINT, MODEL)
        
        # Evaluate both
        eval_baseline = evaluate_grounded(output_baseline, allowed)
        eval_grounded = evaluate_grounded(output_grounded, allowed)
        
        # Store result
        result = {
            "example_id": len(results) + 1,
            "base_recipe": {
                "name": base["name"],
                "ingredients": base["ingredients"]
            },
            "retrieved_recipes": [
                {"name": r["name"], "score": s} 
                for r, s in retrieved
            ],
            "allowed_ingredients": list(allowed),
            "baseline": {
                "output": output_baseline,
                "evaluation": eval_baseline
            },
            "grounded": {
                "output": output_grounded,
                "evaluation": eval_grounded
            }
        }
        results.append(result)
    
    # Compute summary statistics
    print(f"\n[4/5] Computing summary statistics...")
    summary = compute_summary(results)
    
    # Save results
    print(f"\n[5/5] Saving results...")
    save_results(results, summary)
    
    # Print summary
    print_summary(summary)
    
    return results, summary


def compute_summary(results):
    """Compute aggregate statistics from results."""
    n = len(results)
    
    # Baseline stats
    baseline_constraint_violations = sum(
        1 for r in results if r["baseline"]["evaluation"]["constraint_check"]["violated"]
    )
    baseline_grounding_violations = sum(
        1 for r in results if r["baseline"]["evaluation"]["grounding_check"]["violated"]
    )
    baseline_avg_novel = sum(
        r["baseline"]["evaluation"]["grounding_check"]["count"] for r in results
    ) / n
    
    # Grounded stats
    grounded_constraint_violations = sum(
        1 for r in results if r["grounded"]["evaluation"]["constraint_check"]["violated"]
    )
    grounded_grounding_violations = sum(
        1 for r in results if r["grounded"]["evaluation"]["grounding_check"]["violated"]
    )
    grounded_avg_novel = sum(
        r["grounded"]["evaluation"]["grounding_check"]["count"] for r in results
    ) / n
    
    return {
        "num_examples": n,
        "baseline": {
            "constraint_violation_count": baseline_constraint_violations,
            "constraint_violation_rate": baseline_constraint_violations / n,
            "grounding_violation_count": baseline_grounding_violations,
            "grounding_violation_rate": baseline_grounding_violations / n,
            "avg_novel_ingredients": baseline_avg_novel
        },
        "grounded": {
            "constraint_violation_count": grounded_constraint_violations,
            "constraint_violation_rate": grounded_constraint_violations / n,
            "grounding_violation_count": grounded_grounding_violations,
            "grounding_violation_rate": grounded_grounding_violations / n,
            "avg_novel_ingredients": grounded_avg_novel
        }
    }


def save_results(results, summary):
    """Save results to JSON file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = results_dir / f"grounded_exp_{timestamp}.json"
    
    output = {
        "metadata": {
            "timestamp": timestamp,
            "config": {
                "num_examples": NUM_EXAMPLES,
                "top_k": TOP_K,
                "model": MODEL,
                "constraint": CONSTRAINT,
                "random_seed": RANDOM_SEED
            }
        },
        "summary": summary,
        "results": results
    }
    
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"  Saved to: {filename}")


def print_summary(summary):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Examples: {summary['num_examples']}")
    print()
    
    print("                        | Baseline | Grounded |")
    print("-" * 50)
    
    b = summary["baseline"]
    g = summary["grounded"]
    
    print(f"Constraint violations   | {b['constraint_violation_count']:>8} | {g['constraint_violation_count']:>8} |")
    print(f"Constraint rate         | {b['constraint_violation_rate']:>7.1%} | {g['constraint_violation_rate']:>7.1%} |")
    print(f"Grounding violations    | {b['grounding_violation_count']:>8} | {g['grounding_violation_count']:>8} |")
    print(f"Grounding rate          | {b['grounding_violation_rate']:>7.1%} | {g['grounding_violation_rate']:>7.1%} |")
    print(f"Avg novel ingredients   | {b['avg_novel_ingredients']:>8.2f} | {g['avg_novel_ingredients']:>8.2f} |")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    run_experiment()
