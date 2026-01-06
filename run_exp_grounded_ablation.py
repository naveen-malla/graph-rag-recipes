"""Grounded Experiment: Baseline vs Grounded adaptation with k ablation.

Runs experiment with multiple top-k values for overnight execution.
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from utils.recipe_utils import load_recipes
from retrieval.grounded_retrieval import retrieve_similar_recipes, extract_allowed_ingredients
from generation.grounded_generation import baseline_adapt, grounded_adapt
from evaluation.grounded_checker import evaluate_grounded


# === CONFIG ===
NUM_EXAMPLES = 1000       # Number of test examples
TOP_K_VALUES = [3, 5]     # Ablation: test with k=3 and k=5
MODEL = "llama3.2:3b"     # LLM model
CONSTRAINT = "vegetarian"
RANDOM_SEED = 42          # For reproducibility
PROGRESS_INTERVAL = 10    # Print progress every N examples


def run_single_experiment(test_bases, corpus, top_k):
    """
    Run experiment for a single top_k value.
    
    Returns:
        tuple: (results list, summary dict)
    """
    results = []
    
    for i, base in enumerate(test_bases, 1):
        if i % PROGRESS_INTERVAL == 0 or i == 1:
            print(f"  [{i}/{len(test_bases)}] {base['name'][:40]}...")
        
        try:
            # Retrieve similar recipes
            retrieved = retrieve_similar_recipes(base, corpus, top_k=top_k)
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
                "example_id": i,
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
            
        except Exception as e:
            print(f"  ERROR on example {i}: {e}")
            # Store error result
            results.append({
                "example_id": i,
                "base_recipe": {"name": base["name"], "ingredients": base["ingredients"]},
                "error": str(e)
            })
    
    # Compute summary (excluding errors)
    valid_results = [r for r in results if "error" not in r]
    summary = compute_summary(valid_results) if valid_results else None
    
    return results, summary


def compute_summary(results):
    """Compute aggregate statistics from results."""
    n = len(results)
    if n == 0:
        return None
    
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


def save_results(results, summary, top_k, timestamp):
    """Save results to JSON file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    filename = results_dir / f"grounded_exp_k{top_k}_{timestamp}.json"
    
    output = {
        "metadata": {
            "timestamp": timestamp,
            "config": {
                "num_examples": NUM_EXAMPLES,
                "top_k": top_k,
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
    
    return filename


def print_summary(summary, top_k):
    """Print summary statistics."""
    if summary is None:
        print("  No valid results to summarize.")
        return
        
    print(f"\n{'='*60}")
    print(f"SUMMARY (k={top_k})")
    print(f"{'='*60}")
    print(f"Examples: {summary['num_examples']}")
    print()
    
    print("                        | Baseline | Grounded | Δ        |")
    print("-" * 60)
    
    b = summary["baseline"]
    g = summary["grounded"]
    
    # Constraint violations
    c_delta = g['constraint_violation_rate'] - b['constraint_violation_rate']
    print(f"Constraint violations   | {b['constraint_violation_count']:>8} | {g['constraint_violation_count']:>8} | {g['constraint_violation_count'] - b['constraint_violation_count']:>+8} |")
    print(f"Constraint rate         | {b['constraint_violation_rate']:>7.1%} | {g['constraint_violation_rate']:>7.1%} | {c_delta:>+7.1%} |")
    
    # Grounding violations
    gr_delta = g['grounding_violation_rate'] - b['grounding_violation_rate']
    print(f"Grounding violations    | {b['grounding_violation_count']:>8} | {g['grounding_violation_count']:>8} | {g['grounding_violation_count'] - b['grounding_violation_count']:>+8} |")
    print(f"Grounding rate          | {b['grounding_violation_rate']:>7.1%} | {g['grounding_violation_rate']:>7.1%} | {gr_delta:>+7.1%} |")
    
    # Avg novel ingredients
    novel_delta = g['avg_novel_ingredients'] - b['avg_novel_ingredients']
    print(f"Avg novel ingredients   | {b['avg_novel_ingredients']:>8.2f} | {g['avg_novel_ingredients']:>8.2f} | {novel_delta:>+8.2f} |")
    
    # Reduction percentage
    if b['avg_novel_ingredients'] > 0:
        reduction = (b['avg_novel_ingredients'] - g['avg_novel_ingredients']) / b['avg_novel_ingredients'] * 100
        print(f"\nNovel ingredient reduction: {reduction:.1f}%")
    
    print(f"\n{'='*60}\n")


def main():
    print(f"\n{'='*70}")
    print("GROUNDED EXPERIMENT: Baseline vs Grounded (Ablation Study)")
    print(f"{'='*70}")
    print(f"Config: {NUM_EXAMPLES} examples, k values: {TOP_K_VALUES}, model={MODEL}")
    print(f"{'='*70}\n")
    
    # Set seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Load data
    print("[1/4] Loading data...")
    data = load_recipes(dataset="recipepairs")
    pairs = data["pairs"]
    
    # Build corpus from all targets (vegetarian recipes)
    corpus = [pair["target"] for pair in pairs]
    print(f"  Corpus: {len(corpus)} vegetarian recipes")
    
    # Extract base recipes
    base_recipes = [pair["base"] for pair in pairs]
    print(f"  Base recipes: {len(base_recipes)} meat dishes")
    
    # Sample test examples (same for all k values)
    print(f"\n[2/4] Sampling {NUM_EXAMPLES} test examples...")
    if NUM_EXAMPLES > len(base_recipes):
        print(f"  WARNING: Requested {NUM_EXAMPLES} but only {len(base_recipes)} available. Using all.")
        test_bases = base_recipes
    else:
        test_bases = random.sample(base_recipes, NUM_EXAMPLES)
    print(f"  Sampled {len(test_bases)} examples")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = {}
    
    # Run experiment for each k value
    for k_idx, top_k in enumerate(TOP_K_VALUES, 1):
        print(f"\n[3/4] Running experiment with k={top_k} ({k_idx}/{len(TOP_K_VALUES)})...")
        
        results, summary = run_single_experiment(test_bases, corpus, top_k)
        
        # Save results
        filename = save_results(results, summary, top_k, timestamp)
        print(f"  Saved: {filename}")
        
        # Print summary
        print_summary(summary, top_k)
        
        all_summaries[top_k] = summary
    
    # Final comparison
    print(f"\n[4/4] Ablation Comparison...")
    print(f"\n{'='*70}")
    print("ABLATION COMPARISON: Effect of k on grounding")
    print(f"{'='*70}")
    print(f"\n{'k':>5} | {'Baseline Avg Novel':>20} | {'Grounded Avg Novel':>20} | {'Reduction':>12} |")
    print("-" * 70)
    for k in TOP_K_VALUES:
        s = all_summaries.get(k)
        if s:
            b_novel = s['baseline']['avg_novel_ingredients']
            g_novel = s['grounded']['avg_novel_ingredients']
            reduction = (b_novel - g_novel) / b_novel * 100 if b_novel > 0 else 0
            print(f"{k:>5} | {b_novel:>20.2f} | {g_novel:>20.2f} | {reduction:>11.1f}% |")
    print(f"\n{'='*70}")
    print("Experiment complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
