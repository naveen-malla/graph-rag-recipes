"""Quick timing test for 5 examples."""
import random
import time
from utils.recipe_utils import load_recipes
from retrieval.grounded_retrieval import retrieve_similar_recipes, extract_allowed_ingredients
from generation.grounded_generation import baseline_adapt, grounded_adapt
from evaluation.grounded_checker import evaluate_grounded

random.seed(42)
data = load_recipes(dataset='recipepairs')
pairs = data['pairs']
corpus = [p['target'] for p in pairs]
bases = [p['base'] for p in pairs]
test = random.sample(bases, 5)

print('Testing 5 examples with k=3...')
start = time.time()

for i, base in enumerate(test, 1):
    t0 = time.time()
    retrieved = retrieve_similar_recipes(base, corpus, top_k=3)
    retrieved_recipes = [r for r, _ in retrieved]
    allowed = extract_allowed_ingredients(retrieved)
    
    out_b = baseline_adapt(base, retrieved_recipes, 'vegetarian', 'llama3.2:3b')
    out_g = grounded_adapt(base, retrieved_recipes, 'vegetarian', 'llama3.2:3b')
    
    eval_b = evaluate_grounded(out_b, allowed)
    eval_g = evaluate_grounded(out_g, allowed)
    
    elapsed = time.time() - t0
    print(f'  [{i}/5] {base["name"][:30]} - {elapsed:.1f}s - B:{eval_b["grounding_check"]["count"]} G:{eval_g["grounding_check"]["count"]}')

total = time.time() - start
print(f'\nTotal: {total:.1f}s for 5 examples')
print(f'Avg per example: {total/5:.1f}s (2 LLM calls each)')
print(f'Estimated for 500 examples (k=3 + k=5): {total/5 * 500 * 2 / 3600:.1f} hours')
