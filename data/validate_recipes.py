import json
import re 

def load_recipes(filepath):
    """Load recipes from a JSON file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data['recipes']

def check_ingredient_coverage(recipe):
    """Check if the graph ingredients appear in text"""
    text = recipe['text'].lower()
    graph_ingredients = [ing['name'].lower() for ing in recipe['graph']['ingredients']]

    issues = []
    for ingredient in graph_ingredients:
        # Handle plural forms by checking both singular and plural
        if ingredient not in text and ingredient + 's' not in text:# 
            issues.append(f"Ingredient '{ingredient}' not found in text.")
    return issues

def check_action_coverage(recipe):
    """Check if action verbs appear in text"""
    text = recipe['text'].lower()
    action_verbs = [action['verb'].lower() for action in recipe['graph']['actions']]        

    issues = []
    for verb in action_verbs:
        if verb not in text:
            issues.append(f"Action verb '{verb}' not found in text.")
    return issues

def check_graph_structure(recipe):
    """Validate bipartite graph structure"""
    issues = []

    # Get all valid node IDs
    ingredient_ids = {ing['id'] for ing in recipe['graph']['ingredients']}
    action_ids = {action['id'] for action in recipe['graph']['actions']}

    # Check edges are action to ingredient
    for edge in recipe['graph']['edges']:
        src, tgt = edge['source'], edge['target']
        if src in ingredient_ids:
            issues.append(f"Edge source '{src}' is an ingredient, should be an action.")
        if tgt in action_ids:
            issues.append(f"Edge target '{tgt}' is an action, should be an ingredient.")

        # Check role is valid
        if edge['role'] not in ['input', 'output']:
            issues.append(f"Edge role '{edge['role']}' is invalid. Must be 'input' or 'output'.")

    return issues

def validate_recipe(recipe):
    """Run all validation checks on a recipe"""
    print(f"\n{'='*60}")
    print(f"Validating: {recipe['name']} (ID: {recipe['id']})")
    print(f"{'='*60}")
    
    all_passed = True
    
    # Run checks
    checks = [
        ("Ingredient Coverage", check_ingredient_coverage),
        ("Action Coverage", check_action_coverage),
        ("Graph Structure", check_graph_structure),
    ]
    
    for check_name, check_func in checks:
        issues = check_func(recipe)
        if issues:
            all_passed = False
            print(f"\n❌ {check_name} FAILED:")
            for issue in issues:
                print(f"   • {issue}")
        else:
            print(f"✅ {check_name} passed")
    
    return all_passed

if __name__ == "__main__":
    recipes = load_recipes("./data/recipes.json")
    
    print(f"Found {len(recipes)} recipe(s) to validate\n")
    
    all_valid = True
    for recipe in recipes:
        valid = validate_recipe(recipe)
        if not valid:
            all_valid = False
    
    print(f"\n{'='*60}")
    if all_valid:
        print("✅ ALL RECIPES VALID")
    else:
        print("❌ SOME RECIPES HAVE ISSUES - Review above")
    print(f"{'='*60}")