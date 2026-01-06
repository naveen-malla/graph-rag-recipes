"""Grounding violation checker for recipe adaptation evaluation."""
import re
import spacy
from ingredient_parser import parse_ingredient


# === CONFIG ===
USE_LEMMATIZATION = True  # Toggle: normalize "tomatoes" → "tomato"

# Meat keywords for constraint violation check (vegetarian constraint)
MEAT_KEYWORDS = [
    "chicken", "beef", "pork", "lamb", "turkey", "bacon",
    "ham", "sausage", "meat", "steak", "veal", "duck",
    "prosciutto", "salami", "pepperoni", "chorizo"
]

# Load spaCy model (lazy loading)
_nlp = None


def _get_nlp():
    """Lazy load spaCy model."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def normalize_ingredient(ingredient, use_lemma=None):
    """
    Normalize ingredient string for matching.
    
    Args:
        ingredient: Raw ingredient string (e.g., "diced tomatoes")
        use_lemma: Override global USE_LEMMATIZATION setting
    
    Returns:
        Normalized string (e.g., "dice tomato" if lemmatized, else "diced tomatoes")
    """
    if use_lemma is None:
        use_lemma = USE_LEMMATIZATION
    
    # Basic normalization: lowercase, strip whitespace
    normalized = ingredient.lower().strip()
    
    if use_lemma:
        nlp = _get_nlp()
        doc = nlp(normalized)
        # Lemmatize each token and rejoin
        normalized = " ".join([token.lemma_ for token in doc])
    
    return normalized


def extract_ingredients_from_text(text):
    """
    Extract ingredient names from recipe text using NER.
    
    Uses ingredient-parser-nlp library for proper ingredient extraction.
    Focuses on the Ingredients section if present.
    
    Args:
        text: Generated recipe text
    
    Returns:
        List of extracted ingredient name strings
    """
    ingredients = []
    seen = set()  # Track seen ingredients to avoid duplicates
    
    # Garbage words to filter out (headers, artifacts, equipment)
    GARBAGE_WORDS = {
        "ingredients", "ingredient", "steps", "step", "instructions", 
        "instruction", "directions", "direction", "method", "note", "notes",
        "tip", "tips", "serves", "serving", "servings", "yield", "prep",
        "saucepan", "pan", "pot", "bowl", "skillet", "oven", "baking sheet",
        "adapted", "recipe", "vegetarian", "vegan", "original"
    }
    
    def add_ingredient(name):
        """Add ingredient if not already seen and not garbage."""
        name_lower = name.lower().strip()
        if not name_lower:
            return
        # Skip garbage words
        if name_lower in GARBAGE_WORDS:
            return
        # Skip very short names (likely parsing errors)
        if len(name_lower) < 2:
            return
        # Skip names that look like sentences (contain many words)
        if len(name_lower.split()) > 4:
            return
        if name_lower not in seen:
            seen.add(name_lower)
            ingredients.append(name)
    
    def parse_line(line):
        """Try to parse a line as an ingredient."""
        line = line.strip()
        if not line:
            return
        
        # Remove bullet points, numbers, dashes at start
        line = re.sub(r"^[\d\.\-\*\•\[\]]+\s*", "", line)
        if not line:
            return
        
        # Skip lines that look like headers or instructions
        if line.endswith(":") or line.startswith("#"):
            return
        
        # Skip lines that are clearly instructions (start with verbs)
        instruction_starters = [
            "cook", "bake", "mix", "stir", "add", "combine", "heat", "place",
            "pour", "serve", "let", "bring", "reduce", "simmer", "boil", "fry",
            "saute", "chop", "dice", "slice", "preheat", "set", "cover", "remove",
            "in", "the", "this", "you", "for", "with", "here", "note", "tip"
        ]
        first_word = line.split()[0].lower() if line.split() else ""
        if first_word in instruction_starters:
            return
        
        try:
            parsed = parse_ingredient(line)
            if parsed.name:
                for name_part in parsed.name:
                    if name_part.text:
                        add_ingredient(name_part.text)
        except Exception:
            pass
    
    # Step 1: Strip markdown formatting (** for bold, etc.)
    clean_text = re.sub(r"\*\*", "", text)
    clean_text = re.sub(r"__", "", clean_text)
    
    # Step 2: Try to find and parse the Ingredients section
    ingredients_section = re.search(
        r"[Ii]ngredients?:?\s*\n(.*?)(?:\n\s*[Ss]teps?:?|\n\s*[Ii]nstructions?:?|\n\s*[Dd]irections?:?|\n\s*[Mm]ethod:?|$)",
        clean_text,
        re.DOTALL
    )
    
    if ingredients_section:
        # Parse only the ingredients section
        section_text = ingredients_section.group(1)
        for line in section_text.split("\n"):
            parse_line(line)
    else:
        # No clear section found - parse all lines but be more conservative
        for line in clean_text.split("\n"):
            parse_line(line)
    
    return ingredients


def normalize_ingredient_set(ingredients, use_lemma=None):
    """
    Normalize a set/list of ingredients for matching.
    
    Args:
        ingredients: Iterable of ingredient strings
        use_lemma: Override global USE_LEMMATIZATION setting
    
    Returns:
        Set of normalized ingredient strings
    """
    return {normalize_ingredient(ing, use_lemma) for ing in ingredients}


def check_constraint_violations(text):
    """
    Check if generated recipe violates vegetarian constraint (contains meat).
    
    Args:
        text: Generated recipe text
    
    Returns:
        dict: {
            "violated": bool,
            "violations": list of found meat ingredients,
            "count": number of violations
        }
    """
    text_lower = text.lower()
    
    found_meat = []
    for keyword in MEAT_KEYWORDS:
        if keyword in text_lower:
            found_meat.append(keyword)
    
    return {
        "violated": len(found_meat) > 0,
        "violations": found_meat,
        "count": len(found_meat)
    }


def check_grounding_violations(extracted_ingredients, allowed_ingredients, use_lemma=None):
    """
    Check if extracted ingredients are grounded in allowed set.
    
    An ingredient is a "grounding violation" if it appears in the output
    but is NOT in the allowed set (from retrieved recipes).
    
    Note: We do NOT include base recipe ingredients in allowed set.
    The whole point of grounding is: LLM uses ONLY retrieved evidence.
    
    Args:
        extracted_ingredients: List of ingredient strings from generated recipe
        allowed_ingredients: Set of allowed ingredient strings (from retrieved recipes)
        use_lemma: Override global USE_LEMMATIZATION setting
    
    Returns:
        dict: {
            "violated": bool,
            "novel_ingredients": list of ingredients not in allowed set,
            "count": number of novel ingredients,
            "extracted_count": total ingredients extracted from output
        }
    """
    # Normalize all sets
    extracted_norm = {normalize_ingredient(ing, use_lemma): ing for ing in extracted_ingredients}
    allowed_norm = normalize_ingredient_set(allowed_ingredients, use_lemma)
    
    # Find novel ingredients (not in allowed set)
    novel = []
    for norm_ing, orig_ing in extracted_norm.items():
        if norm_ing not in allowed_norm:
            novel.append(orig_ing)
    
    return {
        "violated": len(novel) > 0,
        "novel_ingredients": novel,
        "count": len(novel),
        "extracted_count": len(extracted_ingredients)
    }


def evaluate_grounded(text, allowed_ingredients, use_lemma=None):
    """
    Full evaluation of generated recipe for grounding and constraint violations.
    
    Args:
        text: Generated recipe text
        allowed_ingredients: Set of allowed ingredients (from retrieved recipes)
        use_lemma: Override global USE_LEMMATIZATION setting
    
    Returns:
        dict: {
            "extracted_ingredients": list of ingredients found in output,
            "constraint_check": constraint violation results,
            "grounding_check": grounding violation results,
            "overall_valid": bool (no constraint violations)
        }
    """
    # Extract ingredients from generated text
    extracted = extract_ingredients_from_text(text)
    
    # Check constraint violations (meat in vegetarian recipe)
    constraint_result = check_constraint_violations(text)
    
    # Check grounding violations (novel ingredients)
    grounding_result = check_grounding_violations(
        extracted,
        allowed_ingredients,
        use_lemma
    )
    
    return {
        "extracted_ingredients": extracted,
        "constraint_check": constraint_result,
        "grounding_check": grounding_result,
        "overall_valid": not constraint_result["violated"]
    }
