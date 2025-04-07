#!/usr/bin/env python
# recipe_model.py
import ast
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
custom_stopwords = stop_words.union({
    # Measurement units and common descriptors
    'c', 'cup', 'cups', 'tsp', 'teaspoon', 'teaspoons', 'tbsp', 'tablespoon', 'tablespoons',
    'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds', 'g', 'gram', 'grams', 'kg',
    'ml', 'milliliter', 'milliliters', 'l', 'liter', 'liters', 'qt', 'quart', 'quarts',
    'pt', 'pint', 'pints', 'gal', 'gallon', 'gallons', 'pkg', 'pkgs', 'package', 'packages',
    'stick', 'sticks', 'dash', 'pinch', 'can', 'cans', 'fluid', 'fl', 'jar', 'jars',
    'box', 'boxes', 'bottle', 'bottles', 't', 'tbs', 'tbls', 'qt.', 'pt.', 'oz.', 'lb.', 'g.', 'ml.', 'kg.', 'l.', 'pkg.', 'pkt',
    # Preparation and cooking descriptors
    'chopped', 'minced', 'diced', 'sliced', 'grated', 'crushed', 'shredded', 'cut',
    'peeled', 'optional', 'seeded', 'halved', 'coarsely', 'finely', 'thinly', 'roughly',
    'cubed', 'crumbled', 'ground', 'trimmed', 'boneless', 'skinless', 'melted', 'softened',
    'cooled', 'boiled', 'cooked', 'uncooked', 'raw', 'drained', 'rinsed', 'beaten',
    # Quantity descriptors
    'small', 'medium', 'large', 'extra', 'light', 'dark', 'best', 'fresh', 'freshly',
    'ripe', 'mini', 'whole', 'big', 'room', 'temperature', 'zero', 'one', 'two', 'three',
    'four', 'five', 'six', 'eight', 'ten', 'twelve', 'half', 'third', 'quarter', 'dozen',
    'thousand',
    # Generic stopwords
    'plus', 'with', 'without', 'into', 'about', 'of', 'the', 'to', 'for', 'in', 'from',
    'as', 'and', 'or', 'on', 'your', 'if', 'such', 'you', 'use', 'may'
})

def preprocess_ingredients(ingredients):
    """
    Clean and standardize the ingredients list.
    """
    try:
        if isinstance(ingredients, str):
            # Evaluate string if it starts with a list indicator
            ingredients_list = ast.literal_eval(ingredients) if ingredients.strip().startswith("[") else [ingredients]
        elif isinstance(ingredients, list):
            ingredients_list = ingredients
        else:
            return ""
    
        cleaned_ingredients = set()
        for ing in ingredients_list:
            ing = re.sub(r'\(.*?\)', '', str(ing)).lower() 
            ing = re.sub(r'[^a-z\s]', '', ing)
            tokens = word_tokenize(ing)
            tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in custom_stopwords and len(token) > 1]
            if tokens:
                phrase = " ".join(tokens)
                # Optionally filter out overly generic words
                if "oil" not in phrase and "salt" not in phrase and "water" not in phrase:
                    cleaned_ingredients.add(phrase)
    
        return ", ".join(sorted(cleaned_ingredients))
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return ""

def preprocess_user_ingredients(user_input):
    """
    Process raw user input into a cleaned string of ingredients.
    """
    ingredients = user_input.split(',')
    ingredients_str = str([ing.strip() for ing in ingredients])
    return preprocess_ingredients(ingredients_str)

def find_similar_recipes(user_input, vectorizer, nn_model, train_df):
    """
    Finds and returns the top similar recipes based on the user's ingredients.
    """
    user_cleaned = preprocess_user_ingredients(user_input)
    user_vector = vectorizer.transform([user_cleaned])
    distances, indices = nn_model.kneighbors(user_vector)
    
    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        similarity_score = round((1 - dist) * 100, 2)
        title = train_df.iloc[idx]['title'] if 'title' in train_df.columns else 'N/A'
        ingredients = train_df.iloc[idx]['ingredients']
        ingredients_clean = train_df.iloc[idx]['ingredients_clean']
        results.append({
            "rank": i + 1,
            "title": title,
            "similarity": similarity_score,
            "ingredients": ingredients,
            "ingredients_clean": ingredients_clean
        })
    return results