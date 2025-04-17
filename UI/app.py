#!/usr/bin/env python
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
import joblib
import base64
import os
import sqlite3

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="PantryPalette - Your Recipe Guide",
    page_icon="🍳",  # Using a simple emoji instead of complex Unicode
    layout="wide"
)

# -------------------------------
# Database Search Functions
# -------------------------------
def get_connection():
    conn = sqlite3.connect("https://drive.google.com/file/d/1FRNAtJiw8hpnvul1LqX3Bjjk4fttevoG/view?usp=drive_link")
    return conn

# ---------------------------------------------
# Load Trained Models (TF-IDF + Nearest Neighbors)
# ---------------------------------------------
@st.cache_resource
def load_similarity_models():
    # TF-IDF Vectorizer
    tfidf_url = "https://drive.google.com/uc?export=download&id=1hTaVi9ZB2pxMFQ5MOwD3Ozc8raf31pL8"
    response1 = requests.get(tfidf_url)
    vectorizer = joblib.load(BytesIO(response1.content))

    # Nearest Neighbors Model
    nn_url = "https://drive.google.com/uc?export=download&id=1uSIeGdZYyZt_gTzZWvI9HuArZ2zNx-Az" 
    response2 = requests.get(nn_url)
    nn_model = joblib.load(BytesIO(response2.content))

    return vectorizer, nn_model

tfidf_vectorizer, nearest_neighbors_model = load_similarity_models()

# ---------------------------------------------
# Basic Ingredient Matching (Simple Overlap)
# ---------------------------------------------
def search_recipes_db(ingredients):
    """Search recipes in the database based on ingredients"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Split ingredients into list
    ingredient_list = [ing.strip().lower() for ing in ingredients.split(',')]
    
    # Build the query
    query = """
        SELECT 
            title,
            ingredients,
            instructions
        FROM recipes
        WHERE LOWER(ingredients) LIKE ?
    """
    
    # Add more conditions for each ingredient
    for _ in range(len(ingredient_list) - 1):
        query += " OR LOWER(ingredients) LIKE ?"
    
    # Prepare parameters
    params = ['%' + ing + '%' for ing in ingredient_list]
    
    try:
        cursor.execute(query, params)
        recipes = cursor.fetchall()
        
        # Convert to DataFrame
        df = pd.DataFrame(recipes, columns=['title', 'ingredients', 'instructions'])
        
        # Calculate match scores
        df['match_score'] = df['ingredients'].apply(lambda x: calculate_match_score(x, ingredient_list))
        
        # Sort by match score
        df = df.sort_values('match_score', ascending=False)
        
        return df
    except Exception as e:
        st.error(f"Error searching recipes: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def calculate_match_score(ingredients, ingredient_list):
    matched_ingredients = sum(1 for ing in ingredient_list if ing in ingredients)
    ingredient_score = matched_ingredients / len(ingredient_list)
    return ingredient_score

# ---------------------------------------------
# Smart Similarity Search (TF-IDF + Nearest Neighbors)
# ---------------------------------------------
def search_recipes_with_similarity(ingredients):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Fetch all recipe titles
        cursor.execute("SELECT title, ingredients, instructions FROM recipes")
        recipes = cursor.fetchall()
        df = pd.DataFrame(recipes, columns=['title', 'ingredients', 'instructions'])

        # Vectorize user ingredients
        user_vec = tfidf_vectorizer.transform([ingredients])

        # Find top 10 nearest recipes
        distances, indices = nearest_neighbors_model.kneighbors(user_vec, n_neighbors=10)

        # Get recipes based on the indices
        matched_recipes = df.iloc[indices[0]].copy()

        # Add similarity score
        matched_recipes['similarity_score'] = 1 - distances[0]  # (closer distance = higher similarity)

        # Sort by similarity score
        matched_recipes = matched_recipes.sort_values('similarity_score', ascending=False)

        return matched_recipes
    except Exception as e:
        st.error(f"Error searching recipes with similarity: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

# ---------------------------------------------
# UI Setup and Styling
# ---------------------------------------------

def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

set_background("UI/image/background.png")

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .title-container {
        background-color: rgba(0,0,0,0.7);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        backdrop-filter: blur(5px);
    }
    .recipe-card {
        background-color: rgba(255,255,255,0.98);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        height: 100%;
    }
    .recipe-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #333;
    }
    .recipe-meta {
        font-size: 0.9rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .ingredient-item {
        display: inline-block;
        background-color: #f8f9fa;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0;
        font-size: 0.95rem;
        white-space: normal;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        width: 100%;
        text-align: left;
        border: 1px solid #e9ecef;
        color: #495057;
    }
    .recipe-instructions {
        font-size: 0.95rem;
        color: #444;
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
        line-height: 1.6;
    }
    .accuracy-card {
        background-color: rgba(255,255,255,0.98);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #ff4b4b;
    }
    .search-container {
        background-color: rgba(255,255,255,0.95);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .instruction-step {
        margin-bottom: 12px;
        padding: 8px;
        background-color: #f8f9fa;
        border-radius: 5px;
        line-height: 1.5;
    }
    .instruction-step strong {
        color: #ff4b4b;
        margin-right: 8px;
    }
    </style>
""", unsafe_allow_html=True)

def format_ingredients(ingredients_str):
    try:
        ingredients = eval(ingredients_str)
        if isinstance(ingredients, list):
            cleaned_ingredients = []
            for ing in ingredients:
                ing = ing.strip().replace('"', '').replace("'", '')
                if ing:
                    cleaned_ingredients.append(ing)
            ingredients_html = []
            for i in range(0, len(cleaned_ingredients), 2):
                if i + 1 < len(cleaned_ingredients):
                    ingredients_html.append(f"""
                        <div style="display: flex; margin-bottom: 12px;">
                            <div style="flex: 1; margin-right: 12px;">
                                <span class="ingredient-item">{cleaned_ingredients[i]}</span>
                            </div>
                            <div style="flex: 1;">
                                <span class="ingredient-item">{cleaned_ingredients[i+1]}</span>
                            </div>
                        </div>
                    """)
                else:
                    ingredients_html.append(f"""
                        <div style="display: flex; margin-bottom: 12px;">
                            <div style="flex: 1;">
                                <span class="ingredient-item">{cleaned_ingredients[i]}</span>
                            </div>
                        </div>
                    """)
            return ''.join(ingredients_html)
    except:
        ingredients = [ing.strip() for ing in ingredients_str.split() if ing.strip()]
        return ''.join([f'<span class="ingredient-item">{ing}</span>' for ing in ingredients])

def display_header():
    header_container = st.container()
    with header_container:
        st.markdown("""
            <div class="title-container">
                <h1 style='color: white; font-size: 3rem;'>🥘 PantryPalette</h1>
                <p style='color: white; font-size: 1.2rem;'>From pantry to plate, your wish is our command! 🍳🥗</p>
            </div>
        """, unsafe_allow_html=True)

def clean_instruction(step):
    step = step.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", '')
    step = re.sub(r'^\d+\.?\s*$', '', step)
    step = re.sub(r'^\d+\.\s*', '', step)
    step = step.lstrip(',')
    step = re.sub(r'\s+\d+\s*$', '', step)
    step = re.sub(r'^\d+\s+', '', step)
    step = re.sub(r'\s+\d+\s+', ' ', step)
    if step:
        step = step[0].upper() + step[1:]
    return step.strip()

def display_recipe_card(recipe, show_accuracy=False):
    """Display a recipe card with ingredients and instructions"""
    with st.container():
        st.markdown(f"""
            <div class="recipe-card">
                <div class="recipe-title"> Recipe: {recipe['title']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display ingredients in a clean format
        st.markdown("### 📝 Ingredients")
        try:
            ingredients = eval(recipe['ingredients']) if isinstance(recipe['ingredients'], str) else []
            if isinstance(ingredients, list):
                # Create two columns for ingredients
                col1, col2 = st.columns(2)
                for i, ingredient in enumerate(ingredients):
                    ingredient = ingredient.strip().replace('"', '').replace("'", '')
                    if ingredient:
                        ingredient_box = f"""
                            <div class="ingredient-item">{ingredient}</div>
                        """
                        if i % 2 == 0:
                            col1.markdown(ingredient_box, unsafe_allow_html=True)
                        else:
                            col2.markdown(ingredient_box, unsafe_allow_html=True)
        except:
            st.write("No ingredients available")
        
        # Display instructions
        st.markdown("### 👩‍🍳 Instructions")
        try:
            instructions = eval(recipe['instructions']) if isinstance(recipe['instructions'], str) else []
            if isinstance(instructions, list):
                step_number = 1
                for step in instructions:
                    # Clean up the instruction text
                    cleaned_step = clean_instruction(step)
                    if cleaned_step:  # Only display non-empty steps
                        st.markdown(f"""
                            <div class="instruction-step">
                                <strong>{step_number}.</strong> {cleaned_step}
                            </div>
                        """, unsafe_allow_html=True)
                        step_number += 1
        except:
            st.write("No instructions available")

def display_accuracy_card(recipe):
    st.markdown(f"""
        <div class="accuracy-card">
            <div class="recipe-title">{recipe['title']}</div>
            <div class="recipe-meta">
                <strong>🎯 Match Score:</strong> {recipe['match_score']*100:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_recipes(matching_recipes, similarity_based=False):
    if not matching_recipes.empty:
        if similarity_based:
            st.success(f"Here are your top recipe matches based on ingredient similarity! 🍽️")
        else:
            st.success(f"Here are recipes based on your entered ingredients! 🍽️")
        
        tab1, tab2 = st.tabs(["Top Recipes", "Recipe Similarity" if similarity_based else "Recipe Accuracy"])
        
        with tab1:
            for i in range(0, min(5, len(matching_recipes)), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < min(5, len(matching_recipes)):
                        with cols[j]:
                            display_recipe_card(matching_recipes.iloc[i + j])
        
        with tab2:
            if similarity_based:
                st.markdown("### Top 10 Similar Recipe Matches")
                for _, recipe in matching_recipes.head(10).iterrows():
                    st.markdown(f"""
                        <div class="accuracy-card">
                            <div class="recipe-title">{recipe['title']}</div>
                            <div class="recipe-meta">
                                <strong>🔍 Similarity Score:</strong> {recipe['similarity_score']*100:.1f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("### Top 10 Recipe Matches")
                for _, recipe in matching_recipes.head(10).iterrows():
                    display_accuracy_card(recipe)
    else:
        st.warning("⚠️ No recipes found. Try different ingredients!")

def display_search_section():
    st.markdown("""
        <div class="search-container">
            <h3>Enter your ingredients</h3>
            <p style='color: #666;'>Separate ingredients with commas</p>
        </div>
    """, unsafe_allow_html=True)
    
    ## st.markdown("<h4><strong>🔍 Search Ingredients</strong></h4>", unsafe_allow_html=True)

    ingredients = st.text_input(label = "", 
                                placeholder="Example: tomato, cheese, pasta",
                                help="Enter ingredients separated by commas",
                                key="ingredient_search")

    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Get Recipes", use_container_width=True):
            st.session_state['search_clicked'] = True  # Save that user clicked
    
    return ingredients

def main():
    display_header()
    ingredients = display_search_section()
    
    if st.session_state.get('search_clicked', False):
        if not ingredients.strip():
            st.warning("⚠️ Please enter one or more ingredients to search for recipes.")
            return
        
        search_mode = st.radio(
            "Choose search method:",
            ["Simple Ingredient Match (Overlap Match)", "Smart Similarity Match (TFIDF + Nearest Neighbors)"],
            horizontal=True
        )

        if search_mode.startswith("Simple Ingredient Match"):
            matching_recipes = search_recipes_db(ingredients)
            display_recipes(matching_recipes, similarity_based=False)
        else:
            matching_recipes = search_recipes_with_similarity(ingredients)
            display_recipes(matching_recipes, similarity_based=True)

    # Footer
    st.markdown("""
        <hr style="height:1px;border:none;color:black;background-color:black;" />
        <p style='text-align: center; color: black;'>
            ©️ 2025 PantryPalette: Your Ingredient-Inspired Recipe Guide<br>
            Developed by 
            <b><a href="https://www.linkedin.com/in/sandhya-kilari" target="_blank" style="text-decoration: none; color: #0072b1;">Sandhya Kilari</a></b> and 
            <b><a href="https://www.linkedin.com/in/madhurya-shankar-7344541b2" target="_blank" style="text-decoration: none; color: #0072b1;">Madhurya Shankar</a></b>
        </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
