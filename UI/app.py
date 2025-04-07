#!/usr/bin/env python
import streamlit as st
import sqlite3
import pandas as pd
import joblib
from recipe_model import find_similar_recipes

# -------------------------------
# Database Search Functions
# -------------------------------
def get_connection():
    conn = sqlite3.connect("database/pantrypalette.db")
    return conn

def search_recipes_db(ingredients):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Convert input ingredients to lowercase and build a search query
    search_query = f"%{ingredients.lower()}%"
    
    # Fetch recipes that contain the user's ingredients
    query = """
    SELECT * FROM recipes WHERE lower(ingredients) LIKE ?
    """
    cursor.execute(query, (search_query,))
    results = cursor.fetchall()
    conn.close()
    return results
    
# -------------------------------
# ML Model Setup (for recommendation)
# -------------------------------
# Load pre-trained TF-IDF vectorizer, Nearest Neighbors model, and training data
try:
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    nn_model = joblib.load("models/nearest_neighbors_model.pkl")
    train_df = pd.read_csv("processed_dataset/train_data.csv")
except Exception as e:
    st.error(f"Error loading ML model components: {e}")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🥘 PantryPalette: Your Ingredient-Inspired Recipe Guide")

# Allow users to select the search method
search_method = st.sidebar.radio("Select Search Method", ("Database Search", "ML Recommendation"))

# User Input for Ingredients
st.subheader("Enter Ingredients You Have")
user_input = st.text_input("Example: tomato, cheese, pasta")

# Search Button: Perform search based on selected method
if st.button("Find Recipes"):
    if user_input:
        if search_method == "Database Search":
            recipes = search_recipes_db(user_input)
            if recipes:
                st.success(f"Let’s get cooking, Here are the delicious recipes just for you! 🍽️ ")
                for idx, recipe in enumerate(recipes[:10]):
                    st.subheader(f"{idx + 1}. {recipe[0]}")  # Recipe Title
                    st.write(f"**Ingredients:** {recipe[1]}")
                    st.write(f"**Instructions:** {recipe[2]}")
                st.write("Enjoy Your Perfect Recipe! 🍽️")
            else:
                st.warning("⚠️ No recipes found. Try different ingredients!")
        else:  # ML Recommendation method
            results = find_similar_recipes(user_input, vectorizer, nn_model, train_df)
            if results:
                for result in results:
                    st.write(f"**{result['rank']}. {result['title']}**")
                    st.write(f"Similarity: {result['similarity']}%")
                    st.write(f"Ingredients: {result['ingredients']}")
                    st.write("---")
            else:
                st.write("No matching recipes found.")
    else:
        st.error("❌ Please enter at least one ingredient.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
    <hr style="height:1px;border:none;color:#333;background-color:#333;" />
    <p class='footer'>
        Developed by 
        <b><a href="https://www.linkedin.com/in/sandhya-kilari" target="_blank" style="text-decoration: none; color: #0072b1;">Sandhya Kilari</a></b> and 
        <b><a href="https://www.linkedin.com/in/madhurya-shankar-7344541b2" target="_blank" style="text-decoration: none; color: #0072b1;">Madhurya Shankar</a></b>
        <br>
        © 2025 PantryPalette: Your Ingredient-Inspired Recipe Guide
    </p>
""", unsafe_allow_html=True)