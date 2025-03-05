# ! pip install streamlit sqlite3 pandas

import streamlit as st
import sqlite3
import pandas as pd

# Connect to SQLite Database
def get_connection():
    conn = sqlite3.connect("pantrypalette.db")
    return conn

# Fetch recipes based on user input ingredients
def search_recipes(ingredients):
    conn = get_connection()
    cursor = conn.cursor()
    
    # Convert input ingredients to lowercase
    search_query = f"%{ingredients.lower()}%"
    
    # Fetch recipes that contain the user's ingredients
    query = """
    SELECT * FROM recipes WHERE lower(ingredients) LIKE ?
    """
    cursor.execute(query, (search_query,))
    results = cursor.fetchall()
    
    conn.close()
    return results

# Streamlit UI
st.title("🥘 PantryPalette: Your Ingredient-Inspired Recipe Guide")

# User Input: Ingredients
st.subheader("Enter Ingredients You Have")
user_input = st.text_input("Example: tomato, cheese, pasta")

# Search Button
if st.button("Find Recipes"):
    if user_input:
        recipes = search_recipes(user_input)
        
        if recipes:
            st.success(f"We found {len(recipes)} delicious recipes just for you! 🍽️ Let’s get cooking!")
            for idx, recipe in enumerate(recipes):
                st.subheader(f"{idx + 1}. {recipe[1]}")  # Recipe Title
                st.write(f"**Ingredients:** {recipe[2]}")
            
            st.write(f"Enjoy Your Perfect Recipe! 🍽️")
            
        else:
            st.warning("⚠️ No recipes found. Try different ingredients!")
    else:
        st.error("❌ Please enter at least one ingredient.")

# Footer
st.markdown("""
    <hr style="height:1px;border:none;color:#333;background-color:#333;" />
    <p class='footer'>
        Developed by 
        <b><a href="https://www.linkedin.com/in/sandhya-kilari" target="_blank" style="text-decoration: none; color: #0072b1;">Sandhya Kilari</a></b> and 
        <b><a href="https://www.linkedin.com/in/madhurya-shankar-7344541b2" target="_blank" style="text-decoration: none; color: #0072b1;">Madhurya Shankar</a></b>
        <br>
        © 2024 Tribal Funding Registry
    </p>
""", unsafe_allow_html=True)