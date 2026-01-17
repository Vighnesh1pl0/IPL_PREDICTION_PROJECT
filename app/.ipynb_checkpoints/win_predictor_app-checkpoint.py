#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and feature columns
model = joblib.load("../models/best_winprob_model.pkl")
x_cols = joblib.load("../models/x_columns.pkl")

st.title("🏏 IPL Win Probability Predictor")

# Team selection
teams = ['Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore', 
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Rajasthan Royals',
         'Delhi Capitals', 'Sunrisers Hyderabad', 'Other']

team1 = st.selectbox("Select Batting Team", teams)
team2 = st.selectbox("Select Bowling Team", teams)

# Numeric inputs
pp_runs = st.number_input("Powerplay Runs (0-50)", min_value=0, max_value=50, value=0)
mid_runs = st.number_input("Middle Overs Runs (0-100)", min_value=0, max_value=100, value=0)
death_runs = st.number_input("Death Overs Runs (0-100)", min_value=0, max_value=100, value=0)

pp_wickets = st.number_input("Powerplay Wickets (0-10)", min_value=0, max_value=10, value=0)
mid_wickets = st.number_input("Middle Overs Wickets (0-10)", min_value=0, max_value=10, value=0)
death_wickets = st.number_input("Death Overs Wickets (0-10)", min_value=0, max_value=10, value=0)

extras = st.number_input("Extras", min_value=0, max_value=50, value=0)

# Predict button
if st.button("Predict Win Probability"):
    # Prepare input dataframe
    inp = pd.DataFrame(np.zeros((1, len(x_cols))), columns=x_cols)
    
    # Assign numeric values
    for col, val in zip(['pp_runs','mid_runs','death_runs','pp_wickets','mid_wickets','death_wickets','extras'],
                        [pp_runs, mid_runs, death_runs, pp_wickets, mid_wickets, death_wickets, extras]):
        if col in inp.columns:
            inp[col] = val
    
    # Assign team dummies
    t1_col = f"team1_{team1}"
    t2_col = f"team2_{team2}"
    if t1_col in inp.columns: inp[t1_col] = 1
    if t2_col in inp.columns: inp[t2_col] = 1
    
    # Prediction
    prob = model.predict_proba(inp)[:,1][0]
    st.success(f"Win Probability for {team1}: {prob*100:.2f}%")


# In[ ]:




