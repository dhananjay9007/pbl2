import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
import altair as alt

@st.cache_data
def load_data(path='survey_data.csv'): # Corrected default path
    return pd.read_csv(path)

st.set_page_config(layout="wide", page_title="Survey Analytics Dashboard")

st.title("Survey Analytics — Classification, Association Rules, Clustering, Regression")

# Load data - Simplified to load from the same directory
df = load_data('survey_data.csv')

with st.sidebar:
    st.header("Filters")
    _min, _max = float(df['response_id'].min()), float(df['response_id'].max())
    response_id_range = st.slider('response_id range', _min, _max, (_min, _max))
    age_group_choices = st.multiselect('age_group', options=df['age_group'].dropna().unique().tolist(), default=df['age_group'].dropna().unique().tolist())
    gender_choices = st.multiselect('gender', options=df['gender'].dropna().unique().tolist(), default=df['gender'].dropna().unique().tolist())
    employment_status_choices = st.multiselect('employment_status', options=df['employment_status'].dropna().unique().tolist(), default=df['employment_status'].dropna().unique().tolist())
    income_choices = st.multiselect('income', options=df['income'].dropna().unique().tolist(), default=df['income'].dropna().unique().tolist())
    education_choices = st.multiselect('education', options=df['education'].dropna().unique().tolist(), default=df['education'].dropna().unique().tolist())
    location_type_choices = st.multiselect('location_type', options=df['location_type'].dropna().unique().tolist(), default=df['location_type'].dropna().unique().tolist())
    household_size_choices = st.multiselect('household_size', options=df['household_size'].dropna().unique().tolist(), default=df['household_size'].dropna().unique().tolist())
    _min, _max = float(df['health_consciousness'].min()), float(df['health_consciousness'].max())
    health_consciousness_range = st.slider('health_consciousness range', _min, _max, (_min, _max))
    exercise_frequency_choices = st.multiselect('exercise_frequency', options=df['exercise_frequency'].dropna().unique().tolist(), default=df['exercise_frequency'].dropna().unique().tolist())
    fitness_goal_choices = st.multiselect('fitness_goal', options=df['fitness_goal'].dropna().unique().tolist(), default=df['fitness_goal'].dropna().unique().tolist())
    _min, _max = float(df['hydration_importance'].min()), float(df['hydration_importance'].max())
    hydration_importance_range = st.slider('hydration_importance range', _min, _max, (_min, _max))
    daily_water_intake_choices = st.multiselect('daily_water_intake', options=df['daily_water_intake'].dropna().unique().tolist(), default=df['daily_water_intake'].dropna().unique().tolist())
    bottle_type_choices = st.multiselect('bottle_type', options=df['bottle_type'].dropna().unique().tolist(), default=df['bottle_type'].dropna().unique().tolist())
    monthly_beverage_spend_choices = st.multiselect('monthly_beverage_spend', options=df['monthly_beverage_spend'].dropna().unique().tolist(), default=df['monthly_beverage_spend'].dropna().unique().tolist())
    _min, _max = float(df['interest_level'].min()), float(df['interest_level'].max())
    interest_level_range = st.slider('interest_level range', _min, _max, (_min, _max))
    purchase_likelihood_choices = st.multiselect('purchase_likelihood', options=df['purchase_likelihood'].dropna().unique().tolist(), default=df['purchase_likelihood'].dropna().unique().tolist())
    _min, _max = float(df['willingness_to_pay_continuous'].min()), float(df['willingness_to_pay_continuous'].max())
    willingness_to_pay_continuous_range = st.slider('willingness_to_pay_continuous range', _min, _max, (_min, _max))
    willingness_to_pay_category_choices = st.multiselect('willingness_to_pay_category', options=df['willingness_to_pay_category'].dropna().unique().tolist(), default=df['willingness_to_pay_category'].dropna().unique().tolist())
    weekly_usage_choices = st.multiselect('weekly_usage', options=df['weekly_usage'].dropna().unique().tolist(), default=df['weekly_usage'].dropna().unique().tolist())
    purchase_preference_choices = st.multiselect('purchase_preference', options=df['purchase_preference'].dropna().unique().tolist(), default=df['purchase_preference'].dropna().unique().tolist())
    _min, _max = float(df['sustainability_importance'].min()), float(df['sustainability_importance'].max())
    sustainability_importance_range = st.slider('sustainability_importance range', _min, _max, (_min, _max))
    _min, _max = float(df['early_adopter_score'].min()), float(df['early_adopter_score'].max())
    early_adopter_score_range = st.slider('early_adopter_score range', _min, _max, (_min, _max))
    _min, _max = float(df['premium_willingness_score'].min()), float(df['premium_willingness_score'].max())
    premium_willingness_score_range = st.slider('premium_willingness_score range', _min, _max, (_min, _max))
    
    # --- FIXED VARIABLE NAMES START ---
    _min, _max = float(df['health_condition_Diabetes (Type 1 or 2)'].min()), float(df['health_condition_Diabetes (Type 1 or 2)'].max())
    health_condition_Diabetes_Type_1_or_2_range = st.slider('health_condition_Diabetes (Type 1 or 2) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_High blood pressure'].min()), float(df['health_condition_High blood pressure'].max())
    health_condition_High_blood_pressure_range = st.slider('health_condition_High blood pressure range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_High cholesterol'].min()), float(df['health_condition_High cholesterol'].max())
    health_condition_High_cholesterol_range = st.slider('health_condition_High cholesterol range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_Heart disease'].min()), float(df['health_condition_Heart disease'].max())
    health_condition_Heart_disease_range = st.slider('health_condition_Heart disease range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_Kidney disease'].min()), float(df['health_condition_Kidney disease'].max())
    health_condition_Kidney_disease_range = st.slider('health_condition_Kidney disease range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_Food allergies'].min()), float(df['health_condition_Food allergies'].max())
    health_condition_Food_allergies_range = st.slider('health_condition_Food allergies range', _min, _max, (_min, _max))
    
    _min, _max = float(df['health_condition_None of the above'].min()), float(df['health_condition_None of the above'].max())
    health_condition_None_of_the_above_range = st.slider('health_condition_None of the above range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_I forget to drink'].min()), float(df['barrier_I forget to drink'].max())
    barrier_I_forget_to_drink_range = st.slider('barrier_I forget to drink range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_Plain water is boring/tasteless'].min()), float(df['barrier_Plain water is boring/tasteless'].max())
    barrier_Plain_water_is_boring_tasteless_range = st.slider('barrier_Plain water is boring/tasteless range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_I dont feel thirsty'].min()), float(df['barrier_I dont feel thirsty'].max())
    barrier_I_dont_feel_thirsty_range = st.slider('barrier_I dont feel thirsty range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_Its inconvenient to carry water'].min()), float(df['barrier_Its inconvenient to carry water'].max())
    barrier_Its_inconvenient_to_carry_water_range = st.slider('barrier_Its inconvenient to carry water range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_I prefer other beverages'].min()), float(df['barrier_I prefer
