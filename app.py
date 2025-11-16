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
    # Make sure 'survey_data.csv' is in the same folder as this script
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: Could not find the data file at {path}.")
        st.error("Please make sure 'survey_data.csv' is in the same directory as 'app.py'.")
        return pd.DataFrame() # Return an empty DataFrame to avoid further errors

st.set_page_config(layout="wide", page_title="Survey Analytics Dashboard")

st.title("Survey Analytics — Classification, Association Rules, Clustering, Regression")

# Load data
df = load_data('survey_data.csv')

# Stop the app if the data wasn't loaded
if df.empty:
    st.stop()

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
    # I have replaced spaces, (), and / with underscores (_)
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
    
    _min, _max = float(df['barrier_I prefer other beverages'].min()), float(df['barrier_I prefer other beverages'].max())
    barrier_I_prefer_other_beverages_range = st.slider('barrier_I prefer other beverages range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_Health reasons (frequent bathroom trips, etc.)'].min()), float(df['barrier_Health reasons (frequent bathroom trips, etc.)'].max())
    barrier_Health_reasons_frequent_bathroom_trips_etc_range = st.slider('barrier_Health reasons (frequent bathroom trips, etc.) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['barrier_Nothing prevents me'].min()), float(df['barrier_Nothing prevents me'].max())
    barrier_Nothing_prevents_me_range = st.slider('barrier_Nothing prevents me range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_At home'].min()), float(df['consume_location_At home'].max())
    consume_location_At_home_range = st.slider('consume_location_At home range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_At work/school'].min()), float(df['consume_location_At work/school'].max())
    consume_location_At_work_school_range = st.slider('consume_location_At work/school range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_At the gym'].min()), float(df['consume_location_At the gym'].max())
    consume_location_At_the_gym_range = st.slider('consume_location_At the gym range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_During commute'].min()), float(df['consume_location_During commute'].max())
    consume_location_During_commute_range = st.slider('consume_location_During commute range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_Restaurants/cafes'].min()), float(df['consume_location_Restaurants/cafes'].max())
    consume_location_Restaurants_cafes_range = st.slider('consume_location_Restaurants/cafes range', _min, _max, (_min, _max))
    
    _min, _max = float(df['consume_location_Outdoor activities'].min()), float(df['consume_location_Outdoor activities'].max())
    consume_location_Outdoor_activities_range = st.slider('consume_location_Outdoor activities range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Plain water'].min()), float(df['beverage_Plain water'].max())
    beverage_Plain_water_range = st.slider('beverage_Plain water range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Flavored water (e.g., LaCroix, Hint)'].min()), float(df['beverage_Flavored water (e.g., LaCroix, Hint)'].max())
    beverage_Flavored_water_eg_LaCroix_Hint_range = st.slider('beverage_Flavored water (e.g., LaCroix, Hint) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Sports drinks (e.g., Gatorade, Powerade)'].min()), float(df['beverage_Sports drinks (e.g., Gatorade, Powerade)'].max())
    beverage_Sports_drinks_eg_Gatorade_Powerade_range = st.slider('beverage_Sports drinks (e.g., Gatorade, Powerade) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Energy drinks (e.g., Red Bull, Monster)'].min()), float(df['beverage_Energy drinks (e.g., Red Bull, Monster)'].max())
    beverage_Energy_drinks_eg_Red_Bull_Monster_range = st.slider('beverage_Energy drinks (e.g., Red Bull, Monster) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)'].min()), float(df['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)'].max())
    beverage_Enhanced_water_eg_Vitaminwater_Smartwater_range = st.slider('beverage_Enhanced water (e.g., Vitaminwater, Smartwater) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Coffee'].min()), float(df['beverage_Coffee'].max())
    beverage_Coffee_range = st.slider('beverage_Coffee range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Tea'].min()), float(df['beverage_Tea'].max())
    beverage_Tea_range = st.slider('beverage_Tea range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Soda/soft drinks'].min()), float(df['beverage_Soda/soft drinks'].max())
    beverage_Soda_soft_drinks_range = st.slider('beverage_Soda/soft drinks range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Juice'].min()), float(df['beverage_Juice'].max())
    beverage_Juice_range = st.slider('beverage_Juice range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Protein shakes'].min()), float(df['beverage_Protein shakes'].max())
    beverage_Protein_shakes_range = st.slider('beverage_Protein shakes range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Pre-workout drinks'].min()), float(df['beverage_Pre-workout drinks'].max())
    beverage_Pre_workout_drinks_range = st.slider('beverage_Pre-workout drinks range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)'].min()), float(df['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)'].max())
    beverage_Electrolyte_tablets_powders_eg_Nuun_Liquid_IV_range = st.slider('beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)'].min()), float(df['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)'].max())
    beverage_Squeeze_flavor_enhancers_eg_MiO_Crystal_Light_range = st.slider('beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Kombucha'].min()), float(df['beverage_Kombucha'].max())
    beverage_Kombucha_range = st.slider('beverage_Kombucha range', _min, _max, (_min, _max))
    
    _min, _max = float(df['beverage_Coconut water'].min()), float(df['beverage_Coconut water'].max())
    beverage_Coconut_water_range = st.slider('beverage_Coconut water range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Taste/flavor variety'].min()), float(df['priority_Taste/flavor variety'].max())
    priority_Taste_flavor_variety_range = st.slider('priority_Taste/flavor variety range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Nutritional benefits (vitamins, minerals)'].min()), float(df['priority_Nutritional benefits (vitamins, minerals)'].max())
    priority_Nutritional_benefits_vitamins_minerals_range = st.slider('priority_Nutritional benefits (vitamins, minerals) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Low/zero sugar'].min()), float(df['priority_Low/zero sugar'].max())
    priority_Low_zero_sugar_range = st.slider('priority_Low/zero sugar range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Natural ingredients'].min()), float(df['priority_Natural ingredients'].max())
    priority_Natural_ingredients_range = st.slider('priority_Natural ingredients range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Caffeine content'].min()), float(df['priority_Caffeine content'].max())
    priority_Caffeine_content_range = st.slider('priority_Caffeine content range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Electrolytes'].min()), float(df['priority_Electrolytes'].max())
    priority_Electrolytes_range = st.slider('priority_Electrolytes range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Protein content'].min()), float(df['priority_Protein content'].max())
    priority_Protein_content_range = st.slider('priority_Protein content range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Convenience/portability'].min()), float(df['priority_Convenience/portability'].max())
    priority_Convenience_portability_range = st.slider('priority_Convenience/portability range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Price/affordability'].min()), float(df['priority_Price/affordability'].max())
    priority_Price_affordability_range = st.slider('priority_Price/affordability range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Brand reputation'].min()), float(df['priority_Brand reputation'].max())
    priority_Brand_reputation_range = st.slider('priority_Brand reputation range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_Environmental sustainability'].min()), float(df['priority_Environmental sustainability'].max())
    priority_Environmental_sustainability_range = st.slider('priority_Environmental sustainability range', _min, _max, (_min, _max))
    
    _min, _max = float(df['priority_No artificial ingredients'].min()), float(df['priority_No artificial ingredients'].max())
    priority_No_artificial_ingredients_range = st.slider('priority_No artificial ingredients range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_Cirkul (flavor cartridge bottle)'].min()), float(df['used_product_Cirkul (flavor cartridge bottle)'].max())
    used_product_Cirkul_flavor_cartridge_bottle_range = st.slider('used_product_Cirkul (flavor cartridge bottle) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_Air Up (scent-based bottle)'].min()), float(df['used_product_Air Up (scent-based bottle)'].max())
    used_product_Air_Up_scent_based_bottle_range = st.slider('used_product_Air Up (scent-based bottle) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_MiO/liquid flavor drops'].min()), float(df['used_product_MiO/liquid flavor drops'].max())
    used_product_MiO_liquid_flavor_drops_range = st.slider('used_product_MiO/liquid flavor drops range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_Dissolvable tablets (Nuun, Liquid I.V.)'].min()), float(df['used_product_Dissolvable tablets (Nuun, Liquid I.V.)'].max())
    used_product_Dissolvable_tablets_Nuun_Liquid_IV_range = st.slider('used_product_Dissolvable tablets (Nuun, Liquid I.V.) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_Powder packets (Crystal Light, etc.)'].min()), float(df['used_product_Powder packets (Crystal Light, etc.)'].max())
    used_product_Powder_packets_Crystal_Light_etc_range = st.slider('used_product_Powder packets (Crystal Light, etc.) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_Pre-flavored bottled water'].min()), float(df['used_product_Pre-flavored bottled water'].max())
    used_product_Pre_flavored_bottled_water_range = st.slider('used_product_Pre-flavored bottled water range', _min, _max, (_min, _max))
    
    _min, _max = float(df['used_product_None of the above'].min()), float(df['used_product_None of the above'].max())
    used_product_None_of_the_above_range = st.slider('used_product_None of the above range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Convenience/ease of use'].min()), float(df['appealing_Convenience/ease of use'].max())
    appealing_Convenience_ease_of_use_range = st.slider('appealing_Convenience/ease of use range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Portion control'].min()), float(df['appealing_Portion control'].max())
    appealing_Portion_control_range = st.slider('appealing_Portion control range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_No mixing required'].min()), float(df['appealing_No mixing required'].max())
    appealing_No_mixing_required_range = st.slider('appealing_No mixing required range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Flavor variety'].min()), float(df['appealing_Flavor variety'].max())
    appealing_Flavor_variety_range = st.slider('appealing_Flavor variety range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Nutritional customization'].min()), float(df['appealing_Nutritional customization'].max())
    appealing_Nutritional_customization_range = st.slider('appealing_Nutritional customization range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Portability'].min()), float(df['appealing_Portability'].max())
    appealing_Portability_range = st.slider('appealing_Portability range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Hygiene (single-use)'].min()), float(df['appealing_Hygiene (single-use)'].max())
    appealing_Hygiene_single_use_range = st.slider('appealing_Hygiene (single-use) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Helps me drink more water'].min()), float(df['appealing_Helps me drink more water'].max())
    appealing_Helps_me_drink_more_water_range = st.slider('appealing_Helps me drink more water range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Health benefits'].min()), float(df['appealing_Health benefits'].max())
    appealing_Health_benefits_range = st.slider('appealing_Health benefits range', _min, _max, (_min, _max))
    
    _min, _max = float(df['appealing_Nothing appeals to me'].min()), float(df['appealing_Nothing appeals to me'].max())
    appealing_Nothing_appeals_to_me_range = st.slider('appealing_Nothing appeals to me range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Price/cost per use'].min()), float(df['concern_Price/cost per use'].max())
    concern_Price_cost_per_use_range = st.slider('concern_Price/cost per use range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Environmental waste (single-use plastic)'].min()), float(df['concern_Environmental waste (single-use plastic)'].max())
    concern_Environmental_waste_single_use_plastic_range = st.slider('concern_Environmental waste (single-use plastic) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Powder might not dissolve well'].min()), float(df['concern_Powder might not dissolve well'].max())
    concern_Powder_might_not_dissolve_well_range = st.slider('concern_Powder might not dissolve well range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Artificial ingredients'].min()), float(df['concern_Artificial ingredients'].max())
    concern_Artificial_ingredients_range = st.slider('concern_Artificial ingredients range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Limited flavor options'].min()), float(df['concern_Limited flavor options'].max())
    concern_Limited_flavor_options_range = st.slider('concern_Limited flavor options range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Not compatible with my bottle'].min()), float(df['concern_Not compatible with my bottle'].max())
    concern_Not_compatible_with_my_bottle_range = st.slider('concern_Not compatible with my bottle range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_Prefer other methods'].min()), float(df['concern_Prefer other methods'].max())
    concern_Prefer_other_methods_range = st.slider('concern_Prefer other methods range', _min, _max, (_min, _max))
    
    _min, _max = float(df['concern_No concerns'].min()), float(df['concern_No concerns'].max())
    concern_No_concerns_range = st.slider('concern_No concerns range', _min, _max, (_min, _max))
    
    price_15_perception_choices = st.multiselect('price_15_perception', options=df['price_15_perception'].dropna().unique().tolist(), default=df['price_15_perception'].dropna().unique().tolist())
    price_25_perception_choices = st.multiselect('price_25_perception', options=df['price_25_perception'].dropna().unique().tolist(), default=df['price_25_perception'].dropna().unique().tolist())
    price_35_perception_choices = st.multiselect('price_35_perception', options=df['price_35_perception'].dropna().unique().tolist(), default=df['price_35_perception'].dropna().unique().tolist())
    price_45_perception_choices = st.multiselect('price_45_perception', options=df['price_45_perception'].dropna().unique().tolist(), default=df['price_45_perception'].dropna().unique().tolist())
    price_60_perception_choices = st.multiselect('price_60_perception', options=df['price_60_perception'].dropna().unique().tolist(), default=df['price_60_perception'].dropna().unique().tolist())
    
    _min, _max = float(df['flavor_Citrus (lemon, lime, orange)'].min()), float(df['flavor_Citrus (lemon, lime, orange)'].max())
    flavor_Citrus_lemon_lime_orange_range = st.slider('flavor_Citrus (lemon, lime, orange) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Berry (strawberry, blueberry, raspberry)'].min()), float(df['flavor_Berry (strawberry, blueberry, raspberry)'].max())
    flavor_Berry_strawberry_blueberry_raspberry_range = st.slider('flavor_Berry (strawberry, blueberry, raspberry) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Tropical (mango, pineapple, coconut)'].min()), float(df['flavor_Tropical (mango, pineapple, coconut)'].max())
    flavor_Tropical_mango_pineapple_coconut_range = st.slider('flavor_Tropical (mango, pineapple, coconut) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Mint/herbal'].min()), float(df['flavor_Mint/herbal'].max())
    flavor_Mint_herbal_range = st.slider('flavor_Mint/herbal range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Green tea/matcha'].min()), float(df['flavor_Green tea/matcha'].max())
    flavor_Green_tea_matcha_range = st.slider('flavor_Green tea/matcha range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Coffee-flavored'].min()), float(df['flavor_Coffee-flavored'].max())
    flavor_Coffee_flavored_range = st.slider('flavor_Coffee-flavored range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Neutral/unflavored (just nutrients)'].min()), float(df['flavor_Neutral/unflavored (just nutrients)'].max())
    flavor_Neutral_unflavored_just_nutrients_range = st.slider('flavor_Neutral/unflavored (just nutrients) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Sour/tangy'].min()), float(df['flavor_Sour/tangy'].max())
    flavor_Sour_tangy_range = st.slider('flavor_Sour/tangy range', _min, _max, (_min, _max))
    
    _min, _max = float(df['flavor_Sweet/dessert-inspired'].min()), float(df['flavor_Sweet/dessert-inspired'].max())
    flavor_Sweet_dessert_inspired_range = st.slider('flavor_Sweet/dessert-inspired range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Energy boost (caffeine, B-vitamins)'].min()), float(df['benefit_rank_Energy boost (caffeine, B-vitamins)'].max())
    benefit_rank_Energy_boost_caffeine_B_vitamins_range = st.slider('benefit_rank_Energy boost (caffeine, B-vitamins) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Hydration/electrolytes'].min()), float(df['benefit_rank_Hydration/electrolytes'].max())
    benefit_rank_Hydration_electrolytes_range = st.slider('benefit_rank_Hydration/electrolytes range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Immunity support (Vitamin C, zinc)'].min()), float(df['benefit_rank_Immunity support (Vitamin C, zinc)'].max())
    benefit_rank_Immunity_support_Vitamin_C_zinc_range = st.slider('benefit_rank_Immunity support (Vitamin C, zinc) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Focus/mental clarity'].min()), float(df['benefit_rank_Focus/mental clarity'].max())
    benefit_rank_Focus_mental_clarity_range = st.slider('benefit_rank_Focus/mental clarity range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Recovery (post-workout)'].min()), float(df['benefit_rank_Recovery (post-workout)'].max())
    benefit_rank_Recovery_post_workout_range = st.slider('benefit_rank_Recovery (post-workout) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Digestive health (probiotics)'].min()), float(df['benefit_rank_Digestive health (probiotics)'].max())
    benefit_rank_Digestive_health_probiotics_range = st.slider('benefit_rank_Digestive health (probiotics) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Antioxidants'].min()), float(df['benefit_rank_Antioxidants'].max())
    benefit_rank_Antioxidants_range = st.slider('benefit_rank_Antioxidants range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Skin health (collagen, biotin)'].min()), float(df['benefit_rank_Skin health (collagen, biotin)'].max())
    benefit_rank_Skin_health_collagen_biotin_range = st.slider('benefit_rank_Skin health (collagen, biotin) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_Weight management'].min()), float(df['benefit_rank_Weight management'].max())
    benefit_rank_Weight_management_range = st.slider('benefit_rank_Weight management range', _min, _max, (_min, _max))
    
    _min, _max = float(df['benefit_rank_General vitamins/minerals'].min()), float(df['benefit_rank_General vitamins/minerals'].max())
    benefit_rank_General_vitamins_minerals_range = st.slider('benefit_rank_General vitamins/minerals range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Sugar-free (for diabetics)'].min()), float(df['specialized_Sugar-free (for diabetics)'].max())
    specialized_Sugar_free_for_diabetics_range = st.slider('specialized_Sugar-free (for diabetics) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Keto-friendly'].min()), float(df['specialized_Keto-friendly'].max())
    specialized_Keto_friendly_range = st.slider('specialized_Keto-friendly range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Vegan'].min()), float(df['specialized_Vegan'].max())
    specialized_Vegan_range = st.slider('specialized_Vegan range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Organic/natural only'].min()), float(df['specialized_Organic/natural only'].max())
    specialized_Organic_natural_only_range = st.slider('specialized_Organic/natural only range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Allergen-free (gluten, dairy, soy)'].min()), float(df['specialized_Allergen-free (gluten, dairy, soy)'].max())
    specialized_Allergen_free_gluten_dairy_soy_range = st.slider('specialized_Allergen-free (gluten, dairy, soy) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Kid-friendly formulas'].min()), float(df['specialized_Kid-friendly formulas'].max())
    specialized_Kid_friendly_formulas_range = st.slider('specialized_Kid-friendly formulas range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Senior-optimized (bone health, etc.)'].min()), float(df['specialized_Senior-optimized (bone health, etc.)'].max())
    specialized_Senior_optimized_bone_health_etc_range = st.slider('specialized_Senior-optimized (bone health, etc.) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_Athletic performance'].min()), float(df['specialized_Athletic performance'].max())
    specialized_Athletic_performance_range = st.slider('specialized_Athletic performance range', _min, _max, (_min, _max))
    
    _min, _max = float(df['specialized_None - prefer standard versions'].min()), float(df['specialized_None - prefer standard versions'].max())
    specialized_None_prefer_standard_versions_range = st.slider('specialized_None - prefer standard versions range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Social media (Instagram, TikTok, Facebook)'].min()), float(df['discovery_Social media (Instagram, TikTok, Facebook)'].max())
    discovery_Social_media_Instagram_TikTok_Facebook_range = st.slider('discovery_Social media (Instagram, TikTok, Facebook) range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Friends/family recommendations'].min()), float(df['discovery_Friends/family recommendations'].max())
    discovery_Friends_family_recommendations_range = st.slider('discovery_Friends/family recommendations range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Health/fitness influencers'].min()), float(df['discovery_Health/fitness influencers'].max())
    discovery_Health_fitness_influencers_range = st.slider('discovery_Health/fitness influencers range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Online reviews'].min()), float(df['discovery_Online reviews'].max())
    discovery_Online_reviews_range = st.slider('discovery_Online reviews range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_In-store displays'].min()), float(df['discovery_In-store displays'].max())
    discovery_In_store_displays_range = st.slider('discovery_In-store displays range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_TV/online ads'].min()), float(df['discovery_TV/online ads'].max())
    discovery_TV_online_ads_range = st.slider('discovery_TV/online ads range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Health blogs/websites'].min()), float(df['discovery_Health blogs/websites'].max())
    discovery_Health_blogs_websites_range = st.slider('discovery_Health blogs/websites range', _min, _max, (_min, _max))
    
    _min, _max = float(df['discovery_Nutritionist/doctor recommendation'].min()), float(df['discovery_Nutritionist/doctor recommendation'].max())
    discovery_Nutritionist_doctor_recommendation_range = st.slider('discovery_Nutritionist/doctor recommendation range', _min, _max, (_min, _max))
    # --- FIXED VARIABLE NAMES END ---

# Apply filters
df_filtered = df.copy()
df_filtered = df_filtered[(df_filtered['response_id']>= response_id_range[0]) & (df_filtered['response_id']<= response_id_range[1])]
if len(age_group_choices) > 0:
    df_filtered = df_filtered[df_filtered['age_group'].isin(age_group_choices)]
if len(gender_choices) > 0:
    df_filtered = df_filtered[df_filtered['gender'].isin(gender_choices)]
if len(employment_status_choices) > 0:
    df_filtered = df_filtered[df_filtered['employment_status'].isin(employment_status_choices)]
if len(income_choices) > 0:
    df_filtered = df_filtered[df_filtered['income'].isin(income_choices)]
if len(education_choices) > 0:
    df_filtered = df_filtered[df_filtered['education'].isin(education_choices)]
if len(location_type_choices) > 0:
    df_filtered = df_filtered[df_filtered['location_type'].isin(location_type_choices)]
if len(household_size_choices) > 0:
    df_filtered = df_filtered[df_filtered['household_size'].isin(household_size_choices)]
df_filtered = df_filtered[(df_filtered['health_consciousness']>= health_consciousness_range[0]) & (df_filtered['health_consciousness']<= health_consciousness_range[1])]
if len(exercise_frequency_choices) > 0:
    df_filtered = df_filtered[df_filtered['exercise_frequency'].isin(exercise_frequency_choices)]
if len(fitness_goal_choices) > 0:
    df_filtered = df_filtered[df_filtered['fitness_goal'].isin(fitness_goal_choices)]
df_filtered = df_filtered[(df_filtered['hydration_importance']>= hydration_importance_range[0]) & (df_filtered['hydration_importance']<= hydration_importance_range[1])]
if len(daily_water_intake_choices) > 0:
    df_filtered = df_filtered[df_filtered['daily_water_intake'].isin(daily_water_intake_choices)]
if len(bottle_type_choices) > 0:
    df_filtered = df_filtered[df_filtered['bottle_type'].isin(bottle_type_choices)]
if len(monthly_beverage_spend_choices) > 0:
    df_filtered = df_filtered[df_filtered['monthly_beverage_spend'].isin(monthly_beverage_spend_choices)]
df_filtered = df_filtered[(df_filtered['interest_level']>= interest_level_range[0]) & (df_filtered['interest_level']<= interest_level_range[1])]
if len(purchase_likelihood_choices) > 0:
    df_filtered = df_filtered[df_filtered['purchase_likelihood'].isin(purchase_likelihood_choices)]
df_filtered = df_filtered[(df_filtered['willingness_to_pay_continuous']>= willingness_to_pay_continuous_range[0]) & (df_filtered['willingness_to_pay_continuous']<= willingness_to_pay_continuous_range[1])]
if len(willingness_to_pay_category_choices) > 0:
    df_filtered = df_filtered[df_filtered['willingness_to_pay_category'].isin(willingness_to_pay_category_choices)]
if len(weekly_usage_choices) > 0:
    df_filtered = df_filtered[df_filtered['weekly_usage'].isin(weekly_usage_choices)]
if len(purchase_preference_choices) > 0:
    df_filtered = df_filtered[df_filtered['purchase_preference'].isin(purchase_preference_choices)]
df_filtered = df_filtered[(df_filtered['sustainability_importance']>= sustainability_importance_range[0]) & (df_filtered['sustainability_importance']<= sustainability_importance_range[1])]
df_filtered = df_filtered[(df_filtered['early_adopter_score']>= early_adopter_score_range[0]) & (df_filtered['early_adopter_score']<= early_adopter_score_range[1])]
df_filtered = df_filtered[(df_filtered['premium_willingness_score']>= premium_willingness_score_range[0]) & (df_filtered['premium_willingness_score']<= premium_willingness_score_range[1])]

# --- FIXED VARIABLE NAMES IN FILTERS ---
# Using the corrected variable names to apply the filters
df_filtered = df_filtered[(df_filtered['health_condition_Diabetes (Type 1 or 2)']>= health_condition_Diabetes_Type_1_or_2_range[0]) & (df_filtered['health_condition_Diabetes (Type 1 or 2)']<= health_condition_Diabetes_Type_1_or_2_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_High blood pressure']>= health_condition_High_blood_pressure_range[0]) & (df_filtered['health_condition_High blood pressure']<= health_condition_High_blood_pressure_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_High cholesterol']>= health_condition_High_cholesterol_range[0]) & (df_filtered['health_condition_High cholesterol']<= health_condition_High_cholesterol_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Heart disease']>= health_condition_Heart_disease_range[0]) & (df_filtered['health_condition_Heart disease']<= health_condition_Heart_disease_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Kidney disease']>= health_condition_Kidney_disease_range[0]) & (df_filtered['health_condition_Kidney disease']<= health_condition_Kidney_disease_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_Food allergies']>= health_condition_Food_allergies_range[0]) & (df_filtered['health_condition_Food allergies']<= health_condition_Food_allergies_range[1])]
df_filtered = df_filtered[(df_filtered['health_condition_None of the above']>= health_condition_None_of_the_above_range[0]) & (df_filtered['health_condition_None of the above']<= health_condition_None_of_the_above_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I forget to drink']>= barrier_I_forget_to_drink_range[0]) & (df_filtered['barrier_I forget to drink']<= barrier_I_forget_to_drink_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Plain water is boring/tasteless']>= barrier_Plain_water_is_boring_tasteless_range[0]) & (df_filtered['barrier_Plain water is boring/tasteless']<= barrier_Plain_water_is_boring_tasteless_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I dont feel thirsty']>= barrier_I_dont_feel_thirsty_range[0]) & (df_filtered['barrier_I dont feel thirsty']<= barrier_I_dont_feel_thirsty_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Its inconvenient to carry water']>= barrier_Its_inconvenient_to_carry_water_range[0]) & (df_filtered['barrier_Its inconvenient to carry water']<= barrier_Its_inconvenient_to_carry_water_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_I prefer other beverages']>= barrier_I_prefer_other_beverages_range[0]) & (df_filtered['barrier_I prefer other beverages']<= barrier_I_prefer_other_beverages_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Health reasons (frequent bathroom trips, etc.)']>= barrier_Health_reasons_frequent_bathroom_trips_etc_range[0]) & (df_filtered['barrier_Health reasons (frequent bathroom trips, etc.)']<= barrier_Health_reasons_frequent_bathroom_trips_etc_range[1])]
df_filtered = df_filtered[(df_filtered['barrier_Nothing prevents me']>= barrier_Nothing_prevents_me_range[0]) & (df_filtered['barrier_Nothing prevents me']<= barrier_Nothing_prevents_me_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At home']>= consume_location_At_home_range[0]) & (df_filtered['consume_location_At home']<= consume_location_At_home_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At work/school']>= consume_location_At_work_school_range[0]) & (df_filtered['consume_location_At work/school']<= consume_location_At_work_school_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_At the gym']>= consume_location_At_the_gym_range[0]) & (df_filtered['consume_location_At the gym']<= consume_location_At_the_gym_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_During commute']>= consume_location_During_commute_range[0]) & (df_filtered['consume_location_During commute']<= consume_location_During_commute_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_Restaurants/cafes']>= consume_location_Restaurants_cafes_range[0]) & (df_filtered['consume_location_Restaurants/cafes']<= consume_location_Restaurants_cafes_range[1])]
df_filtered = df_filtered[(df_filtered['consume_location_Outdoor activities']>= consume_location_Outdoor_activities_range[0]) & (df_filtered['consume_location_Outdoor activities']<= consume_location_Outdoor_activities_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Plain water']>= beverage_Plain_water_range[0]) & (df_filtered['beverage_Plain water']<= beverage_Plain_water_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Flavored water (e.g., LaCroix, Hint)']>= beverage_Flavored_water_eg_LaCroix_Hint_range[0]) & (df_filtered['beverage_Flavored water (e.g., LaCroix, Hint)']<= beverage_Flavored_water_eg_LaCroix_Hint_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Sports drinks (e.g., Gatorade, Powerade)']>= beverage_Sports_drinks_eg_Gatorade_Powerade_range[0]) & (df_filtered['beverage_Sports drinks (e.g., Gatorade, Powerade)']<= beverage_Sports_drinks_eg_Gatorade_Powerade_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Energy drinks (e.g., Red Bull, Monster)']>= beverage_Energy_drinks_eg_Red_Bull_Monster_range[0]) & (df_filtered['beverage_Energy drinks (e.g., Red Bull, Monster)']<= beverage_Energy_drinks_eg_Red_Bull_Monster_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)']>= beverage_Enhanced_water_eg_Vitaminwater_Smartwater_range[0]) & (df_filtered['beverage_Enhanced water (e.g., Vitaminwater, Smartwater)']<= beverage_Enhanced_water_eg_Vitaminwater_Smartwater_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Coffee']>= beverage_Coffee_range[0]) & (df_filtered['beverage_Coffee']<= beverage_Coffee_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Tea']>= beverage_Tea_range[0]) & (df_filtered['beverage_Tea']<= beverage_Tea_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Soda/soft drinks']>= beverage_Soda_soft_drinks_range[0]) & (df_filtered['beverage_Soda/soft drinks']<= beverage_Soda_soft_drinks_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Juice']>= beverage_Juice_range[0]) & (df_filtered['beverage_Juice']<= beverage_Juice_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Protein shakes']>= beverage_Protein_shakes_range[0]) & (df_filtered['beverage_Protein shakes']<= beverage_Protein_shakes_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Pre-workout drinks']>= beverage_Pre_workout_drinks_range[0]) & (df_filtered['beverage_Pre-workout drinks']<= beverage_Pre_workout_drinks_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)']>= beverage_Electrolyte_tablets_powders_eg_Nuun_Liquid_IV_range[0]) & (df_filtered['beverage_Electrolyte tablets/powders (e.g., Nuun, Liquid I.V.)']<= beverage_Electrolyte_tablets_powders_eg_Nuun_Liquid_IV_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)']>= beverage_Squeeze_flavor_enhancers_eg_MiO_Crystal_Light_range[0]) & (df_filtered['beverage_Squeeze flavor enhancers (e.g., MiO, Crystal Light)']<= beverage_Squeeze_flavor_enhancers_eg_MiO_Crystal_Light_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Kombucha']>= beverage_Kombucha_range[0]) & (df_filtered['beverage_Kombucha']<= beverage_Kombucha_range[1])]
df_filtered = df_filtered[(df_filtered['beverage_Coconut water']>= beverage_Coconut_water_range[0]) & (df_filtered['beverage_Coconut water']<= beverage_Coconut_water_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Taste/flavor variety']>= priority_Taste_flavor_variety_range[0]) & (df_filtered['priority_Taste/flavor variety']<= priority_Taste_flavor_variety_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Nutritional benefits (vitamins, minerals)']>= priority_Nutritional_benefits_vitamins_minerals_range[0]) & (df_filtered['priority_Nutritional benefits (vitamins, minerals)']<= priority_Nutritional_benefits_vitamins_minerals_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Low/zero sugar']>= priority_Low_zero_sugar_range[0]) & (df_filtered['priority_Low/zero sugar']<= priority_Low_zero_sugar_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Natural ingredients']>= priority_Natural_ingredients_range[0]) & (df_filtered['priority_Natural ingredients']<= priority_Natural_ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Caffeine content']>= priority_Caffeine_content_range[0]) & (df_filtered['priority_Caffeine content']<= priority_Caffeine_content_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Electrolytes']>= priority_Electrolytes_range[0]) & (df_filtered['priority_Electrolytes']<= priority_Electrolytes_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Protein content']>= priority_Protein_content_range[0]) & (df_filtered['priority_Protein content']<= priority_Protein_content_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Convenience/portability']>= priority_Convenience_portability_range[0]) & (df_filtered['priority_Convenience/portability']<= priority_Convenience_portability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Price/affordability']>= priority_Price_affordability_range[0]) & (df_filtered['priority_Price/affordability']<= priority_Price_affordability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Brand reputation']>= priority_Brand_reputation_range[0]) & (df_filtered['priority_Brand reputation']<= priority_Brand_reputation_range[1])]
df_filtered = df_filtered[(df_filtered['priority_Environmental sustainability']>= priority_Environmental_sustainability_range[0]) & (df_filtered['priority_Environmental sustainability']<= priority_Environmental_sustainability_range[1])]
df_filtered = df_filtered[(df_filtered['priority_No artificial ingredients']>= priority_No_artificial_ingredients_range[0]) & (df_filtered['priority_No artificial ingredients']<= priority_No_artificial_ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Cirkul (flavor cartridge bottle)']>= used_product_Cirkul_flavor_cartridge_bottle_range[0]) & (df_filtered['used_product_Cirkul (flavor cartridge bottle)']<= used_product_Cirkul_flavor_cartridge_bottle_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Air Up (scent-based bottle)']>= used_product_Air_Up_scent_based_bottle_range[0]) & (df_filtered['used_product_Air Up (scent-based bottle)']<= used_product_Air_Up_scent_based_bottle_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_MiO/liquid flavor drops']>= used_product_MiO_liquid_flavor_drops_range[0]) & (df_filtered['used_product_MiO/liquid flavor drops']<= used_product_MiO_liquid_flavor_drops_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Dissolvable tablets (Nuun, Liquid I.V.)']>= used_product_Dissolvable_tablets_Nuun_Liquid_IV_range[0]) & (df_filtered['used_product_Dissolvable tablets (Nuun, Liquid I.V.)']<= used_product_Dissolvable_tablets_Nuun_Liquid_IV_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Powder packets (Crystal Light, etc.)']>= used_product_Powder_packets_Crystal_Light_etc_range[0]) & (df_filtered['used_product_Powder packets (Crystal Light, etc.)']<= used_product_Powder_packets_Crystal_Light_etc_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_Pre-flavored bottled water']>= used_product_Pre_flavored_bottled_water_range[0]) & (df_filtered['used_product_Pre-flavored bottled water']<= used_product_Pre_flavored_bottled_water_range[1])]
df_filtered = df_filtered[(df_filtered['used_product_None of the above']>= used_product_None_of_the_above_range[0]) & (df_filtered['used_product_None of the above']<= used_product_None_of_the_above_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Convenience/ease of use']>= appealing_Convenience_ease_of_use_range[0]) & (df_filtered['appealing_Convenience/ease of use']<= appealing_Convenience_ease_of_use_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Portion control']>= appealing_Portion_control_range[0]) & (df_filtered['appealing_Portion control']<= appealing_Portion_control_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_No mixing required']>= appealing_No_mixing_required_range[0]) & (df_filtered['appealing_No mixing required']<= appealing_No_mixing_required_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Flavor variety']>= appealing_Flavor_variety_range[0]) & (df_filtered['appealing_Flavor variety']<= appealing_Flavor_variety_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Nutritional customization']>= appealing_Nutritional_customization_range[0]) & (df_filtered['appealing_Nutritional customization']<= appealing_Nutritional_customization_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Portability']>= appealing_Portability_range[0]) & (df_filtered['appealing_Portability']<= appealing_Portability_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Hygiene (single-use)']>= appealing_Hygiene_single_use_range[0]) & (df_filtered['appealing_Hygiene (single-use)']<= appealing_Hygiene_single_use_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Helps me drink more water']>= appealing_Helps_me_drink_more_water_range[0]) & (df_filtered['appealing_Helps me drink more water']<= appealing_Helps_me_drink_more_water_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Health benefits']>= appealing_Health_benefits_range[0]) & (df_filtered['appealing_Health benefits']<= appealing_Health_benefits_range[1])]
df_filtered = df_filtered[(df_filtered['appealing_Nothing appeals to me']>= appealing_Nothing_appeals_to_me_range[0]) & (df_filtered['appealing_Nothing appeals to me']<= appealing_Nothing_appeals_to_me_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Price/cost per use']>= concern_Price_cost_per_use_range[0]) & (df_filtered['concern_Price/cost per use']<= concern_Price_cost_per_use_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Environmental waste (single-use plastic)']>= concern_Environmental_waste_single_use_plastic_range[0]) & (df_filtered['concern_Environmental waste (single-use plastic)']<= concern_Environmental_waste_single_use_plastic_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Powder might not dissolve well']>= concern_Powder_might_not_dissolve_well_range[0]) & (df_filtered['concern_Powder might not dissolve well']<= concern_Powder_might_not_dissolve_well_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Artificial ingredients']>= concern_Artificial_ingredients_range[0]) & (df_filtered['concern_Artificial ingredients']<= concern_Artificial_ingredients_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Limited flavor options']>= concern_Limited_flavor_options_range[0]) & (df_filtered['concern_Limited flavor options']<= concern_Limited_flavor_options_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Not compatible with my bottle']>= concern_Not_compatible_with_my_bottle_range[0]) & (df_filtered['concern_Not compatible with my bottle']<= concern_Not_compatible_with_my_bottle_range[1])]
df_filtered = df_filtered[(df_filtered['concern_Prefer other methods']>= concern_Prefer_other_methods_range[0]) & (df_filtered['concern_Prefer other methods']<= concern_Prefer_other_methods_range[1])]
df_filtered = df_filtered[(df_filtered['concern_No concerns']>= concern_No_concerns_range[0]) & (df_filtered['concern_No concerns']<= concern_No_concerns_range[1])]
if len(price_15_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_15_perception'].isin(price_15_perception_choices)]
if len(price_25_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_25_perception'].isin(price_25_perception_choices)]
if len(price_35_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_35_perception'].isin(price_35_perception_choices)]
if len(price_45_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_45_perception'].isin(price_45_perception_choices)]
if len(price_60_perception_choices) > 0:
    df_filtered = df_filtered[df_filtered['price_60_perception'].isin(price_60_perception_choices)]
df_filtered = df_filtered[(df_filtered['flavor_Citrus (lemon, lime, orange)']>= flavor_Citrus_lemon_lime_orange_range[0]) & (df_filtered['flavor_Citrus (lemon, lime, orange)']<= flavor_Citrus_lemon_lime_orange_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Berry (strawberry, blueberry, raspberry)']>= flavor_Berry_strawberry_blueberry_raspberry_range[0]) & (df_filtered['flavor_Berry (strawberry, blueberry, raspberry)']<= flavor_Berry_strawberry_blueberry_raspberry_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Tropical (mango, pineapple, coconut)']>= flavor_Tropical_mango_pineapple_coconut_range[0]) & (df_filtered['flavor_Tropical (mango, pineapple, coconut)']<= flavor_Tropical_mango_pineapple_coconut_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Mint/herbal']>= flavor_Mint_herbal_range[0]) & (df_filtered['flavor_Mint/herbal']<= flavor_Mint_herbal_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Green tea/matcha']>= flavor_Green_tea_matcha_range[0]) & (df_filtered['flavor_Green tea/matcha']<= flavor_Green_tea_matcha_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Coffee-flavored']>= flavor_Coffee_flavored_range[0]) & (df_filtered['flavor_Coffee-flavored']<= flavor_Coffee_flavored_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Neutral/unflavored (just nutrients)']>= flavor_Neutral_unflavored_just_nutrients_range[0]) & (df_filtered['flavor_Neutral/unflavored (just nutrients)']<= flavor_Neutral_unflavored_just_nutrients_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Sour/tangy']>= flavor_Sour_tangy_range[0]) & (df_filtered['flavor_Sour/tangy']<= flavor_Sour_tangy_range[1])]
df_filtered = df_filtered[(df_filtered['flavor_Sweet/dessert-inspired']>= flavor_Sweet_dessert_inspired_range[0]) & (df_filtered['flavor_Sweet/dessert-inspired']<= flavor_Sweet_dessert_inspired_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Energy boost (caffeine, B-vitamins)']>= benefit_rank_Energy_boost_caffeine_B_vitamins_range[0]) & (df_filtered['benefit_rank_Energy boost (caffeine, B-vitamins)']<= benefit_rank_Energy_boost_caffeine_B_vitamins_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Hydration/electrolytes']>= benefit_rank_Hydration_electrolytes_range[0]) & (df_filtered['benefit_rank_Hydration/electrolytes']<= benefit_rank_Hydration_electrolytes_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Immunity support (Vitamin C, zinc)']>= benefit_rank_Immunity_support_Vitamin_C_zinc_range[0]) & (df_filtered['benefit_rank_Immunity support (Vitamin C, zinc)']<= benefit_rank_Immunity_support_Vitamin_C_zinc_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Focus/mental clarity']>= benefit_rank_Focus_mental_clarity_range[0]) & (df_filtered['benefit_rank_Focus/mental clarity']<= benefit_rank_Focus_mental_clarity_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Recovery (post-workout)']>= benefit_rank_Recovery_post_workout_range[0]) & (df_filtered['benefit_rank_Recovery (post-workout)']<= benefit_rank_Recovery_post_workout_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Digestive health (probiotics)']>= benefit_rank_Digestive_health_probiotics_range[0]) & (df_filtered['benefit_rank_Digestive health (probiotics)']<= benefit_rank_Digestive_health_probiotics_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Antioxidants']>= benefit_rank_Antioxidants_range[0]) & (df_filtered['benefit_rank_Antioxidants']<= benefit_rank_Antioxidants_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Skin health (collagen, biotin)']>= benefit_rank_Skin_health_collagen_biotin_range[0]) & (df_filtered['benefit_rank_Skin health (collagen, biotin)']<= benefit_rank_Skin_health_collagen_biotin_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_Weight management']>= benefit_rank_Weight_management_range[0]) & (df_filtered['benefit_rank_Weight management']<= benefit_rank_Weight_management_range[1])]
df_filtered = df_filtered[(df_filtered['benefit_rank_General vitamins/minerals']>= benefit_rank_General_vitamins_minerals_range[0]) & (df_filtered['benefit_rank_General vitamins/minerals']<= benefit_rank_General_vitamins_minerals_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Sugar-free (for diabetics)']>= specialized_Sugar_free_for_diabetics_range[0]) & (df_filtered['specialized_Sugar-free (for diabetics)']<= specialized_Sugar_free_for_diabetics_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Keto-friendly']>= specialized_Keto_friendly_range[0]) & (df_filtered['specialized_Keto-friendly']<= specialized_Keto_friendly_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Vegan']>= specialized_Vegan_range[0]) & (df_filtered['specialized_Vegan']<= specialized_Vegan_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Organic/natural only']>= specialized_Organic_natural_only_range[0]) & (df_filtered['specialized_Organic/natural only']<= specialized_Organic_natural_only_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Allergen-free (gluten, dairy, soy)']>= specialized_Allergen_free_gluten_dairy_soy_range[0]) & (df_filtered['specialized_Allergen-free (gluten, dairy, soy)']<= specialized_Allergen_free_gluten_dairy_soy_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Kid-friendly formulas']>= specialized_Kid_friendly_formulas_range[0]) & (df_filtered['specialized_Kid-friendly formulas']<= specialized_Kid_friendly_formulas_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Senior-optimized (bone health, etc.)']>= specialized_Senior_optimized_bone_health_etc_range[0]) & (df_filtered['specialized_Senior-optimized (bone health, etc.)']<= specialized_Senior_optimized_bone_health_etc_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_Athletic performance']>= specialized_Athletic_performance_range[0]) & (df_filtered['specialized_Athletic performance']<= specialized_Athletic_performance_range[1])]
df_filtered = df_filtered[(df_filtered['specialized_None - prefer standard versions']>= specialized_None_prefer_standard_versions_range[0]) & (df_filtered['specialized_None - prefer standard versions']<= specialized_None_prefer_standard_versions_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Social media (Instagram, TikTok, Facebook)']>= discovery_Social_media_Instagram_TikTok_Facebook_range[0]) & (df_filtered['discovery_Social media (Instagram, TikTok, Facebook)']<= discovery_Social_media_Instagram_TikTok_Facebook_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Friends/family recommendations']>= discovery_Friends_family_recommendations_range[0]) & (df_filtered['discovery_Friends/family recommendations']<= discovery_Friends_family_recommendations_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Health/fitness influencers']>= discovery_Health_fitness_influencers_range[0]) & (df_filtered['discovery_Health/fitness influencers']<= discovery_Health_fitness_influencers_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Online reviews']>= discovery_Online_reviews_range[0]) & (df_filtered['discovery_Online reviews']<= discovery_Online_reviews_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_In-store displays']>= discovery_In_store_displays_range[0]) & (df_filtered['discovery_In-store displays']<= discovery_In_store_displays_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_TV/online ads']>= discovery_TV_online_ads_range[0]) & (df_filtered['discovery_TV/online ads']<= discovery_TV_online_ads_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Health blogs/websites']>= discovery_Health_blogs_websites_range[0]) & (df_filtered['discovery_Health blogs/websites']<= discovery_Health_blogs_websites_range[1])]
df_filtered = df_filtered[(df_filtered['discovery_Nutritionist/doctor recommendation']>= discovery_Nutritionist_doctor_recommendation_range[0]) & (df_filtered['discovery_Nutritionist/doctor recommendation']<= discovery_Nutritionist_doctor_recommendation_range[1])]

st.write(f"Filtered rows: {len(df_filtered)}")
st.dataframe(df_filtered.head(50))

tab1, tab2, tab3, tab4 = st.tabs(["Classification","Association Rules","Clustering","Regression"])

with tab1:
    st.header("Classification")
    st.write("Select target (categorical) and features.")
    cat_cols = [c for c in df_filtered.columns if (df_filtered[c].nunique()<=20) and (df_filtered[c].dtype=='object' or df_filtered[c].dtype.name=='category')]
    st.write("Candidate categorical targets (<=20 unique):", cat_cols)
    target = st.selectbox("Target column (classification)", options=cat_cols)
    features = st.multiselect("Features (use numeric and categorical)", options=[c for c in df_filtered.columns if c!=target], default=[c for c in df_filtered.columns if c!=target][:5])
    
    if st.button("Run classification"):
        if not target:
            st.error("Please select a target column for classification.")
        elif not features:
            st.error("Please select at least one feature for classification.")
        else:
            sub = df_filtered[features+[target]].dropna()
            if len(sub) < 10:
                st.error("Not enough data to run classification after dropping NaNs. Please adjust filters.")
                st.stop()
                
            X = sub[features].copy()
            y = sub[target].copy()
            X_proc = X.copy()
            
            # Preprocessing features
            for col in X_proc.columns:
                if X_proc[col].dtype=='object' or X_proc[col].dtype.name=='category':
                    X_proc[col] = LabelEncoder().fit_transform(X_proc[col].astype(str))
                else:
                    # Fill NaNs in numeric columns with the mean
                    X_proc[col] = X_proc[col].astype(float).fillna(X_proc[col].mean())
            
            y_enc = LabelEncoder().fit_transform(y.astype(str))
            
            if len(np.unique(y_enc)) > 1: # Check if there is more than one class
                X_train, X_test, y_train, y_test = train_test_split(X_proc, y_enc, test_size=0.25, random_state=42, stratify=y_enc)
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)
                st.write(f"Accuracy: {acc:.3f}")
                st.text(classification_report(y_test, preds, zero_division=0))
            else:
                st.error("Classification requires more than one class in the target variable. Try adjusting your filters.")


with tab2:
    st.header("Association Rule Mining")
    st.write("Prepare transactional data: select columns to treat as one-hot encoded items (categorical).")
    # Candidates for association rules are typically categorical with a reasonable number of unique values
    cat_cols = [c for c in df_filtered.columns if (df_filtered[c].nunique() <= 50) and (df_filtered[c].dtype == 'object' or df_filtered[c].dtype.name == 'category')]
    items = st.multiselect("Columns to include as items", options=cat_cols, default=cat_cols[:5])
    min_support = st.slider("Min support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.6)
    
    if st.button("Run association rules"):
        if len(items) > 0:
            trans = df_filtered[items].astype(str).fillna('NA')
            
            # One-hot encode the selected categorical columns
            one_hot_list = []
            for col in trans.columns:
                # Add prefix to distinguish items (e.g., 'age_group_25-34')
                one_hot_list.append(pd.get_dummies(trans[col], prefix=col, dtype=bool)) 
            
            one_hot = pd.concat(one_hot_list, axis=1)
            
            if one_hot.empty:
                st.error("No data to process. Adjust filters.")
                st.stop()

            frequent = apriori(one_hot, min_support=min_support, use_colnames=True)
            
            if not frequent.empty:
                rules = association_rules(frequent, metric="confidence", min_threshold=min_conf)
                st.write("Frequent itemsets:", frequent.sort_values('support', ascending=False).head(20))
                st.write("Rules:", rules[['antecedents','consequents','support','confidence','lift']].sort_values('confidence', ascending=False).head(20))
            else:
                st.warning("No frequent itemsets found with the current settings. Try lowering the 'Min support'.")
        else:
            st.error("Please select at least one column for items.")


with tab3:
    st.header("Clustering (KMeans)")
    # Select only numeric columns for clustering
    numeric_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    st.write("Available numeric columns:", numeric_cols)
    k = st.slider("Number of clusters (k)", 2, 10, 3)
    
    # Let user select columns for clustering from the numeric list
    default_cols = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
    cols_for_k = st.multiselect("Columns to use for clustering", options=numeric_cols, default=default_cols)
    
    if st.button("Run clustering"):
        if len(cols_for_k) < 1:
            st.error("Pick at least one numeric column.")
        else:
            X = df_filtered[cols_for_k].dropna()
            if len(X) < k:
                st.error(f"Not enough data (found {len(X)} rows) to form {k} clusters. Adjust filters or lower k.")
            else:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # Added n_init=10 to suppress warning
                labels = kmeans.fit_predict(Xs)
                
                # Create DataFrame for results
                X_results = pd.DataFrame(X, columns=cols_for_k)
                X_results['cluster'] = labels
                
                # Show mean values per cluster (in original scale)
                st.write("Cluster Means (Original Scale):")
                st.dataframe(X_results.groupby('cluster').mean())
                
                # Create scatter plot if exactly 2 dimensions are chosen
                if len(cols_for_k) == 2:
                    # Add cluster label back to scaled data for plotting
                    Xv_scaled = pd.DataFrame(Xs, columns=cols_for_k)
                    Xv_scaled['cluster'] = labels

                    chart = alt.Chart(Xv_scaled).mark_circle(size=60).encode(
                        x=alt.X(cols_for_k[0], axis=alt.Axis(title=f"Scaled {cols_for_k[0]}")),
                        y=alt.Y(cols_for_k[1], axis=alt.Axis(title=f"Scaled {cols_for_k[1]}")),
                        color=alt.Color('cluster:N', title='Cluster'), # 'cluster:N' treats cluster as Nominal (categorical)
                        tooltip=cols_for_k + ['cluster']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

with tab4:
    st.header("Regression")
    st.write("Select target (numeric) and features.")
    num_cols = [c for c in df_filtered.columns if pd.api.types.is_numeric_dtype(df_filtered[c])]
    st.write("Available numeric columns:", num_cols)
    
    target_r = st.selectbox("Target column (regression)", options=num_cols)
    
    # Features can be numeric or categorical
    feature_options = [c for c in df_filtered.columns if c != target_r]
    default_features = [c for c in feature_options if pd.api.types.is_numeric_dtype(df_filtered[c])][:5] # Default to first 5 numeric
    
    features_r = st.multiselect("Features (numeric or encoded)", options=feature_options, default=default_features)
    
    if st.button("Run regression"):
        if not target_r:
            st.error("Please select a target column.")
        elif not features_r:
            st.error("Please select at least one feature.")
        else:
            sub = df_filtered[features_r + [target_r]].dropna()
            
            if len(sub) < 10:
                st.error("Not enough data to run regression after dropping NaNs. Please adjust filters.")
            else:
                X = sub[features_r].copy()
                y = sub[target_r].astype(float).copy()
                
                # Preprocessing features
                for col in X.columns:
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                    elif pd.api.types.is_numeric_dtype(X[col]):
                        # Fill NaNs with the mean for numeric columns
                        X[col] = X[col].fillna(X[col].mean())
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
                reg = RandomForestRegressor(n_estimators=100, random_state=42)
                reg.fit(X_train, y_train)
                preds = reg.predict(X_test)
                
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                
                st.write(f"**Mean Squared Error (MSE):** {mse:.3f}")
                st.write(f"**R² Score:** {r2:.3f}")
                
                res_df = pd.DataFrame({'actual': y_test, 'predicted': preds})
                
                # Create scatter plot for actual vs. predicted
                scatter_chart = alt.Chart(res_df).mark_circle(opacity=0.5).encode(
                    x=alt.X('actual', title='Actual Values'),
                    y=alt.Y('predicted', title='Predicted Values'),
                    tooltip=['actual', 'predicted']
                ).interactive()
                
                # Add a perfect prediction line (y=x)
                line_data = pd.DataFrame({'x': [y.min(), y.max()], 'y': [y.min(), y.max()]})
                line = alt.Chart(line_data).mark_line(color='red').encode(
                    x=alt.X('x', title='Actual Values'),
                    y=alt.Y('y', title='Predicted Values')
                )
                
                st.altair_chart(scatter_chart + line, use_container_width=True)
