import streamlit as st
import joblib
import re
import pandas as pd
import numpy as np
import warnings
import os
import sklearn

# --- BUILD ROBUST FILE PATHS ---
# This finds the directory the app.py file is in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'DiseaseAndSymptoms.csv')

# --- !!! V2 FILE PATHS !!! ---
# We are renaming the files to force Streamlit to load the new ones
# and bust its cache.
MODEL_PATH = os.path.join(BASE_DIR, 'disease_model_v2.joblib')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'symptom_vectorizer_v2.joblib')
LE_PATH = os.path.join(BASE_DIR, 'label_encoder_v2.joblib')

# Suppress warnings
warnings.filterwarnings('ignore')

# --- THIS IS THE CRITICAL FUNCTION ---
# This function MUST be IDENTICAL to the one in your training_v2.py
def clean_symptom(symptom):
    """Cleans a single symptom string robustly."""
    symptom = str(symptom).strip() # 1. Remove leading/trailing spaces
    symptom = re.sub(r'_\d+$', '', symptom) # 2. Remove suffixes like _1
    symptom = symptom.replace('_', ' ') # 3. Replace underscore with SPACE
    symptom = re.sub(r'\s+', ' ', symptom) # 4. Collapse multiple spaces
    return symptom.strip() # 5. Strip again

# --- LOAD ARTIFACTS ---

@st.cache_data
def load_symptom_list():
    """Loads and cleans the symptom list from the original CSV."""
    if not os.path.exists(CSV_PATH):
        st.error(f"Error: 'DiseaseAndSymptoms.csv' not found at {CSV_PATH}")
        return []
        
    try:
        df = pd.read_csv(CSV_PATH)
        symptom_cols = [col for col in df.columns if 'Symptom_' in col]
        
        # Get all symptoms from all columns and flatten into a single list
        all_symptoms = df[symptom_cols].values.flatten()
        
        unique_symptoms = set()
        for s in all_symptoms:
            # Run EACH symptom through the correct cleaning function
            cleaned = clean_symptom(s) 
            if cleaned: 
                unique_symptoms.add(cleaned)
        
        # The list will now contain "abdominal pain" (with a space)
        return sorted(list(unique_symptoms))
    except Exception as e:
        st.error(f"Error loading DiseaseAndSymptoms.csv: {e}")
        return []

@st.cache_resource
def load_models():
    """Loads the saved model, vectorizer, and label encoder."""
    # Check for the NEW V2 file paths
    for f_path in [MODEL_PATH, VECTORIZER_PATH, LE_PATH]:
        if not os.path.exists(f_path):
            st.error(f"Error: Model file not found at {f_path}")
            st.error("Please ensure disease_model_v2.joblib, symptom_vectorizer_v2.joblib, and label_encoder_v2.joblib are in the GitHub repo.")
            return None, None, None
            
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        le = joblib.load(LE_PATH)
        
        # Log the scikit-learn version the app is running on
        st.sidebar.text(f"Running sklearn: {sklearn.__version__}")
        
        return model, vectorizer, le
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

# Load all assets
symptom_list = load_symptom_list()
model, vectorizer, le = load_models()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Disease Predictor", layout="centered")
st.title("Disease Predictor from Symptoms")
st.markdown("Select your symptoms from the list. The model will predict the top 3 most likely diseases.")

user_symptoms = st.multiselect(
    "Select your symptoms:",
    options=symptom_list # This list is now correctly cleaned
)

if st.button("Predict Disease"):
    if model and vectorizer and le and user_symptoms:
        
        # --- 1. Process User Input ---
        # The symptoms from the list are ALREADY CLEAN
        symptom_string = ' '.join(user_symptoms)

        if not symptom_string:
            st.warning("Please select at least one symptom.")
        else:
            try:
                # --- 2. Vectorize the Input ---
                # This vectorizer was trained on 'abdominal pain' (with space)
                vectorized_input = vectorizer.transform([symptom_string])
                
                # Check if vector is all zeros (meaning no words were found)
                if vectorized_input.sum() == 0:
                    st.error("The selected symptoms are not recognized by the model's vocabulary.")
                else:
                    # --- 3. Make Prediction ---
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(vectorized_input)[0]
                        top_3_indices = np.argsort(probabilities)[::-1][:3]
                        
                        st.subheader("Top 3 Predictions:")
                        
                        for i, idx in enumerate(top_3_indices):
                            disease_name = le.classes_[idx]
                            confidence = probabilities[idx] * 100
                            
                            if confidence > 0.01: # Only show relevant predictions
                                if i == 0:
                                    st.success(f"**1. {disease_name}** (Confidence: {confidence:.2f}%)")
                                elif i == 1:
                                    st.warning(f"**2. {disease_name}** (Confidence: {confidence:.2f}%)")
                                else:
                                    st.info(f"**3. {disease_name}** (Confidence: {confidence:.2f}%)")
                            
                    else:
                        # Fallback for models without predict_proba
                        prediction = model.predict(vectorized_input)
                        disease_name = le.inverse_transform(prediction)
                        st.subheader("Prediction Result:")
                        st.success(f"**The model predicts: {disease_name[0]}**")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                
    elif not user_symptoms:
        st.warning("Please select your symptoms from the list.")

