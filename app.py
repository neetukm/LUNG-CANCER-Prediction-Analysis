# advanced_streamlit_app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Set page configuration
st.set_page_config(page_title="Lung Cancer Predictor", page_icon="🫁", layout="wide")

# Load trained model
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Add logo
with st.sidebar:
    st.image("lung cancer image.jpg", width=150)

# Title and Description
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>🫁 Lung Cancer Prediction Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict lung cancer risk based on patient's health indicators and explore visualization.</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar for patient details
st.sidebar.header("📝 Patient Information")

# Sidebar Inputs
gender = st.sidebar.radio('Gender', ['Female (0)', 'Male (1)'])
age = st.sidebar.slider('Age', 1, 120, 35)
smoking = st.sidebar.radio('Smoking', ['Yes (1)', 'No (2)'])
yellow_fingers = st.sidebar.radio('Yellow Fingers', ['Yes (1)', 'No (2)'])
anxiety = st.sidebar.radio('Anxiety', ['Yes (1)', 'No (2)'])
peer_pressure = st.sidebar.radio('Peer Pressure', ['Yes (1)', 'No (2)'])
chronic_disease = st.sidebar.radio('Chronic Disease', ['Yes (1)', 'No (2)'])
fatigue = st.sidebar.radio('Fatigue', ['Yes (1)', 'No (2)'])
allergy = st.sidebar.radio('Allergy', ['Yes (1)', 'No (2)'])
wheezing = st.sidebar.radio('Wheezing', ['Yes (1)', 'No (2)'])
alcohol_consuming = st.sidebar.radio('Alcohol Consuming', ['Yes (1)', 'No (2)'])
coughing = st.sidebar.radio('Coughing', ['Yes (1)', 'No (2)'])
shortness_of_breath = st.sidebar.radio('Shortness of Breath', ['Yes (1)', 'No (2)'])
swallowing_difficulty = st.sidebar.radio('Swallowing Difficulty', ['Yes (1)', 'No (2)'])
chest_pain = st.sidebar.radio('Chest Pain', ['Yes (1)', 'No (2)'])

# Predict Button
if st.sidebar.button('🔍 Predict'):
    # Prepare the input
    input_features = np.array([[int(gender[-2]), age, int(smoking[-2]), int(yellow_fingers[-2]), int(anxiety[-2]),
                                int(peer_pressure[-2]), int(chronic_disease[-2]), int(fatigue[-2]),
                                int(allergy[-2]), int(wheezing[-2]), int(alcohol_consuming[-2]),
                                int(coughing[-2]), int(shortness_of_breath[-2]), int(swallowing_difficulty[-2]),
                                int(chest_pain[-2])]])

    prediction = model.predict(input_features)

    # Main page results
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ **High Risk** of Lung Cancer Detected!")
        st.progress(90)
    else:
        st.success("✅ **Low Risk** - No Lung Cancer Detected!")
        st.progress(30)

    # Patient Summary
    st.write("---")
    st.subheader("🧾 Patient Details Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Gender", value="Male" if int(gender[-2]) == 1 else "Female")
        st.metric(label="Age", value=f"{age} years")
        st.metric(label="Smoker", value="Yes" if int(smoking[-2]) == 1 else "No")
        st.metric(label="Yellow Fingers", value="Yes" if int(yellow_fingers[-2]) == 1 else "No")
        st.metric(label="Chronic Disease", value="Yes" if int(chronic_disease[-2]) == 1 else "No")
    
    with col2:
        st.metric(label="Coughing", value="Yes" if int(coughing[-2]) == 1 else "No")
        st.metric(label="Shortness of Breath", value="Yes" if int(shortness_of_breath[-2]) == 1 else "No")
        st.metric(label="Chest Pain", value="Yes" if int(chest_pain[-2]) == 1 else "No")
        st.metric(label="Fatigue", value="Yes" if int(fatigue[-2]) == 1 else "No")
        st.metric(label="Alcohol Consumption", value="Yes" if int(alcohol_consuming[-2]) == 1 else "No")

    st.write("---")

    # Visualizations Section
    st.subheader("📈 Visual Analysis Based on Input Data")
    
    # Create  patient dataframe
    patient_data = {
        'Features': ['Gender', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 'Peer Pressure',
                     'Chronic Disease', 'Fatigue', 'Allergy', 'Wheezing', 'Alcohol Consuming',
                     'Coughing', 'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain'],
        'Values': [int(gender[-2]), age, int(smoking[-2]), int(yellow_fingers[-2]), int(anxiety[-2]),
                   int(peer_pressure[-2]), int(chronic_disease[-2]), int(fatigue[-2]),
                   int(allergy[-2]), int(wheezing[-2]), int(alcohol_consuming[-2]),
                   int(coughing[-2]), int(shortness_of_breath[-2]), int(swallowing_difficulty[-2]),
                   int(chest_pain[-2])]
    }
    df = pd.DataFrame(patient_data)


    
# # --- Create Tabs for Different Visualizations ---

    tab1, tab2, tab3 = st.tabs(["📊 Histogram", "📊 Bar Plot", "🥧 Pie Chart"])

    with tab1:
        st.subheader("Histogram (Patient's Feature Values)")
        fig, ax = plt.subplots()
        # Here: Plot all numeric Values
        sns.histplot(patient_data['Values'], kde=True, bins=10, ax=ax, color="skyblue")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Patient Detail")
        st.pyplot(fig)
    
    with tab2:
        st.subheader("Bar Plot ")
        fig2, ax2 = plt.subplots(figsize=(8,6))
        sns.barplot(x='Values', y='Features', data=patient_data, palette="viridis", ax=ax2)
        ax2.set_xlabel("Feature Value")
        ax2.set_ylabel("Feature")
        ax2.set_title("Patient's Feature Contributions")
        st.pyplot(fig2)
    
    # --- Pie Chart (Advanced) ---
   # Smoking Feature Pie Chart
    with tab3:
        st.subheader("🚬 Smoking Status Pie Chart")

        # Create a small dataframe for Smoking
        smoking_data = pd.DataFrame({
            'Smoking Status': ['Smoker', 'Non-Smoker'],
            'Count': [1 if int(smoking[-2]) == 1 else 0, 1 if int(smoking[-2]) == 2 else 0]
        })
        
        # Pie Chart
        fig_smoke, ax_smoke = plt.subplots()
        ax_smoke.pie(smoking_data['Count'], labels=smoking_data['Smoking Status'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
        ax_smoke.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        st.pyplot(fig_smoke)




    # Expandable Section
    with st.expander("ℹ️ Model Explanation"):
        st.write("""
        - The model is trained to predict the likelihood of lung cancer based on patient health factors.
        - Features like smoking, chronic diseases, fatigue, and coughing greatly influence predictions.
        - Visualizations help in understanding which features are contributing more towards the prediction.
        - Always consult a medical professional for final diagnosis.
        """)

    st.write("---")
    st.markdown("<h6 style='text-align: center; color: gray;'>Stay healthy and take care! 🫶</h6>", unsafe_allow_html=True)
