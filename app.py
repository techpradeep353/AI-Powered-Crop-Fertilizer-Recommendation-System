import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

# Load Models
crop_model = joblib.load("Models/Crop_Recommendation_model.pkl")
fertilizer_model = joblib.load("Models/Fertilizer_Recommendation_model.pkl")

# Extract base models and meta-model
crop_base_models = crop_model['base_models']
crop_meta_model = crop_model['meta_model']

fertilizer_base_models = fertilizer_model['base_models1']
fertilizer_meta_model = fertilizer_model['meta_model1']

# Streamlit UI
st.set_page_config(page_title="AgroVision AI", page_icon="🌾", layout="wide" , initial_sidebar_state="expanded")

st.markdown("""
<style>

@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.animated-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    font-family: 'Segoe UI', sans-serif;
    background: linear-gradient(
        90deg,
        #1B5E20,
        #2E7D32,
        #66BB6A,
        #C0CA33,
        #FDD835,
        #66BB6A,
        #2E7D32
    );
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientFlow 5s linear infinite;
    margin-top: 10px;
    margin-bottom: 20px;
}

</style>

<div class="animated-title">
🌾 Intelligent Crop & Fertilizer Advisory System
</div>

""", unsafe_allow_html=True)

st.markdown(""""
<h1 style="text-align:center;color:#2E7D32;">
🌾 AI-Powered Crop & Fertilizer Recommendation System
</h1>

<h4 style="text-align:center;color:#4CAF50;">
Using Machine Learning
</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("""
<style>

@keyframes glow{
0%{text-shadow:0 0 5px #90EE90;}
50%{text-shadow:0 0 20px #00FF7F;}
100%{text-shadow:0 0 5px #90EE90;}
}

.logo{
font-size:34px;
text-align:center;
animation:glow 2s infinite;
}

.title{
font-size:22px;
font-weight:bold;
text-align:center;
color:white;
margin-top:10px;
}

.subtitle{
text-align:center;
font-size:14px;
color:#C8FACC;
}

.line{
height:2px;
background:linear-gradient(to right,#00ff99,#ffffff,#00ff99);
margin:15px 0;
border-radius:10px;
}

</style>

<div class="logo">🌾</div>

<div class="title">
AgroVision AI
</div>

<div class="subtitle">
Machine Learning Dashboard
</div>

<div class="line"></div>

""", unsafe_allow_html=True)
st.sidebar.title("🌿 AgroVision AI")
st.sidebar.success("Smart Agriculture Dashboard")
st.sidebar.info("🤖 Machine Learning")
st.sidebar.warning("🌱 Smart Farming")
page = st.sidebar.radio("Go to", ["Crop Recommendation", "Fertilizer Recommendation"])

# Crop Recommendation Section
if page == "Crop Recommendation":
    st.header("🌾 Crop Recommendation")
    st.markdown("Enter the details below to get the best crop recommendation.")

    with st.expander("🔍 Input Parameters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            N = st.slider('Nitrogen (N)', 0, 300, 50, 1)
            P = st.slider('Phosphorous (P)', 0, 300, 50, 1)

        with col2:
            K = st.slider('Potassium (K)', 0, 300, 50, 1)
            temperature = st.slider('Temperature (°C)', 10, 50, 25, 1)

        with col3:
            humidity = st.slider('Humidity (%)', 0, 100, 50, 1)
            ph = st.slider('pH Level', 4.0, 9.0, 6.5, 0.1)
            rainfall = st.slider('Rainfall (mm)', 0, 1000, 500, 1)


    if st.button("🌱 Get Crop Recommendation"):
        with st.spinner("Predicting best crop..."):
            new_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                                    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

            # Get base model predictions
            base_preds = np.hstack([model.predict_proba(new_data) for model in crop_base_models.values()])
            final_prediction = crop_meta_model.predict(base_preds)

            st.success(f"✅ Recommended Crop: **{final_prediction[0]}**")

            # Feature Importance Visualization
            feature_importance = crop_base_models['RandomForestClassifier'].feature_importances_
            fig = px.bar(x=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'],
                         y=feature_importance, labels={'x': 'Feature', 'y': 'Importance'}, title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

            # Feature Distributions
            fig = go.Figure([go.Bar(x=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], y=new_data.values[0])])
            fig.update_layout(title="Feature Distribution", xaxis_title="Feature", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🌾 Prediction Probabilities for Crops")

            # Get class probabilities from meta-model
            crop_probabilities = crop_meta_model.predict_proba(base_preds)[0]

            # Get class names (crop labels)
            crop_classes = crop_meta_model.classes_

            # Create DataFrame for Plotly
            crop_prob_df = pd.DataFrame({'Crop': crop_classes, 'Probability': crop_probabilities})
            crop_prob_df = crop_prob_df.sort_values(by="Probability", ascending=False)

            # Plot using Plotly
            fig = px.bar(crop_prob_df, x="Probability", y="Crop", orientation='h', 
                        title="Prediction Probabilities for Different Crops",
                        labels={"Probability": "Prediction Probability", "Crop": "Crop"},
                        color="Probability",
                        color_continuous_scale="greens")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort by probability
            st.plotly_chart(fig, use_container_width=True)

            data = {
                    "Nitrogen (N)": [N],
                    "Phosphorous (P)": [P],
                    "Potassium (K)": [K],
                    "Temperature (°C)": [temperature],
                    "Humidity (%)": [humidity],
                    "pH Level": [ph],
                    "Rainfall (mm)": [rainfall],
                    "Predicted Crop": [final_prediction[0]]  # Store model prediction
                }

            # Convert data to DataFrame
            df = pd.DataFrame(data)

            # Convert DataFrame to CSV format
            csv = df.to_csv(index=False).encode('utf-8')

            # Add Download Button
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{page.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )


# Fertilizer Recommendation Section
elif page == "Fertilizer Recommendation":
    st.header("💊 Fertilizer Recommendation")
    st.markdown("Enter soil and crop details to get the best fertilizer recommendation.")

    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    
    soil_encoder = LabelEncoder().fit(soil_types)
    crop_encoder = LabelEncoder().fit(crop_types)

    with st.expander("🔍 Input Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            temp_fert = st.slider('Temperature (°C)', 0, 50, 25, 1)
            hum_fert = st.slider('Humidity (%)', 0, 100, 50, 1)
            moisture = st.slider('Moisture (%)', 0, 100, 50, 1)

        with col2:
            soil_type = st.selectbox('Soil Type', soil_types)
            crop_type = st.selectbox('Crop Type', crop_types)
            nitrogen = st.slider('Nitrogen (N)', 0, 300, 50, 1)
            potassium = st.slider('Potassium (K)', 0, 300, 50, 1)
            phosphorous = st.slider('Phosphorous (P)', 0, 300, 50, 1)

    if st.button("💊 Get Fertilizer Recommendation"):
        with st.spinner("Predicting best fertilizer..."):
            new_data_fert = pd.DataFrame([[temp_fert, hum_fert, moisture, soil_encoder.transform([soil_type])[0],
                                           crop_encoder.transform([crop_type])[0], nitrogen, potassium, phosphorous]],
                                         columns=['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type',
                                                  'Nitrogen', 'Potassium', 'Phosphorous'])

            # Get base model predictions
            base_preds_fert = np.hstack([model.predict_proba(new_data_fert) for model in fertilizer_base_models.values()])
            final_prediction_fert = fertilizer_meta_model.predict(base_preds_fert)

            st.success(f"✅ Recommended Fertilizer: **{final_prediction_fert[0]}**")

            # Feature Importance Visualization
            feature_importance_fert = fertilizer_base_models['RandomForestClassifier'].feature_importances_
            fig = px.bar(x=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'],
                         y=feature_importance_fert, labels={'x': 'Feature', 'y': 'Importance'}, title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

            # Feature Distributions
            fig = go.Figure([go.Bar(x=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'],
                                    y=new_data_fert.values[0])])
            fig.update_layout(title="Feature Distribution", xaxis_title="Feature", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💊 Prediction Probabilities for Fertilizers")

            # Get class probabilities from meta-model
            fert_probabilities = fertilizer_meta_model.predict_proba(base_preds_fert)[0]

            # Get class names (fertilizers)
            fert_classes = fertilizer_meta_model.classes_

            # Create DataFrame for Plotly
            fert_prob_df = pd.DataFrame({'Fertilizer': fert_classes, 'Probability': fert_probabilities})
            fert_prob_df = fert_prob_df.sort_values(by="Probability", ascending=False)

            # Plot using Plotly
            fig = px.bar(fert_prob_df, x="Probability", y="Fertilizer", orientation='h', 
                        title="Prediction Probabilities for Different Fertilizers",
                        labels={"Probability": "Prediction Probability", "Fertilizer": "Fertilizer"},
                        color="Probability",
                        color_continuous_scale="reds")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort by probability
            st.plotly_chart(fig, use_container_width=True)

            data = {
                    "Temperature (°C)": [temp_fert],
                    "Humidity (%)": [hum_fert],
                    "Moisture (%)": [moisture],
                    "Soil Type": [soil_type],
                    "Crop Type": [crop_type],
                    "Nitrogen (N)": [nitrogen],
                    "Potassium (K)": [potassium],
                    "Phosphorous (P)": [phosphorous],
                    "Predicted Fertilizer": [final_prediction_fert[0]]  # Store model prediction
                }

            # Convert data to DataFrame
            df = pd.DataFrame(data)

            # Convert DataFrame to CSV format
            csv = df.to_csv(index=False).encode('utf-8')

            # Add Download Button
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{page.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )

# Fertilizer Recommendation Section
elif page == "Fertilizer Recommendation":
    st.header("💊 Fertilizer Recommendation")
    st.markdown("Enter soil and crop details to get the best fertilizer recommendation.")

    soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
    crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
    
    soil_encoder = LabelEncoder().fit(soil_types)
    crop_encoder = LabelEncoder().fit(crop_types)

    with st.expander("🔍 Input Parameters", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            temp_fert = st.slider('Temperature (°C)', 0, 50, 25, 1)
            hum_fert = st.slider('Humidity (%)', 0, 100, 50, 1)
            moisture = st.slider('Moisture (%)', 0, 100, 50, 1)

        with col2:
            soil_type = st.selectbox('Soil Type', soil_types)
            crop_type = st.selectbox('Crop Type', crop_types)
            nitrogen = st.slider('Nitrogen (N)', 0, 300, 50, 1)
            potassium = st.slider('Potassium (K)', 0, 300, 50, 1)
            phosphorous = st.slider('Phosphorous (P)', 0, 300, 50, 1)

    if st.button("💊 Get Fertilizer Recommendation"):
        with st.spinner("Predicting best fertilizer..."):
            new_data_fert = pd.DataFrame([[temp_fert, hum_fert, moisture, soil_encoder.transform([soil_type])[0],
                                           crop_encoder.transform([crop_type])[0], nitrogen, potassium, phosphorous]],
                                         columns=['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type',
                                                  'Nitrogen', 'Potassium', 'Phosphorous'])

            # Get base model predictions
            base_preds_fert = np.hstack([model.predict_proba(new_data_fert) for model in fertilizer_base_models.values()])
            final_prediction_fert = fertilizer_meta_model.predict(base_preds_fert)

            st.success(f"✅ Recommended Fertilizer: **{final_prediction_fert[0]}**")

            # Feature Importance Visualization
            feature_importance_fert = fertilizer_base_models['RandomForestClassifier'].feature_importances_
            fig = px.bar(x=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'],
                         y=feature_importance_fert, labels={'x': 'Feature', 'y': 'Importance'}, title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

            # Feature Distributions
            fig = go.Figure([go.Bar(x=['Temperature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous'],
                                    y=new_data_fert.values[0])])
            fig.update_layout(title="Feature Distribution", xaxis_title="Feature", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("💊 Prediction Probabilities for Fertilizers")

            # Get class probabilities from meta-model
            fert_probabilities = fertilizer_meta_model.predict_proba(base_preds_fert)[0]

            # Get class names (fertilizers)
            fert_classes = fertilizer_meta_model.classes_

            # Create DataFrame for Plotly
            fert_prob_df = pd.DataFrame({'Fertilizer': fert_classes, 'Probability': fert_probabilities})
            fert_prob_df = fert_prob_df.sort_values(by="Probability", ascending=False)

            # Plot using Plotly
            fig = px.bar(fert_prob_df, x="Probability", y="Fertilizer", orientation='h', 
                        title="Prediction Probabilities for Different Fertilizers",
                        labels={"Probability": "Prediction Probability", "Fertilizer": "Fertilizer"},
                        color="Probability",
                        color_continuous_scale="reds")

            fig.update_layout(yaxis={'categoryorder': 'total ascending'})  # Sort by probability
            st.plotly_chart(fig, use_container_width=True)

            data = {
                    "Temperature (°C)": [temp_fert],
                    "Humidity (%)": [hum_fert],
                    "Moisture (%)": [moisture],
                    "Soil Type": [soil_type],
                    "Crop Type": [crop_type],
                    "Nitrogen (N)": [nitrogen],
                    "Potassium (K)": [potassium],
                    "Phosphorous (P)": [phosphorous],
                    "Predicted Fertilizer": [final_prediction_fert[0]]  # Store model prediction
                }

            # Convert data to DataFrame
            df = pd.DataFrame(data)

            # Convert DataFrame to CSV format
            csv = df.to_csv(index=False).encode('utf-8')

            # Add Download Button
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"{page.replace(' ', '_').lower()}.csv",
                mime="text/csv"
            )