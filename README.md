---
license: apache-2.0
title: 🌱 Crop & Fertilizer Recommendation System
sdk: streamlit
emoji: 🌍
colorFrom: green
colorTo: blue
pinned: false

short_description: The system provides real-time recommendations for crp & fert
sdk_version: 1.43.2
---
# **🌱 Intelligent Crop & Fertilizer Advisory System**  

## **📌 Introduction**  
This project is a **AI Powered Crop & Fertilizer Recommendation System**, developed as part of my **BrainOvision summer Internship**. The system provides **personalized crop and fertilizer recommendations** based on soil composition, weather conditions, and crop requirements.  

🔗 **Deployed Application:** [Crop & Fertilizer Recommendation System](https://ai-powered-crop-fertilizer-recommendation-system-pax2rg7gdfhcz.streamlit.app/)  

## **🎯 Project Goals**  
✅ Integrate **AI and Green Skills** into agriculture.  
✅ Explore **Machine Learning techniques** for precision farming.  
✅ Develop a **real-time recommendation system** for farmers.  
✅ Enhance **data preprocessing, model selection, and optimization skills**.  
✅ Deploy an **interactive web app** using **Streamlit**.  

## **📂 Datasets Used**  
The project uses two datasets:  

### **1️⃣ Crop Dataset**  
- **Features:** Nitrogen, Phosphorous, Potassium, Temperature, Humidity, pH, Rainfall  
- **Target Variable:** Crop Label (Crop Name)  

### **2️⃣ Fertilizer Dataset**  
- **Features:** Temperature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Phosphorous, Potassium  
- **Target Variable:** Fertilizer Name  

## **📊 Methodology**  

### **Step 1: Data Collection & Preprocessing**  
🔹 **Performed Exploratory Data Analysis (EDA)** to understand data distributions.  
🔹 **Encoded categorical variables** using Label Encoding.  
🔹 **Split the dataset** into **train & test sets** using Scikit-Learn.  

### **Step 2: Model Selection & Training**  
🔹 Tested multiple **ML algorithms**, including:  
   - Logistic Regression, GaussianNB, SVC, KNN, DecisionTree, ExtraTree, RandomForest, Bagging, Gradient Boosting, AdaBoost, CatBoost, and LGBM.

🔹 Compared **model performance** based on accuracy & validation metrics.  

### **Step 3: Ensemble Learning for Optimization**  
🔹 Evaluated different ensemble techniques:  
   - **Voting Classifier, Stacking, Averaging Probabilities, Weighted Ensemble, Blend Ensemble (Custom Blending).**

🔹 **Blend Ensemble provided the best results**, so it was selected.  
🔹 **Cross-validation** was performed to validate model performance.  

### **Step 4: Deployment & Integration**  
🔹 Exported trained models as **.pkl files** (`crop_recommendation.pkl`, `fertilizer_recommendation.pkl`).  
🔹 Integrated models into a **Streamlit app** for real-time predictions.  
🔹 **Deployed the application on Streamlit Cloud.**  

## **🔍 Key Features**  
✅ **Real-time Model Performance Metrics** (Accuracy Tracking).  
✅ **Feature Importance Analysis** for better interpretability.  
✅ **Feature Distributions** to understand data variations.  
✅ **Prediction Probabilities** to assess model confidence.  

## **🚀 Technologies Used**  
| Category            | Tools & Libraries |
|---------------------|-------------------|
| **Development**    | Python, Jupyter Notebook, Anaconda |
| **ML Frameworks**  | Scikit-Learn, CatBoost, LGBM |
| **Data Processing**| Pandas, NumPy |
| **Visualization**  | Matplotlib, Seaborn, Plotly |
| **Deployment**     | HuggingFace, Streamlit, Streamlit Cloud |

## **📷 Screenshots**  

| **Streamlit App - Crop Recommendation** |
|-----------------------------------------|
|![Screenshot_20250211_223522_Chrome-imageonline co-merged](https://github.com/user-attachments/assets/7c3b2339-37c4-49ba-8ee4-a74e5d7575ac)|

| **Streamlit App - Fertilizer Recommendation** |
|-----------------------------------------------|
|![Screenshot_20250211_223923_Chrome-imageonline co-merged](https://github.com/user-attachments/assets/63446c22-de9c-4b9c-a3f3-e576bc7294e5)|

## **🎯 Future Improvements**  
🔹 Expand the dataset to include **more crop varieties & soil types**.  
🔹 Integrate **real-time weather data** for better recommendations.  
🔹 Incorporate **IoT & satellite data** for advanced precision farming.  
🔹 Optimize **model efficiency & deployment** for faster predictions.  

## **📥 Installation & Setup**  

### **🔹 Clone the Repository**  
```bash
git clone https://github.com/Samarth4023/Shell-Internship-1.git
cd Shell-Internship-1
```

### **🔹 Install Required Dependencies**  
```bash
pip install -r requirements.txt
```

### **🔹 Run the Streamlit App**  
```bash
streamlit run app.py
```

## **📜 License**  
This project is **open-source** and free to use. Feel free to contribute!  

## **📧 Contact**  
📌 **Author:** Pradeep kumar pradhan
📌 **GitHub:** [pradeep353](https://github.com/techpradeep353)  
