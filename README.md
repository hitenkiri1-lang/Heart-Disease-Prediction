# ❤️ Heart Disease Risk Prediction App

A **machine learning–based web application** built with **Streamlit** that predicts the **risk of heart disease** using clinical patient data.  
This project is intended for **educational purposes only** and is **not a medical diagnosis tool**.

---

## 🚀 Project Overview

This app:
- Trains a **Random Forest Classifier** on a heart disease dataset
- Uses **standardized clinical features**
- Provides an **interactive web UI** for user input
- Outputs:
  - Predicted risk category (High / Low)
  - Probability score (risk percentage)
  - Model performance metrics (Accuracy & ROC-AUC)

---

## 🧠 Machine Learning Details

- **Model:** Random Forest Classifier  
- **Preprocessing:** StandardScaler  
- **Train/Test Split:** 80% / 20% (stratified)  
- **Evaluation Metrics:**
  - Accuracy
  - ROC-AUC score  

The model is trained dynamically when the app runs and cached for performance.

---

## 📊 Dataset

- File: `heart.csv`
- Target column: `target`
  - `0` → No heart disease
  - `1` → Heart disease present
- Features include:
  - Age, Sex
  - Chest pain type
  - Blood pressure, cholesterol
  - ECG results
  - Max heart rate
  - Exercise-induced angina
  - ST depression
  - Number of major vessels
  - Thalassemia type

---

## 🖥️ User Interface

The Streamlit UI allows users to:
- Enter patient details via sliders and dropdowns
- View dataset samples and model performance
- Get a **risk percentage** with an easy-to-understand interpretation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
    ```bash
    git clone https://github.com/your-username/Heart-Disease-Prediction.git

2️⃣ Install dependencies
      
      pip install streamlit pandas numpy scikit-learn

3️⃣ Run the app
      
      streamlit run app.py


The app will open automatically in your browser.

⚠️ Disclaimer

This application is a study project created for learning and demonstration purposes.
It must not be used for real medical diagnosis or treatment decisions.
Always consult a qualified healthcare professional.
