# app.py
# ======================================
# HEART DISEASE RISK PREDICTION + UI
# ======================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ==========================
# 1. LOAD & PREPARE DATA
# ==========================

@st.cache_data
def load_data(csv_path: str = "heart.csv"):
    df = pd.read_csv(csv_path)
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """
    Train a RandomForest model on the heart dataset.
    Returns: model, scaler, feature_columns, metrics_dict
    """

    # Separate features & target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Metrics
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "accuracy": acc,
        "roc_auc": auc,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
    }

    return model, scaler, X.columns.tolist(), metrics


# ==========================
# 2. PREDICTION FUNCTION
# ==========================

def predict_risk(model, scaler, feature_cols, user_input_dict):
    """
    user_input_dict: dictionary of feature -> value
    Returns predicted_class (0/1), prob (0-1)
    """

    # Ensure correct order of columns
    x = np.array([[user_input_dict[col] for col in feature_cols]])
    x_scaled = scaler.transform(x)

    prob = model.predict_proba(x_scaled)[0][1]  # probability of class 1 (disease)
    pred = int(model.predict(x_scaled)[0])

    return pred, prob


# ==========================
# 3. STREAMLIT UI
# ==========================

def main():
    st.set_page_config(
        page_title="Heart Disease Risk Predictor",
        page_icon="❤️",
        layout="centered"
    )

    st.title("❤️ Heart Disease Risk Prediction")
    st.write("""
    This app uses a machine learning model to estimate the **risk of heart disease**
    based on clinical parameters.  
    **Note:** This is a *study project*, not a medical diagnosis. Always consult a doctor.
    """)

    # Load data and train model
    with st.spinner("Loading data and training model..."):
        df = load_data("heart.csv")
        model, scaler, feature_cols, metrics = train_model(df)

    # Show dataset info
    with st.expander("📊 Dataset & Model Info"):
        st.write("**Rows, Columns:**", df.shape)
        st.write("**Sample Data (first 5 rows):**")
        st.dataframe(df.head())
        st.write("**Model Performance (Test Set):**")
        st.write(f"- Accuracy: `{metrics['accuracy']:.3f}`")
        st.write(f"- ROC-AUC: `{metrics['roc_auc']:.3f}`")
        st.write(f"- Train Size: `{metrics['train_size']}`")
        st.write(f"- Test Size: `{metrics['test_size']}`")

    st.markdown("---")
    st.header("🧍 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)
        sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0], index=0)
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            [0, 1, 2, 3],
            index=0,
            help="0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic"
        )
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", 80, 200, 130, step=1)
        chol = st.number_input("Cholesterol (chol)", 100, 600, 240, step=1)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1], index=0)

    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2], index=0)
        thalach = st.number_input("Maximum Heart Rate (thalach)", 60, 220, 150, step=1)
        exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1], index=0)
        oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment (slope)", [0, 1, 2], index=2)
        ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4], index=0)
        thal = st.selectbox(
            "Thal (3 = normal, 6 = fixed defect, 7 = reversible defect)",
            [3, 6, 7],
            index=0
        )

    # Map UI inputs to model feature order
    user_input = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    st.markdown("")

    if st.button("🔍 Predict Risk"):
        pred, prob = predict_risk(model, scaler, feature_cols, user_input)

        st.subheader("📈 Prediction Result")

        risk_percent = prob * 100.0

        if pred == 1:
            st.error(f"High Risk of Heart Disease: **{risk_percent:.1f}%**")
        else:
            st.success(f"Low Risk of Heart Disease: **{risk_percent:.1f}%**")

        st.markdown("### 🧾 Interpretation (Simple)")
        if risk_percent < 30:
            st.write("- Model suggests **low** estimated risk.")
            st.write("- Maintain a healthy lifestyle and regular checkups.")
        elif risk_percent < 60:
            st.write("- Model suggests **moderate** risk.")
            st.write("- Consider consulting a doctor and monitoring regularly.")
        else:
            st.write("- Model suggests **high** risk.")
            st.write("- Strongly recommend professional medical consultation.")

        st.caption(
            "⚠ This is a machine learning estimate based on historical data. "
            "It is **not** a medical diagnosis."
        )


if __name__ == "__main__":
    main()
