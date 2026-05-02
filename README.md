import streamlit as st
import pandas as pd
import numpy as np
import LabelEncoder() as le

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("Commercial Appeal Predictor")

TARGET = "7. Which advertisement appeals to you the most?"
FEATURES = [
    "1. Where are you from?",
    "2. How old are you?",
    "3. How would you describe your gender identity?",
    "4. What is the highest level of education you have?",
    "5. Which category best describes your total annual household income before taxes in CAD?",
    "6. Do you think it is important for brands to address political/societal issues in commercials?",
    "11. On a scale from 1-5, how impactful is the message of this advertisement on your personal perception of this issue?",
    "12. Are you encouraged to learn more about this issue?",
    "13. Do you think the advertisement effectively achieves its goal of raising awareness to this type of issue?",
    "14. Does this type of commercials change your perception of the brand? If so, how?",
]

uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

if uploaded_file is not None:
    dataset = pd.read_excel(uploaded_file)

    dataset = dataset[FEATURES + [TARGET]].copy()
    dataset[TARGET] = dataset[TARGET].astype(str).str.split("┋").str[0].str.strip()

    X = dataset[FEATURES]
    y = dataset[TARGET]

    st.write("Data loaded:", dataset.shape)

    if st.button("Train model"):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        x_processed = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            x_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"Model trained! Accuracy: {acc:.2f}")
        st.write("Ad classes:", list(le.classes_))

        st.session_state.model = model
        st.session_state.le = le
        st.session_state.columns = x_processed.columns

if "model" in st.session_state:
    st.header("Make Prediction")

    country = st.text_input("Country")
    age = st.selectbox("Age", ["Under 18", "18-24", "25-34", "35-44", "45-54", "55-64"])
    gender = st.text_input("Gender")
    education = st.selectbox(
        "Education Level",
        ["High school", "Apprenticeship", "Associate degree", "Bachelor degree", "Graduate or professional degree (e.g. MA or PhD)"]
    )
    income = st.selectbox(
        "Income",
        ["under 15,000", "between 15,000 and 29,000", "between 30,000 and 49,000", "between 50,000 and 74,000", "between 75,000 and 99,999", "over 100,000"]
    )
    important = st.selectbox("Important for brands?", ["yes", "no"])
    impact = st.selectbox("Impact 1-5", [1, 2, 3, 4, 5])
    learn_more = st.selectbox("Encouraged to learn more?", ["yes", "no"])
    effective = st.selectbox("Ad effective?", ["yes", "no"])
    brand_change = st.text_input("Brand perception change")

    if st.button("Predict"):
        new_data = pd.DataFrame([{
            "1. Where are you from?": country,
            "2. How old are you?": age,
            "3. How would you describe your gender identity?": gender,
            "4. What is the highest level of education you have?": education,
            "5. Which category best describes your total annual household income before taxes in CAD?": income,
            "6. Do you think it is important for brands to address political/societal issues in commercials?": important,
            "11. On a scale from 1-5, how impactful is the message of this advertisement on your personal perception of this issue?": impact,
            "12. Are you encouraged to learn more about this issue?": learn_more,
            "13. Do you think the advertisement effectively achieves its goal of raising awareness to this type of issue?": effective,
            "14. Does this type of commercials change your perception of the brand? If so, how?": brand_change,
        }])

        new_processed = pd.get_dummies(new_data, drop_first=True)
        new_processed = new_processed.reindex(columns=st.session_state.columns, fill_value=0)

        pred = st.session_state.model.predict(new_processed)
        pred_class = st.session_state.le.inverse_transform(pred)[0]
        pred_proba = st.session_state.model.predict_proba(new_processed)[0]

        st.write(f"**Predicted ad appeal:** {pred_class}")
        st.write("**Probabilities:**")
        for cls, p in zip(st.session_state.le.classes_, pred_proba):
            st.write(f"{cls}: {p:.2f}")
