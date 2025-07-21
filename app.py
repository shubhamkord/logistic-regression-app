
import streamlit as st
import pickle
import numpy as np

# Load model
with open('logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.number_input("Age", 0, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
fare = st.number_input("Fare", 0.0, 500.0, 50.0)

# Encode
sex_encoded = 1 if sex == 'male' else 0
features = np.array([[pclass, age, sex_encoded, fare]])

if st.button("Predict"):
    result = model.predict(features)
    if result[0] == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")
