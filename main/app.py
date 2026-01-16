import streamlit as st
import joblib
import numpy as np
import os

# Get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models directory
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Available models
model_files = {
    "XGBClassifier": "XGBClassifier_model.pkl",
    "RandomForestClassifier": "RandomForestClassifier_model.pkl",
    "LogisticRegression": "LogisticRegression_model.pkl",
    "SVC": "SVC_model.pkl",
    "DecisionTreeClassifier": "DecisionTreeClassifier_model.pkl",
    "GaussianNB": "GaussianNB_model.pkl",
    "KNeighborsClassifier": "KNeighborsClassifier_model.pkl"
}

st.set_page_config(
    page_title="DonDew Rainfall Prediction System",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ DonDew Rainfall Prediction System")
st.subheader("Machine Learning Based Rainfall Detection")
st.write("Enter the weather feature values below to predict rainfall.")


st.sidebar.title("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(model_files.keys())
)

# Load selected model
model_path = os.path.join(MODEL_DIR, model_files[model_name])
model = joblib.load(model_path)

st.sidebar.success(f"Loaded Model: {model_name}")

with st.form("rainfall_form"):
    st.subheader("Weather Input Parameters")

    temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
    pressure = st.number_input("Pressure (hPa)", 900.0, 1100.0, 1013.0)
    wind_speed = st.number_input("Wind Speed (km/h)", 0.0, 150.0, 10.0)
    cloud_cover = st.number_input("Cloud Cover (%)", 0.0, 100.0, 50.0)
    rainfall_last = st.number_input("Previous Rainfall (mm)", 0.0, 500.0, 0.0)
    dew_point = st.number_input("Dew Point (°C)", -10.0, 40.0, 20.0)
    visibility = st.number_input("Visibility (km)", 0.0, 20.0, 10.0)

    submit = st.form_submit_button("Predict Rainfall")

if submit:
    input_data = np.array([[
        temperature,
        humidity,
        pressure,
        wind_speed,
        cloud_cover,
        rainfall_last,
        dew_point,
        visibility
    ]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success(f"🌧️ Rainfall Expected")
        st.write(f"Probability of Rainfall: **{probability:.2f}**")
    else:
        st.info(f"☀️ No Rainfall Expected")
        st.write(f"Probability of Rainfall: **{probability:.2f}**")


st.markdown("---")
st.caption("Rainfall Prediction System | Machine Learning Project")
