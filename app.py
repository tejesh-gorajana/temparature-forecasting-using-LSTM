
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow import keras

MODEL_PATH = "temperature_lstm.keras"
SCALER_PATH = "temperature_scaler.pkl"
SEQUENCE_LENGTH = 24

st.set_page_config(
    page_title="Temperature Forecasting",
    page_icon="🌡️"
)

st.title("🌡️ Temperature Forecasting Using LSTM")
st.write(
    "Enter the most recent 24 temperature values to predict "
    "the next temperature."
)

@st.cache_resource
def load_model_and_scaler():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_model_and_scaler()

input_text = st.text_area(
    "Enter 24 temperatures in °C, separated by commas:",
    value=", ".join(["15.0"] * SEQUENCE_LENGTH)
)

if st.button("Predict Temperature"):

    try:
        temperatures = np.array(
            [float(x.strip()) for x in input_text.split(",")],
            dtype=np.float32
        )

        if len(temperatures) != SEQUENCE_LENGTH:
            st.error(
                f"Please enter exactly {SEQUENCE_LENGTH} temperature values."
            )

        else:
            # Scale the input in the same way as training data
            scaled_input = scaler.transform(
                temperatures.reshape(-1, 1)
            )

            # Create LSTM input shape
            X_new = scaled_input.reshape(
                1, SEQUENCE_LENGTH, 1
            )

            # Predict
            prediction_scaled = model.predict(
                X_new,
                verbose=0
            )

            # Convert back to Celsius
            prediction = scaler.inverse_transform(
                prediction_scaled
            )[0, 0]

            st.success(
                f"Predicted Next Temperature: {prediction:.2f} °C"
            )

            chart_data = pd.DataFrame({
                "Temperature (°C)": temperatures
            })

            st.line_chart(chart_data)

    except ValueError:
        st.error(
            "Please enter only valid numeric temperature values."
        )
