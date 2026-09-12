import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Hotel Cheapest Rate Predictor",
    page_icon="🏨",
    layout="centered"
)

st.title("🏨 Hotel Cheapest Rate Predictor")

st.write(
    "Predict the historical cheapest hotel rate using "
    "hotel, date, occupancy, and competitor pricing."
)


# --------------------------------------------------
# Model Loading
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    # Get the directory where app.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "model.pkl")
    encoder_path = os.path.join(base_dir, "hotel_encoder.pkl")

    # Check whether files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"model.pkl not found at: {model_path}"
        )

    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"hotel_encoder.pkl not found at: {encoder_path}"
        )

    # Load model and encoder
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    return model, encoder


# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:

    model, le = load_artifacts()

    st.success("✅ Model loaded successfully")

except Exception as e:

    st.error("❌ Error while loading the model")

    # Show the REAL error
    st.exception(e)

    st.stop()


# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------

st.sidebar.header("🏨 Hotel & Date")

hotel_id = st.sidebar.selectbox(
    "Hotel ID",
    [str(x) for x in le.classes_]
)

check_in = st.sidebar.date_input(
    "Check-in Date"
)


# --------------------------------------------------
# Occupancy
# --------------------------------------------------

st.sidebar.header("📊 Occupancy")

occupancy = st.sidebar.slider(
    "Occupancy (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)


# --------------------------------------------------
# Competitor Pricing
# --------------------------------------------------

st.sidebar.header("💰 Competitor Pricing")

competitor_min = st.sidebar.number_input(
    "Competitor Minimum Rate",
    min_value=0.0,
    value=2000.0,
    step=100.0
)

competitor_avg = st.sidebar.number_input(
    "Competitor Average Rate",
    min_value=0.0,
    value=5000.0,
    step=100.0
)

competitor_median = st.sidebar.number_input(
    "Competitor Median Rate",
    min_value=0.0,
    value=4000.0,
    step=100.0
)

competitor_max = st.sidebar.number_input(
    "Competitor Maximum Rate",
    min_value=0.0,
    value=8500.0,
    step=100.0
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "💰 Predict Cheapest Rate",
    use_container_width=True
):

    try:

        # Encode hotel ID using the same encoder
        # used during model training
        hotel_encoded = le.transform([hotel_id])[0]

        # Create input dataframe
        input_data = pd.DataFrame({

            "hotel_id": [hotel_encoded],

            "day_of_month": [
                check_in.day
            ],

            "day_of_week": [
                check_in.weekday()
            ],

            "occupancy": [
                occupancy
            ],

            "competitor_min": [
                competitor_min
            ],

            "competitor_avg": [
                competitor_avg
            ],

            "competitor_median": [
                competitor_median
            ],

            "competitor_max": [
                competitor_max
            ]
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Prevent negative rate
        prediction = max(0, prediction)

        # --------------------------------------------------
        # Display Prediction
        # --------------------------------------------------

        st.success(
            f"### Predicted Cheapest Rate: ₹{prediction:,.2f}"
        )

        # --------------------------------------------------
        # Input Summary
        # --------------------------------------------------

        st.subheader("📋 Input Summary")

        summary = pd.DataFrame({

            "Parameter": [

                "Hotel ID",
                "Check-in Date",
                "Day of Month",
                "Day of Week",
                "Occupancy",
                "Competitor Minimum",
                "Competitor Average",
                "Competitor Median",
                "Competitor Maximum"

            ],

            "Value": [

                hotel_id,

                str(check_in),

                check_in.day,

                check_in.weekday(),

                f"{occupancy:.0f}%",

                f"₹{competitor_min:,.2f}",

                f"₹{competitor_avg:,.2f}",

                f"₹{competitor_median:,.2f}",

                f"₹{competitor_max:,.2f}"

            ]
        })

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

    except Exception as e:

        st.error("❌ Prediction failed")

        st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Model: Gradient Boosting Regressor | "
    "Target: Historical Cheapest Hotel Rate"
)
