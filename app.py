import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load trained model and encoder
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoder = joblib.load("hotel_encoder.pkl")
    return model, encoder

model, hotel_encoder = load_model()

# Page configuration
st.set_page_config(
    page_title="Hotel Price Prediction",
    page_icon="🏨",
    layout="centered"
)

st.title("Hotel Cheapest Rate Predictor")
st.write("Predict the expected cheapest hotel rate using occupancy and competitor prices.")

# Hotel selection
hotel_ids = list(hotel_encoder.classes_)

hotel_id = st.selectbox("Select Hotel ID", hotel_ids)

# Check-in date
check_in = st.date_input(
    "Check-in Date",
    value=date.today()
)

# Calculate lead time
raw_lead_time = (check_in - date.today()).days
lead_time = max(1, min(31, raw_lead_time))

st.write(f"Lead time: {lead_time} day(s)")

# Occupancy
occupancy = st.slider(
    "Occupancy (%)",
    min_value=0,
    max_value=100,
    value=50
)

# Competitor prices
st.subheader("Competitor Rates (₹)")

competitor_min = st.number_input(
    "Minimum Competitor Rate",
    min_value=0.0,
    value=3000.0,
    step=100.0
)

competitor_avg = st.number_input(
    "Average Competitor Rate",
    min_value=0.0,
    value=4000.0,
    step=100.0
)

competitor_median = st.number_input(
    "Median Competitor Rate",
    min_value=0.0,
    value=4000.0,
    step=100.0
)

competitor_max = st.number_input(
    "Maximum Competitor Rate",
    min_value=0.0,
    value=6000.0,
    step=100.0
)

# Prediction
if st.button("Predict Cheapest Rate"):

    if not (
        competitor_min <= competitor_avg <= competitor_max
        and competitor_min <= competitor_median <= competitor_max
    ):
        st.error("Please enter valid competitor rates.")

    else:
        encoded_hotel_id = hotel_encoder.transform([hotel_id])[0]

        input_data = pd.DataFrame([{
            "hotel_id": encoded_hotel_id,
            "lead_time": lead_time,
            "day_of_week": check_in.weekday(),
            "occupancy": occupancy,
            "competitor_min": competitor_min,
            "competitor_avg": competitor_avg,
            "competitor_median": competitor_median,
            "competitor_max": competitor_max
        }])

        # Match the exact feature order used during training
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[list(model.feature_names_in_)]

        prediction = float(model.predict(input_data)[0])

        st.success(
            f"Predicted Cheapest Rate: ₹{max(0, prediction):,.2f}"
        )

        st.caption(
            "This is an estimated rate from the trained model, "
            "not a live booking price."
        )
