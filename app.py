import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load saved model and encoder

model = joblib.load("model.pkl")
hotel_encoder = joblib.load("hotel_encoder.pkl")

st.set_page_config(
page_title="Hotel Price Prediction",
page_icon="🏨",
layout="centered"
)

st.title("🏨 Hotel Cheapest Rate Predictor")
st.write(
"Predict a hotel's expected cheapest rate using "
"check-in date, occupancy, and competitor prices."
)

# Hotel selection

hotel_ids = list(hotel_encoder.classes_)

hotel_id = st.selectbox(
"Select Hotel ID",
hotel_ids
)

# Check-in date

check_in = st.date_input(
"Check-in Date",
value=date.today()
)

# Occupancy percentage

occupancy = st.slider(
"Occupancy (%)",
min_value=0,
max_value=100,
value=50
)

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

if st.button("Predict Cheapest Rate", type="primary"):

```
# Encode hotel ID using the saved encoder
encoded_hotel_id = hotel_encoder.transform([hotel_id])[0]

# Keep feature names consistent with model training
input_data = pd.DataFrame([{
    "hotel_id": encoded_hotel_id,
    "day_of_month": check_in.day,
    "day_of_week": check_in.weekday(),
    "occupancy": occupancy,
    "competitor_min": competitor_min,
    "competitor_avg": competitor_avg,
    "competitor_median": competitor_median,
    "competitor_max": competitor_max
}])

prediction = model.predict(input_data)[0]

st.success(f"Predicted Cheapest Rate: ₹{max(0, prediction):,.2f}")

st.caption(
    "This is an estimated rate from the trained model, "
    "not a guaranteed live booking price."
)
```
