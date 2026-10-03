import streamlit as st
import pandas as pd
import joblib
from datetime import date

# --------------------------------------------------

# 1. Load Model and Hotel Encoder

# --------------------------------------------------

@st.cache_resource
def load_artifacts():
model = joblib.load("model.pkl")
hotel_encoder = joblib.load("hotel_encoder.pkl")
return model, hotel_encoder

model, hotel_encoder = load_artifacts()

# --------------------------------------------------

# 2. Page Configuration

# --------------------------------------------------

st.set_page_config(
page_title="Hotel Price Prediction",
page_icon="🏨",
layout="centered"
)

st.title("🏨 Hotel Cheapest Rate Predictor")
st.markdown(
"Predict a hotel's expected cheapest rate using "
"check-in date, occupancy, and competitor prices."
)

st.info(
"Enter the check-in date and current market information "
"to estimate the hotel's cheapest rate."
)

# --------------------------------------------------

# 3. Hotel Selection

# --------------------------------------------------

hotel_ids = list(hotel_encoder.classes_)

hotel_id = st.selectbox(
"Select Hotel ID",
hotel_ids
)

# --------------------------------------------------

# 4. Check-in Date and Lead Time

# --------------------------------------------------

check_in = st.date_input(
"Check-in Date",
value=date.today()
)

# Lead time = check-in date - current scraping/prediction date

raw_lead_time = (check_in - date.today()).days

# Match the training notebook's 1–31 day range

lead_time = max(1, min(31, raw_lead_time))

st.caption(f"Lead time used by model: {lead_time} day(s)")

if raw_lead_time < 1:
st.warning(
"The check-in date is today or in the past. "
"The model will use a lead time of 1 day."
)
elif raw_lead_time > 31:
st.warning(
"The check-in date is more than 31 days away. "
"The model will use a lead time of 31 days."
)

# --------------------------------------------------

# 5. Occupancy

# --------------------------------------------------

occupancy = st.slider(
"Occupancy (%)",
min_value=0,
max_value=100,
value=50,
step=1
)

# --------------------------------------------------

# 6. Competitor Prices

# --------------------------------------------------

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

# --------------------------------------------------

# 7. Prediction

# --------------------------------------------------

if st.button("Predict Cheapest Rate", type="primary"):

```
# Validate competitor prices
if not (
    competitor_min <= competitor_median <= competitor_max
    and competitor_min <= competitor_avg <= competitor_max
):
    st.error(
        "Please check competitor prices. "
        "The minimum should not exceed the average, "
        "median, or maximum, and the maximum should "
        "not be below them."
    )

else:
    try:
        # Encode hotel ID using the saved encoder
        encoded_hotel_id = hotel_encoder.transform(
            [hotel_id]
        )[0]

        # IMPORTANT:
        # Feature names must match the training notebook.
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

        # Reorder columns to match the trained model
        if hasattr(model, "feature_names_in_"):
            input_data = input_data[
                list(model.feature_names_in_)
            ]

        # Predict rate
        prediction = float(model.predict(input_data)[0])

        # Display result
        st.success(
            f"Predicted Cheapest Rate: ₹{max(0, prediction):,.2f}"
        )

        st.metric(
            label="Estimated Cheapest Rate",
            value=f"₹{max(0, prediction):,.2f}"
        )

        st.caption(
            "This is a model estimate, not a live booking quote "
            "or a guaranteed available rate."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
```

# --------------------------------------------------

# 8. Footer

# --------------------------------------------------

st.divider()
st.caption(
"Hotel Price Prediction | Machine Learning Prototype"
)
