import streamlit as st
import pandas as pd
import joblib
from datetime import date

# --------------------------------------------------
# 1. Load trained model and encoder
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoder = joblib.load("hotel_encoder.pkl")
    return model, encoder

model, hotel_encoder = load_model()

# --------------------------------------------------
# 2. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Hotel Price Prediction",
    page_icon="🏨",
    layout="centered"
)

st.title("Hotel Cheapest Rate Predictor")
st.write(
    "Predict the expected cheapest hotel rate using "
    "lead time, occupancy, and competitor prices."
)

# --------------------------------------------------
# 3. Hotel selection
# --------------------------------------------------
hotel_ids = list(hotel_encoder.classes_)

hotel_id = st.selectbox(
    "Select Hotel ID",
    hotel_ids
)

# --------------------------------------------------
# 4. Check-in date
# --------------------------------------------------
check_in = st.date_input(
    "Check-in Date",
    value=date.today()
)

# --------------------------------------------------
# 5. Lead time (manual input)
# --------------------------------------------------
lead_time = st.number_input(
    "Lead Time (days)",
    min_value=1,
    max_value=31,
    value=7,
    step=1,
    help="Number of days between the scraping date and check-in date."
)

# --------------------------------------------------
# 6. Occupancy
# --------------------------------------------------
occupancy = st.slider(
    "Occupancy (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

# --------------------------------------------------
# 7. Competitor prices
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
# 8. Predict cheapest rate
# --------------------------------------------------
if st.button("Predict Cheapest Rate", type="primary"):

    # Validate competitor prices
    if not (
        competitor_min <= competitor_avg <= competitor_max
        and competitor_min <= competitor_median <= competitor_max
    ):
        st.error(
            "Invalid competitor rates. Ensure the minimum "
            "is not greater than the average, median, or maximum, "
            "and the maximum is not lower than them."
        )

    else:
        try:
            # Encode hotel ID
            encoded_hotel_id = hotel_encoder.transform(
                [hotel_id]
            )[0]

            # Prepare input features
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

            # Match the exact feature names and order
            # used when training the model
            if hasattr(model, "feature_names_in_"):
                input_data = input_data[
                    list(model.feature_names_in_)
                ]

            # Generate prediction
            prediction = float(model.predict(input_data)[0])

            # Display result
            st.success(
                f"Predicted Cheapest Rate: ₹{max(0, prediction):,.2f}"
            )

            st.metric(
                "Estimated Cheapest Rate",
                f"₹{max(0, prediction):,.2f}"
            )

            st.caption(
                "This is a model-based estimate, not a guaranteed "
                "live booking price."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --------------------------------------------------
# 9. Footer
# --------------------------------------------------
st.divider()
st.caption("Hotel Price Prediction | Machine Learning Prototype")
