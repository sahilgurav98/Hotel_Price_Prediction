import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Hotel Cheapest Rate Predictor", page_icon="🏨")

st.title("🏨 Hotel Cheapest Rate Predictor")
st.write("Predict the historical cheapest hotel rate from hotel, date, occupancy, and competitor pricing.")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    encoder = joblib.load("hotel_encoder.pkl")
    return model, encoder

try:
    model, le = load_artifacts()
except Exception:
    st.error("Place model.pkl and hotel_encoder.pkl in the same folder as app.py.")
    st.stop()

st.sidebar.header("Hotel & Date")

hotel_id = st.sidebar.selectbox("Hotel ID", [str(x) for x in le.classes_])
check_in = st.sidebar.date_input("Check-in Date")

st.sidebar.header("Occupancy")
occupancy = st.sidebar.slider("Occupancy (%)", 0.0, 100.0, 50.0, 1.0)

st.sidebar.header("Competitor Pricing")
competitor_min = st.sidebar.number_input("Competitor Minimum Rate", 0.0, value=2000.0, step=100.0)
competitor_avg = st.sidebar.number_input("Competitor Average Rate", 0.0, value=5000.0, step=100.0)
competitor_median = st.sidebar.number_input("Competitor Median Rate", 0.0, value=4000.0, step=100.0)
competitor_max = st.sidebar.number_input("Competitor Maximum Rate", 0.0, value=8500.0, step=100.0)

if st.button("💰 Predict Cheapest Rate", use_container_width=True):
    hotel_encoded = le.transform([hotel_id])[0]

    input_data = pd.DataFrame({
        "hotel_id": [hotel_encoded],
        "day_of_month": [check_in.day],
        "day_of_week": [check_in.weekday()],
        "occupancy": [occupancy],
        "competitor_min": [competitor_min],
        "competitor_avg": [competitor_avg],
        "competitor_median": [competitor_median],
        "competitor_max": [competitor_max]
    })

    prediction = max(0, model.predict(input_data)[0])

    st.success(f"### Predicted Cheapest Rate: ₹{prediction:,.2f}")

    st.subheader("Input Summary")
    summary = pd.DataFrame({
        "Parameter": ["Hotel ID", "Check-in Date", "Day of Month", "Day of Week",
                      "Occupancy", "Competitor Minimum", "Competitor Average",
                      "Competitor Median", "Competitor Maximum"],
        "Value": [hotel_id, str(check_in), check_in.day, check_in.weekday(),
                  f"{occupancy:.0f}%", f"₹{competitor_min:,.2f}",
                  f"₹{competitor_avg:,.2f}", f"₹{competitor_median:,.2f}",
                  f"₹{competitor_max:,.2f}"]
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

st.divider()
st.caption("Model: Gradient Boosting Regressor | Target: historical cheapest hotel rate")
