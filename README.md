# Hotel Cheapest Rate Prediction - Streamlit App

## Files

```text
hotel_pricing_app/
├── app.py
├── model.pkl
├── hotel_encoder.pkl
└── requirements.txt
```

## Export the trained model from the notebook

Run this after the model and LabelEncoder have been trained:

```python
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(le, "hotel_encoder.pkl")
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model inputs

The app uses the same eight features as the current notebook:

- hotel_id
- day_of_month
- day_of_week
- occupancy
- competitor_min
- competitor_avg
- competitor_median
- competitor_max

The output is the predicted historical cheapest hotel rate.
