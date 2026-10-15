# ============================================================
# Hotel Booking Cancellation Prediction - predict.py
# Test from project root:  python src/predict.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import os
import joblib
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")


# ------------------------------------------------------------
# 3. Load saved files from train.py
# ------------------------------------------------------------
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
encoder = joblib.load(os.path.join(MODELS_DIR, "encoder.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
threshold = joblib.load(os.path.join(MODELS_DIR, "threshold.pkl"))
children_median = joblib.load(os.path.join(MODELS_DIR, "children_median.pkl"))
country_mode = joblib.load(os.path.join(MODELS_DIR, "country_mode.pkl"))
cat_cols = joblib.load(os.path.join(MODELS_DIR, "cat_cols.pkl"))
num_cols = joblib.load(os.path.join(MODELS_DIR, "num_cols.pkl"))

month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}


# ------------------------------------------------------------
# 4. Prediction function (same steps as train.py)
# ------------------------------------------------------------
def predict_booking(booking):

    # dict -> one-row DataFrame
    df = pd.DataFrame([booking])

    # Fill nulls with the values learned in training
    df["children"] = df["children"].fillna(children_median)
    df["country"] = df["country"].fillna(country_mode)

    # Feature engineering
    df["has_agent"] = df["agent"].notnull().astype(int)
    df["has_company"] = df["company"].notnull().astype(int)
    df["agent_id"] = df["agent"].fillna(0).astype(int).astype(str)

    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["total_guests"] = df["adults"] + df["children"] + df["babies"]

    df["arrival_month"] = df["arrival_date_month"].map(month_map)
    df["month_sin"] = np.sin(2 * np.pi * df["arrival_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["arrival_month"] / 12)

    # Encoding
    df_cat = pd.DataFrame(encoder.transform(df[cat_cols]),
                          columns=encoder.get_feature_names_out())

    # Scaling
    df_num = pd.DataFrame(scaler.transform(df[num_cols]),
                          columns=num_cols)

    # Same column order as training
    df_final = pd.concat([df_num, df_cat], axis=1)

    # Predict
    probability = float(model.predict_proba(df_final)[0, 1])

    if probability >= threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction, probability


# ------------------------------------------------------------
# 5. Quick test with one sample booking
# ------------------------------------------------------------
if __name__ == "__main__":

    sample_booking = {
        "hotel": "City Hotel",
        "lead_time": 250,
        "arrival_date_month": "August",
        "arrival_date_week_number": 32,
        "arrival_date_day_of_month": 10,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Groups",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 1,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "Non Refund",
        "agent": 1,
        "company": None,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 90.0,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 0,
    }

    prediction, probability = predict_booking(sample_booking)

    print("Cancel probability:", round(probability * 100, 1), "%")
    print("Threshold         :", threshold)

    if prediction == 1:
        print("Prediction        : LIKELY TO CANCEL")
    else:
        print("Prediction        : LIKELY TO SHOW UP")
