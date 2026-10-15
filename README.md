# 🏨 Hotel Booking Cancellation Predictor

**Live app:** https://hotel-cancellation-predictor-kwvppyv5qdmeqibwyhzyxx.streamlit.app/

Hotels lose a lot of money when guests cancel at the last minute, so I wanted to build something that flags risky bookings *before* they cancel. You enter a booking's details, and the app tells you how likely the guest is to cancel, what's driving that risk, and how much revenue is at stake.


---

## 📊 Results

I compared three models, tuned each one with GridSearchCV, and XGBoost came out on top:

| Model | Test Accuracy | Test F1 |
|---|---|---|
| Logistic Regression | 76.3% | 0.65 |
| Random Forest | 81.2% | 0.71 |
| **XGBoost** ✅ | **83.2%** | **0.72** |

**Final XGBoost model (test set):**
- Accuracy: **83.2%**
- ROC-AUC: **0.91**
- Recall on cancellations: **79%**, so it catches about 8 out of 10 cancellations
- Train accuracy is 89.1%, so the gap to the test score is small and there's no heavy overfitting

---

## 💡 What I learned (and the traps I avoided)

**1. Data leakage was the biggest trap.**
The dataset has a `reservation_status` column that literally says "Canceled" or "Check-Out". Keep it and you get a fake ~100% accuracy. I also dropped `assigned_room_type` (the room is assigned at check-in, *after* the event) and `reservation_status_date`. A lot of projects online claim 95%+ on this dataset. My 83% is lower, but it's honest.

**2. 27% of the rows were exact duplicates.**
If I'd left them in, the same booking could land in both train and test and inflate the score. So I removed them, which left about 87K unique bookings.

**3. Accuracy alone wasn't enough.**
My first XGBoost model had 83% accuracy but missed 4 in 10 cancellations. Adding `scale_pos_weight` for the class imbalance, plus tuning the decision threshold (0.542 instead of 0.5), pushed recall from **62% to 79%**, and accuracy stayed about the same.

**4. Split first, then preprocess.**
I fit the imputers, encoder and scaler on the training data only, so nothing from the test set leaks in.

---

## 🔍 Key insights from EDA

- **Non-refundable deposits** were cancelled 99% of the time. Surprising, and probably a quirk of group or agent bookings.
- **Lead time** matters a lot: about 10% of last-minute bookings cancel, versus 68% of those booked a year ahead.
- Guests who **cancelled before** cancelled again 92% of the time.
- **Special requests** and **parking requests** are strong signs that the guest will actually show up.

---

## 🛠️ What I did, step by step

1. **Cleaning:** removed leakage columns, duplicates, zero-guest bookings and negative prices
2. **Feature engineering:** `total_nights`, `total_guests`, `has_agent`, `has_company`, and sin/cos encoding of the arrival month
3. **Missing values:** median for `children`, mode for `country`; for `agent` and `company`, missing means "none"
4. **Encoding:** One-Hot for the nominal columns, with rare countries and agents grouped together
5. **Scaling:** StandardScaler on the numeric columns
6. **Modelling:** Logistic Regression, Random Forest and XGBoost, with GridSearchCV and regularization
7. **Threshold tuning:** picked the threshold with the best F1 using cross-validation on the training data
8. **Deployment:** Streamlit app on Streamlit Community Cloud

---

## 📁 Project structure

```
hotel_cancel_prediction_project/
├── .streamlit/config.toml   # app theme
├── models/                  # saved model, encoder, scaler, threshold
├── src/
│   ├── train.py             # cleaning, training, evaluation
│   ├── predict.py           # prediction for a single booking
│   └── app.py               # Streamlit app
├── hotel_bookings.csv
└── requirements.txt
```

---

## ▶️ Run it locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
streamlit run src/app.py
```

---

## 📦 Dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) on Kaggle: 119,390 bookings from a city hotel and a resort hotel in Portugal, 2015–2017.

---

## 🧰 Tech stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · Streamlit · Joblib

---

## 👋 About me

I'm Rajat, a mechanical engineer turned data scientist, building end-to-end ML projects.

[GitHub](https://github.com/ranarajat994) · [LinkedIn](https://linkedin.com/in/rajat-rana-86365991)
