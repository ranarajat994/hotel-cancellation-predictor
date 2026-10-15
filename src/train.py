# ============================================================
# Hotel Booking Cancellation Prediction - train.py
# Run from project root:  python src/train.py
# ============================================================

# ------------------------------------------------------------
# 1. Import libraries
# ------------------------------------------------------------
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "hotel_bookings.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Original shape:", df.shape)
print(df.head())


# ------------------------------------------------------------
# 4. Check null values
# ------------------------------------------------------------
print("\nNull values:")
print(df.isnull().sum())


# ------------------------------------------------------------
# 5. Drop leakage columns
#    reservation_status      -> tells us directly if guest cancelled
#    reservation_status_date -> date set AFTER cancel / check-out
#    assigned_room_type      -> room given at check-in (after the event)
# ------------------------------------------------------------
df = df.drop(columns=["reservation_status"])
df = df.drop(columns=["reservation_status_date"])
df = df.drop(columns=["assigned_room_type"])


# ------------------------------------------------------------
# 6. Basic cleaning
# ------------------------------------------------------------
# children column is text -> convert to number
df["children"] = pd.to_numeric(df["children"], errors="coerce")

# Remove bookings with 0 guests
df = df[(df["adults"] + df["children"].fillna(0) + df["babies"]) > 0]

# Remove duplicate rows (27% of the data is exact copies)
df = df.drop_duplicates()

# Remove negative adr and cap the extreme value
df = df[df["adr"] >= 0]
df["adr"] = df["adr"].clip(upper=1000)

print("\nShape after cleaning:", df.shape)


# ------------------------------------------------------------
# 7. Feature engineering
# ------------------------------------------------------------
# agent and company: missing means "no agent" / "no company"
df["has_agent"] = df["agent"].notnull().astype(int)
df["has_company"] = df["company"].notnull().astype(int)
df["agent_id"] = df["agent"].fillna(0).astype(int).astype(str)

# Total nights and total guests
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
df["total_guests"] = df["adults"] + df["children"].fillna(0) + df["babies"]

# Month name -> number (ordinal, so mapped manually)
month_map = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}
df["arrival_month"] = df["arrival_date_month"].map(month_map)

# Cyclical encoding: December (12) sits next to January (1)
df["month_sin"] = np.sin(2 * np.pi * df["arrival_month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["arrival_month"] / 12)


# ------------------------------------------------------------
# 8. Feature selection
# ------------------------------------------------------------
df = df.drop(columns=["agent"])                  # replaced by agent_id + has_agent
df = df.drop(columns=["company"])                # replaced by has_company
df = df.drop(columns=["arrival_date_month"])     # replaced by month_sin / month_cos
df = df.drop(columns=["arrival_month"])          # replaced by month_sin / month_cos
df = df.drop(columns=["arrival_date_year"])      # future bookings will have new years


# ------------------------------------------------------------
# 9. Split X and y
# ------------------------------------------------------------
X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

print("\nTarget distribution:")
print(y.value_counts(normalize=True))


# ------------------------------------------------------------
# 10. Train test split
#     (done BEFORE filling nulls / encoding / scaling so test data
#      does not leak into training)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nX_train:", X_train.shape)
print("X_test :", X_test.shape)


# ------------------------------------------------------------
# 11. Fill null values (values learned from TRAIN only)
#     children -> median (numeric)
#     country  -> mode (categorical)
# ------------------------------------------------------------
children_median = X_train["children"].median()
country_mode = X_train["country"].mode()[0]

X_train["children"] = X_train["children"].fillna(children_median)
X_test["children"] = X_test["children"].fillna(children_median)

X_train["country"] = X_train["country"].fillna(country_mode)
X_test["country"] = X_test["country"].fillna(country_mode)

X_train["total_guests"] = X_train["adults"] + X_train["children"] + X_train["babies"]
X_test["total_guests"] = X_test["adults"] + X_test["children"] + X_test["babies"]

print("\nNulls left in X_train:", X_train.isnull().sum().sum())
print("Nulls left in X_test :", X_test.isnull().sum().sum())


# ------------------------------------------------------------
# 12. Encoding
#     All these columns are NOMINAL (no order) -> OneHotEncoder
#     Month was the only ordinal one and is already numeric.
#     Rare countries / agents (< 50 bookings) become "infrequent".
# ------------------------------------------------------------
cat_cols = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "deposit_type", "customer_type", "agent_id",
]

num_cols = [
    "lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list",
    "adr", "required_car_parking_spaces", "total_of_special_requests",
    "has_agent", "has_company", "total_nights", "total_guests",
    "month_sin", "month_cos",
]

encoder = OneHotEncoder(handle_unknown="infrequent_if_exist",
                        min_frequency=50,
                        sparse_output=False)
encoder.fit(X_train[cat_cols])

X_train_cat = pd.DataFrame(encoder.transform(X_train[cat_cols]),
                           columns=encoder.get_feature_names_out(),
                           index=X_train.index)
X_test_cat = pd.DataFrame(encoder.transform(X_test[cat_cols]),
                          columns=encoder.get_feature_names_out(),
                          index=X_test.index)


# ------------------------------------------------------------
# 13. Scaling (numeric columns only, fit on TRAIN only)
# ------------------------------------------------------------
scaler = StandardScaler()
scaler.fit(X_train[num_cols])

X_train_num = pd.DataFrame(scaler.transform(X_train[num_cols]),
                           columns=num_cols,
                           index=X_train.index)
X_test_num = pd.DataFrame(scaler.transform(X_test[num_cols]),
                          columns=num_cols,
                          index=X_test.index)

X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_final = pd.concat([X_test_num, X_test_cat], axis=1)

print("\nFinal X_train:", X_train_final.shape)
print("Final X_test :", X_test_final.shape)

# PCA is NOT applied: only ~120 columns, tree models don't need it,
# and PCA would make the features impossible to explain in the app.


# ------------------------------------------------------------
# 14. Class imbalance ratio (for class_weight / scale_pos_weight)
# ------------------------------------------------------------
not_cancelled = (y_train == 0).sum()
cancelled = (y_train == 1).sum()
imbalance_ratio = round(not_cancelled / cancelled, 2)
print("\nImbalance ratio:", imbalance_ratio)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# ------------------------------------------------------------
# 15. Model 1 - Logistic Regression (L2 regularization via C)
# ------------------------------------------------------------
print("\n========== Logistic Regression ==========")
lr_params = {
    "C": [0.01, 0.1, 1, 10],
    "class_weight": [None, "balanced"],
}
lr_grid = GridSearchCV(LogisticRegression(max_iter=2000), lr_params,
                       cv=cv, scoring="f1", n_jobs=-1)
lr_grid.fit(X_train_final, y_train)
lr_model = lr_grid.best_estimator_

lr_train_pred = lr_model.predict(X_train_final)
lr_test_pred = lr_model.predict(X_test_final)
print("Best params   :", lr_grid.best_params_)
print("Train accuracy:", round(accuracy_score(y_train, lr_train_pred), 4))
print("Test accuracy :", round(accuracy_score(y_test, lr_test_pred), 4))
print("Test F1       :", round(f1_score(y_test, lr_test_pred), 4))


# ------------------------------------------------------------
# 16. Model 2 - Random Forest (max_depth / min_samples_leaf = regularization)
# ------------------------------------------------------------
print("\n========== Random Forest ==========")
rf_params = {
    "max_depth": [15, 25],
    "min_samples_leaf": [1, 3],
    "class_weight": [None, "balanced"],
}
rf_grid = GridSearchCV(RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                       rf_params, cv=cv, scoring="f1", n_jobs=1)
rf_grid.fit(X_train_final, y_train)
rf_model = rf_grid.best_estimator_

rf_train_pred = rf_model.predict(X_train_final)
rf_test_pred = rf_model.predict(X_test_final)
print("Best params   :", rf_grid.best_params_)
print("Train accuracy:", round(accuracy_score(y_train, rf_train_pred), 4))
print("Test accuracy :", round(accuracy_score(y_test, rf_test_pred), 4))
print("Test F1       :", round(f1_score(y_test, rf_test_pred), 4))


# ------------------------------------------------------------
# 17. Model 3 - XGBoost (reg_lambda = L2 regularization,
#     scale_pos_weight = handles class imbalance)
# ------------------------------------------------------------
print("\n========== XGBoost ==========")
xgb_params = {
    "max_depth": [6, 8],
    "learning_rate": [0.05, 0.1],
    "reg_lambda": [1, 5],
    "scale_pos_weight": [1, imbalance_ratio],
}
xgb_grid = GridSearchCV(XGBClassifier(n_estimators=400, eval_metric="logloss",
                                      random_state=42, n_jobs=-1),
                        xgb_params, cv=cv, scoring="f1", n_jobs=1)
xgb_grid.fit(X_train_final, y_train)
xgb_model = xgb_grid.best_estimator_

xgb_train_pred = xgb_model.predict(X_train_final)
xgb_test_pred = xgb_model.predict(X_test_final)
print("Best params   :", xgb_grid.best_params_)
print("Train accuracy:", round(accuracy_score(y_train, xgb_train_pred), 4))
print("Test accuracy :", round(accuracy_score(y_test, xgb_test_pred), 4))
print("Test F1       :", round(f1_score(y_test, xgb_test_pred), 4))


# ------------------------------------------------------------
# 18. Choose the best model by CV F1 score
# ------------------------------------------------------------
print("\nCV F1 - Logistic Regression:", round(lr_grid.best_score_, 4))
print("CV F1 - Random Forest      :", round(rf_grid.best_score_, 4))
print("CV F1 - XGBoost            :", round(xgb_grid.best_score_, 4))

best_name = "XGBoost"
best_model = xgb_model
best_score = xgb_grid.best_score_

if rf_grid.best_score_ > best_score:
    best_name = "Random Forest"
    best_model = rf_model
    best_score = rf_grid.best_score_

if lr_grid.best_score_ > best_score:
    best_name = "Logistic Regression"
    best_model = lr_model
    best_score = lr_grid.best_score_

print("\nBest model:", best_name)


# ------------------------------------------------------------
# 19. Find the best decision threshold (instead of default 0.5)
#     Uses cross-validated probabilities on TRAIN data only.
# ------------------------------------------------------------
train_cv_proba = cross_val_predict(best_model, X_train_final, y_train,
                                   cv=cv, method="predict_proba")[:, 1]

precision, recall, thresholds = precision_recall_curve(y_train, train_cv_proba)
f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
best_threshold = round(float(thresholds[np.argmax(f1_values)]), 3)
print("Best threshold:", best_threshold)


# ------------------------------------------------------------
# 20. Final result on train and test
# ------------------------------------------------------------
train_proba = best_model.predict_proba(X_train_final)[:, 1]
test_proba = best_model.predict_proba(X_test_final)[:, 1]

train_pred = (train_proba >= best_threshold).astype(int)
test_pred = (test_proba >= best_threshold).astype(int)

print("\n========== FINAL RESULT -", best_name, "==========")
print("Train accuracy:", round(accuracy_score(y_train, train_pred), 4))
print("Test accuracy :", round(accuracy_score(y_test, test_pred), 4))
print("Test F1       :", round(f1_score(y_test, test_pred), 4))
print("Test ROC-AUC  :", round(roc_auc_score(y_test, test_proba), 4))

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, test_pred))

print("\nClassification report (test):")
print(classification_report(y_test, test_pred))


# ------------------------------------------------------------
# 21. Save everything the app needs
# ------------------------------------------------------------
joblib.dump(best_model, os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(encoder, os.path.join(MODELS_DIR, "encoder.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(best_threshold, os.path.join(MODELS_DIR, "threshold.pkl"))
joblib.dump(children_median, os.path.join(MODELS_DIR, "children_median.pkl"))
joblib.dump(country_mode, os.path.join(MODELS_DIR, "country_mode.pkl"))
joblib.dump(cat_cols, os.path.join(MODELS_DIR, "cat_cols.pkl"))
joblib.dump(num_cols, os.path.join(MODELS_DIR, "num_cols.pkl"))

print("\nAll files saved in:", MODELS_DIR)
