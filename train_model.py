"""
train_model.py
Trains a Random Forest model on IPL data and saves model.pkl + encoders.
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")


# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv")
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── 2. Feature Engineering ────────────────────────────────────────────────────
df["run_rate"]         = df["current_score"] / df["overs"].replace(0, 1)
df["wickets_left"]     = 10 - df["wickets"]
df["balls_remaining"]  = (20 - df["overs"]) * 6
df["rr_last5"]         = df["runs_last5"] / 5.0
df["pressure_index"]   = df["wickets"] / df["overs"].replace(0, 1)

FEATURES = [
    "batting_team", "bowling_team",
    "overs", "current_score", "wickets",
    "runs_last5", "wickets_last5",
    "run_rate", "wickets_left", "balls_remaining",
    "rr_last5", "pressure_index",
]
TARGET = "final_score"

# ── 3. Encode categorical columns ─────────────────────────────────────────────
le_bat  = LabelEncoder()
le_bowl = LabelEncoder()

df["batting_team"]  = le_bat.fit_transform(df["batting_team"])
df["bowling_team"]  = le_bowl.fit_transform(df["bowling_team"])

# ── 4. Train / Test split ─────────────────────────────────────────────────────
X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

# ── 5. Train models & compare ─────────────────────────────────────────────────
models = {
    "RandomForest":      RandomForestRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=3, random_state=42, n_jobs=-1),
    "GradientBoosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                                   max_depth=5, random_state=42),
    "Ridge":             Ridge(alpha=10.0),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    cv    = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    results[name] = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "CV_MAE": round(cv, 2)}
    print(f"{name:25s} → MAE: {mae:.2f}  RMSE: {rmse:.2f}  CV_MAE: {cv:.2f}")

# ── 6. Pick best model (lowest CV MAE) ────────────────────────────────────────
best_name = min(results, key=lambda k: results[k]["CV_MAE"])
best_model = models[best_name]
print(f"\nBest model: {best_name}")

# ── 7. Save artefacts ─────────────────────────────────────────────────────────
joblib.dump(best_model, "model.pkl")
joblib.dump(le_bat,     "le_bat.pkl")
joblib.dump(le_bowl,    "le_bowl.pkl")

meta = {
    "best_model":  best_name,
    "features":    FEATURES,
    "metrics":     results[best_name],
    "teams":       sorted(list(le_bat.classes_)),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved: model.pkl, le_bat.pkl, le_bowl.pkl, model_meta.json")
print(f"Final MAE  : {results[best_name]['MAE']}")
print(f"Final RMSE : {results[best_name]['RMSE']}")
