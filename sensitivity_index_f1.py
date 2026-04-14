# f1_sensitivity_index.py
# Sensitivity Index for Outcome-Reversal Probability & Entropy (2014–2024)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
df = pd.read_csv("master_driver_race.csv")

# Keep only 2014–2024 era
df = df[(df["year"] >= 2014) & (df["year"] <= 2024)].copy()

# --------------------------------------------------
# 2. Clean Data & Feature Engineering
# --------------------------------------------------
# Positions gained
df["positions_gained"] = df["grid"] - df["positionOrder"]

# Ensure DNF is numeric
df["is_dnf"] = df["is_dnf"].astype(int)

# Championship pressure (inverse of championship position)
df["champ_pressure"] = 1 / df["championship_position"].replace(0, np.nan)

# Constructor strength proxy (rolling average of points)
df["constructor_form"] = df.groupby("constructor_name")["points"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

# Replace missing pit stop values with 0 (driver did not pit or data missing)
df["total_pit_stops"] = df["total_pit_stops"].fillna(0)
df["total_pit_stop_duration_sec"] = df["total_pit_stop_duration_sec"].fillna(0)

# --------------------------------------------------
# 3. Outcome Reversal Definition
# --------------------------------------------------
# If driver finishes >=3 places better than grid => positive reversal
# If finishes >=3 places worse => negative reversal

def classify_reversal(row):
    delta = row["grid"] - row["positionOrder"]
    if delta >= 3:
        return 1
    elif delta <= -3:
        return -1
    else:
        return 0


df["outcome_reversal"] = df.apply(classify_reversal, axis=1)
df["reversal_event"] = df["outcome_reversal"].apply(lambda x: 1 if x != 0 else 0)

# --------------------------------------------------
# 4. Race Entropy (Unpredictability)
# --------------------------------------------------
# Higher entropy means less predictable race results

entropy_by_race = df.groupby(["year", "raceId"]).apply(
    lambda x: entropy(np.histogram(x["positionOrder"], bins=20, density=True)[0] + 1e-9)
).rename("race_entropy").reset_index()

df = df.merge(entropy_by_race, on=["year", "raceId"], how="left")

# --------------------------------------------------
# 5. Feature Matrix
# --------------------------------------------------
features = [
    "grid",
    "positions_gained",
    "champ_pressure",
    "constructor_form",
    "is_dnf",
    "total_pit_stops",
    "race_entropy"
]

X = df[features]
y = df["reversal_event"]

# Add constructor effect
X = pd.concat([X, pd.get_dummies(df["constructor_name"], drop_first=True)], axis=1)

# Drop missing values
X = X.fillna(0)

# --------------------------------------------------
# 6. Train Random Forest
# --------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X, y)

sensitivity = pd.DataFrame({
    "Feature": X.columns,
    "Sensitivity": rf.feature_importances_
}).sort_values(by="Sensitivity", ascending=False)

sensitivity["Sensitivity"] /= sensitivity["Sensitivity"].sum()

print("\nTop Sensitivity Drivers:")
print(sensitivity.head(10))

# --------------------------------------------------
# 7. Season Simulation
# --------------------------------------------------
n_simulations = 500
reversal_rates = []

for _ in range(n_simulations):
    probs = rf.predict_proba(X)[:, 1]
    sim = np.random.binomial(1, probs)
    reversal_rates.append(sim.mean())

print("\nMean Outcome-Reversal Probability:", np.mean(reversal_rates))
print("Std Dev:", np.std(reversal_rates))

plt.hist(reversal_rates, bins=30)
plt.title("Outcome-Reversal Probability Distribution")
plt.show()

# --------------------------------------------------
# 8. Driver Performance Sensitivity Index (DPSI)
# --------------------------------------------------
# Weight driver features by sensitivity values
feature_weights = sensitivity.set_index("Feature")["Sensitivity"]

feature_cols = [c for c in X.columns if c in feature_weights.index]
driver_features = X[feature_cols].copy()
driver_features["driver_name"] = df["driver_name"]

driver_avg = driver_features.groupby("driver_name").mean()

# Normalize
norm = (driver_avg - driver_avg.min()) / (driver_avg.max() - driver_avg.min() + 1e-9)

# Weighted sum
for f in feature_cols:
    norm[f] *= feature_weights.get(f, 0)

norm["DPSI"] = norm.sum(axis=1)

print("\nTop Drivers by DPSI:")
print(norm[["DPSI"]].sort_values(by="DPSI", ascending=False).head(10))
