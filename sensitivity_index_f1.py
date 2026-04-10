# sensitivity_index_f1.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load Data
# ----------------------------
df = pd.read_csv("data/processed/master_driver_race.csv")

# ----------------------------
# 2. Feature Engineering
# ----------------------------
df["positions_gained"] = df["grid_position"] - df["finish_position"]

# Ensure DNF flag is binary
df["DNF_flag"] = df["dnf_flag"].apply(lambda x: 1 if x == 1 else 0)

# Avoid divide-by-zero issues
df["championship_position"] = df["championship_position"].replace(0, np.nan)
df["champ_pos_inv"] = 1 / df["championship_position"]

# Constructor rolling performance
df["constructor_avg_points"] = df.groupby("constructor")["points_scored"] \
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

# Drop NA rows created during feature engineering
df = df.dropna()

# ----------------------------
# 3. Define Outcome Variable
# ----------------------------
y = df["points_scored"]

# ----------------------------
# 4. Define Feature Set
# ----------------------------
X = df[[
    "grid_position",
    "positions_gained",
    "DNF_flag",
    "champ_pos_inv",
    "constructor_avg_points"
]]

# One-hot encode categorical variables
X = pd.concat([X, pd.get_dummies(df["constructor"], drop_first=True)], axis=1)

if "dnf_reason" in df.columns:
    X = pd.concat([X, pd.get_dummies(df["dnf_reason"], drop_first=True)], axis=1)

# ----------------------------
# 5. Standardization
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------
# 6. Linear Regression Sensitivity
# ----------------------------
lr_model = LinearRegression()
lr_model.fit(X_scaled, y)

lr_sensitivity = pd.DataFrame({
    "Variable": X.columns,
    "Sensitivity_Index": abs(lr_model.coef_)
}).sort_values(by="Sensitivity_Index", ascending=False)

print("\nLinear Regression Sensitivity Index:")
print(lr_sensitivity.head(10))

# ----------------------------
# 7. Random Forest Sensitivity
# ----------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=1)
rf_model.fit(X, y)

rf_sensitivity = pd.DataFrame({
    "Variable": X.columns,
    "Sensitivity_Index": rf_model.feature_importances_
}).sort_values(by="Sensitivity_Index", ascending=False)

# Normalize
rf_sensitivity["Sensitivity_Index"] = (
    rf_sensitivity["Sensitivity_Index"] /
    rf_sensitivity["Sensitivity_Index"].sum()
)

print("\nRandom Forest Sensitivity Index:")
print(rf_sensitivity.head(10))

# ----------------------------
# 8. Plot Results
# ----------------------------
plt.figure()
rf_sensitivity.head(10).plot(
    x="Variable",
    y="Sensitivity_Index",
    kind="barh",
    legend=False
)

plt.title("Top 10 Sensitivity Drivers (Random Forest)")
plt.xlabel("Sensitivity Index")
plt.tight_layout()
plt.show()
