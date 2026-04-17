# f1_sensitivity_index.py
# Championship + Driver-Level Sensitivity Index

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------
df = pd.read_csv("master_driver_race.csv")

# Filter years

df = df[(df["year"] >= 2014) & (df["year"] <= 2024)].copy()

# --------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------
df["positions_gained"] = df["grid"] - df["positionOrder"]
df["is_dnf"] = df["is_dnf"].astype(int)

df["champ_pressure"] = 1 / df["championship_position"].replace(0, np.nan)

df["constructor_form"] = df.groupby(["year", "constructor_name"])["points"].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

df["total_pit_stops"] = df["total_pit_stops"].fillna(0)

# --------------------------------------------------
# 3. Outcome Reversal Definition
# --------------------------------------------------

def classify_reversal(row):
    delta = row["grid"] - row["positionOrder"]
    if delta >= 3:
        return 1
    elif delta <= -3:
        return -1
    else:
        return 0


df["reversal_event"] = df.apply(lambda x: 1 if classify_reversal(x) != 0 else 0, axis=1)

# --------------------------------------------------
# 4. Championship-Level Sensitivity
# --------------------------------------------------
results = []
years = sorted(df["year"].unique())

for yr in years:
    season_df = df[df["year"] == yr]

    features = [
        "grid",
        "positions_gained",
        "champ_pressure",
        "constructor_form",
        "is_dnf",
        "total_pit_stops"
    ]

    X = season_df[features]
    y = season_df["reversal_event"]

    X = pd.concat([X, pd.get_dummies(season_df["constructor_name"], drop_first=True)], axis=1)
    X = X.fillna(0)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    sensitivity = pd.Series(rf.feature_importances_, index=X.columns)

    grid_effect = sensitivity.get("grid", 0)
    car_effect = sensitivity[sensitivity.index.str.contains("constructor_")].sum()
    risk_effect = sensitivity[[c for c in sensitivity.index if c in ["is_dnf", "total_pit_stops"]]].sum()

    total = grid_effect + car_effect + risk_effect

    results.append({
        "year": yr,
        "grid_sensitivity": grid_effect / total,
        "car_sensitivity": car_effect / total,
        "risk_sensitivity": risk_effect / total
    })

champ_sensitivity = pd.DataFrame(results)

print("\nChampionship Sensitivity Index by Year:")
print(champ_sensitivity)

# --------------------------------------------------
# 5. Driver-Level Sensitivity per Year
# --------------------------------------------------
driver_year_results = []

for yr in years:
    season_df = df[df["year"] == yr].copy()

    features = [
        "grid",
        "positions_gained",
        "champ_pressure",
        "constructor_form",
        "is_dnf",
        "total_pit_stops"
    ]

    X = season_df[features]
    y = season_df["reversal_event"]

    X = pd.concat([X, pd.get_dummies(season_df["constructor_name"], drop_first=True)], axis=1)
    X = X.fillna(0)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)

    sens = pd.Series(rf.feature_importances_, index=X.columns)
    sens = sens / sens.sum()

    X_driver = X.copy()
    X_driver["driver_name"] = season_df["driver_name"].values

    driver_avg = X_driver.groupby("driver_name").mean()

    driver_norm = (driver_avg - driver_avg.min()) / (driver_avg.max() - driver_avg.min() + 1e-9)

    common_cols = [c for c in driver_norm.columns if c in sens.index]

    dpsi = (driver_norm[common_cols] * sens[common_cols]).sum(axis=1)

    tmp = pd.DataFrame({
        "year": yr,
        "driver_name": dpsi.index,
        "DPSI": dpsi.values
    })

    driver_year_results.append(tmp)


driver_year_dpsi = pd.concat(driver_year_results, ignore_index=True)

print("\nTop Drivers by DPSI (sample):")
print(driver_year_dpsi.sort_values(["year", "DPSI"], ascending=[True, False]).head(10))

# --------------------------------------------------
# 6. Save Outputs
# --------------------------------------------------
champ_sensitivity.to_csv("championship_sensitivity_index.csv", index=False)
driver_year_dpsi.to_csv("driver_year_dpsi.csv", index=False)

# --------------------------------------------------
# 7. Plot Championship Trends
# --------------------------------------------------
plt.figure()
for col in ["grid_sensitivity", "car_sensitivity", "risk_sensitivity"]:
    plt.plot(champ_sensitivity["year"], champ_sensitivity[col], label=col)

plt.legend()
plt.title("Championship Sensitivity Trends")
plt.xlabel("Year")
plt.ylabel("Sensitivity")
plt.tight_layout()
plt.show()
