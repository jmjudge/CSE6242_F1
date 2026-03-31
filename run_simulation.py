import json
import os
import pandas as pd
from src.monte_carlo import MonteCarloSimulator

# Load og data & initialize sim
df = pd.read_csv("data/processed/master_driver_race.csv")

sim = MonteCarloSimulator(df)

# Limit years to simulate. ONLY use 2014 - 2024 for modern f1
years = [y for y in sorted(df['year'].unique()) if 2014 <= y <= 2024]

all_results = []

for year in years:
    print(f"Running simulation for {year}...")

    results = sim.simulate_season(year=year, n_simulations=100)

    # Save individual file for each year
    output_path = f"outputs/simulations/{year}_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    all_results.append({
        "year": results["year"],
        "csi": results["csi"],
        "top_driver": list(results["win_probabilities"].keys())[0],
        "top_driver_prob": list(results["win_probabilities"].values())[0],
        "num_contenders": len(results["win_probabilities"])
    })
# Save summary file with all the years
with open("outputs/simulations/summary.json", "w") as f:
    json.dump(all_results, f, indent=4)

print("All simulations complete.")