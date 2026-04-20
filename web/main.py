from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.monte_carlo import MonteCarloSimulator

app = Flask(__name__)

def load_f1_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'master_driver_race.csv')
    df = pd.read_csv(csv_path)
    df = df[df['year'] >= 2014]  # Filter to 2010 onwards like the hardcoded data
    data = {}
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        # Get all rounds in order
        rounds_data = year_df[['round', 'race_name']].drop_duplicates().sort_values('round')
        rounds = rounds_data['round'].tolist()
        races = rounds_data['race_name'].tolist()
        
        drivers_group = year_df.groupby(['driver_name', 'constructor_name'])
        series = []
        for (driver, constructor), group in drivers_group:
            # Create a mapping of round -> cumulative_points
            round_to_points = dict(zip(group['round'], group['cumulative_points']))
            
            # Build points array for all rounds, with None for missed races
            points = []
            for round_num in rounds:
                if round_num in round_to_points:
                    points.append(round_to_points[round_num])
                else:
                    # Driver missed this race; add None
                    points.append(None)
            
            # Get the final points (last non-None value, or 0 if all None)
            final_points = next((p for p in reversed(points) if p is not None), 0)
            
            series.append({'driver': driver, 'constructor': constructor, 'points': points, 'final_points': final_points})
        
        # Sort series by final points descending
        series.sort(key=lambda x: x['final_points'], reverse=True)
        data[str(year)] = {'races': races, 'series': series[:10]}
    return data


def load_driver_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'master_driver_race.csv')
    df = pd.read_csv(csv_path)
    df = df[df['year'] >= 2014]
    drivers_by_year = {}
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        race_groups = year_df.groupby('race_name')
        year_drivers = {}
        for race_name, group in race_groups:
            drivers = sorted(group['driver_name'].drop_duplicates().tolist())
            year_drivers[race_name] = drivers
        drivers_by_year[str(year)] = year_drivers
    return drivers_by_year


def load_master_driver_race_df():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'master_driver_race.csv')
    df = pd.read_csv(csv_path)
    df = df[df['year'] >= 2014]
    return df

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/f1-history")
def static_page():
    return render_template("static.html")

@app.route("/historical-progression")
def historical_progression():
    data = load_f1_data()
    return render_template("f1_season_progression.html", data=data)

@app.route("/csi-analysis")
def csi_analysis():
    # Load CSI data from summary.json
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'simulations', 'summary.json')
    with open(summary_path, 'r') as f:
        csi_data = json.load(f)
    
    # Calculate stats
    csi_values = [d['csi'] for d in csi_data]
    avg_csi = sum(csi_values) / len(csi_values) if csi_values else 0
    min_csi = min(csi_values) if csi_values else 0
    max_csi = max(csi_values) if csi_values else 0
    
    return render_template("static_csi.html", 
                         csi_data=csi_data,
                         avg_csi=avg_csi,
                         min_csi=min_csi,
                         max_csi=max_csi)

@app.route("/simulation-results", methods=["GET", "POST"])
def visualize_simulation():
    sim_results = None
    if request.method == "POST":
        config_str = request.form.get("config")   ## this is the user-submitted data from the form in dynamic.html as a JSON object
        if config_str:
            try:
                config_data = json.loads(config_str)
                year = config_data.get("year")
                races = config_data.get("configs", [])
            except json.JSONDecodeError:
                year = None
                races = []
        else:
            year = None
            races = []

        if year and races:
            df = load_master_driver_race_df()
            simulator = MonteCarloSimulator(df)
            sim_results = simulator.simulate_season(year, n_simulations=100, configs=races)
    else:
        year = None
        races = []

    return render_template(
        "simulation_results.html",
        year=year,
        races=races,
        sim_results=sim_results,
    )

@app.route("/simulate", methods=["GET", "POST"])
def dynamic():
    driver_data = load_driver_data()
    return render_template("dynamic.html", driver_data=driver_data)

if __name__ == "__main__":
    app.run(debug=True)