from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import json

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

@app.route("/simulation-results", methods=["GET", "POST"])
def visualize_simulation():
    if request.method == "POST":
        year = request.form.get("year")
        races_str = request.form.get("races")
        if races_str:
            try:
                races = json.loads(races_str)
            except json.JSONDecodeError:
                races = []
        else:
            races = []
    else:
        year = None
        races = []
    return render_template("simulation_results.html", year=year, races=races)

@app.route("/simulate", methods=["GET", "POST"])
def dynamic():
    return render_template("dynamic.html")

if __name__ == "__main__":
    app.run(debug=True)