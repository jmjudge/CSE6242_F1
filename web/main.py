from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/f1-history")
def static_page():
    return render_template("static.html")

@app.route("/historical-progression")
def historical_progression():
    return render_template("f1_season_progression.html")

@app.route("/simulate", methods=["GET", "POST"])
def dynamic():
    return render_template("dynamic.html")

if __name__ == "__main__":
    app.run(debug=True)