import pandas as pd
import numpy as np
from collections import defaultdict

class MonteCarloSimulator:
    def __init__(self, df):
        """
        df: master_driver_race dataframe
        """
        self.df = df.copy()

        self.dnf_prob = 0.1  # tweak this val later if needed

        # scoring system (F1 points)
        self.points_system = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }

    def _build_race_labels(self, df_year):
        if 'round' in df_year.columns:
            order = df_year.drop_duplicates('raceId').set_index('raceId')['round']
        elif 'race_round' in df_year.columns:
            order = df_year.drop_duplicates('raceId').set_index('raceId')['race_round']
        else:
            order = pd.Series(
                data=range(1, len(df_year['raceId'].unique()) + 1),
                index=sorted(df_year['raceId'].unique())
            )

        labels = {}
        for race_id, race_order in order.items():
            label = (
                df_year.loc[df_year['raceId'] == race_id, 'raceName'].iloc[0]
                if 'raceName' in df_year.columns else f"Race {int(race_order)}"
            )
            labels[race_id] = {"order": int(race_order), "label": str(label)}

        return labels

    def apply_random_dnf(self, df_race, dnf_prob=0.05):
        """
        Randomly assign DNFs to drivers in a race.
        """
        df_race = df_race.copy()

        for idx in df_race.index:
            if np.random.rand() < dnf_prob:
                df_race.at[idx, 'is_dnf'] = True
                df_race.at[idx, 'status'] = 'DNF'
                df_race.at[idx, 'positionOrder'] = np.nan

        return df_race

    def recompute_race(self, df_race):
        """
        Re-rank the positions and assign points
        """
        df_race = df_race.copy()

        finishers = df_race[df_race['is_dnf'] == False].copy()
        dnfs = df_race[df_race['is_dnf'] == True].copy()

        finishers = finishers.sort_values('positionOrder')
        finishers['positionOrder'] = range(1, len(finishers) + 1)

        finishers['points'] = finishers['positionOrder'].map(self.points_system).fillna(0)
        dnfs['points'] = 0

        return pd.concat([finishers, dnfs])

    def simulate_season(self, year, n_simulations=100, df_override=None, configs=None):
        """
        Monte Carlo season simulation
        """

        # counterfactual mode (unchanged)
        if configs is not None:
            return self.simulate_configs(year, configs, n_simulations)

        if df_override is not None:
            df_year = df_override[df_override["year"] == int(year)].copy()
        else:
            df_year = self.df[self.df["year"] == int(year)].copy()

        if df_year.empty:
            raise ValueError(f"No data found for year {year}")

        df_year['scenario_dnf_prob'] = df_year.get('scenario_dnf_prob', np.nan)
      ##  df_year['scenario_time_delay'] = df_year.get('scenario_time_delay', np.nan)

        race_labels = self._build_race_labels(df_year)
        race_order_map = {race_id: race_labels[race_id]['order'] for race_id in race_labels}
        cumulative_point_sums = defaultdict(float)
        cumulative_point_counts = defaultdict(int)

        print("YEAR:", year)
        print("ROWS:", len(df_year))
        print("Drivers:", df_year["driver_name"].nunique())
        print(df_year["driver_name"].value_counts().head(10))
        print(df_year[["driver_name", "positionOrder"]].head(20))

        champion_counts = {}

        # DRIVER BASELINE STRENGTH... lower = better driver historically
        driver_strength = (
            df_year.groupby("driverId")["positionOrder"]
            .mean()
            .to_dict()
        )

        for _ in range(n_simulations):
            sim_df = df_year.copy()

            # attach driver strength
            sim_df["strength"] = sim_df["driverId"].map(driver_strength)

            # RANDOM DNFs (with scenario overrides)
            dnf_probabilities = np.where(
                sim_df['scenario_dnf_prob'].notna(),
                sim_df['scenario_dnf_prob'],
                self.dnf_prob
            )
            dnf_mask = np.random.rand(len(sim_df)) < dnf_probabilities
            sim_df.loc[dnf_mask, "is_dnf"] = True

            # PERFORMANCE MODEL
            noise = np.random.normal(0, 1.5, len(sim_df))

     #       delay_noise = np.random.normal(0, 0.1, len(sim_df))
     #       sim_df["sim_time_delay"] = (
     #           sim_df["scenario_time_delay"].fillna(0.0) + delay_noise
     #       ).clip(lower=0.0)

            sim_df["sim_score"] = sim_df["strength"] + noise ## + sim_df["sim_time_delay"]
            # DNFs go to back
            sim_df.loc[sim_df["is_dnf"], "sim_score"] = 999

            # RACE RANKING
            sim_df["sim_position"] = sim_df.groupby("raceId")["sim_score"] \
                .rank(method="first", ascending=True)

            # POINTS
            sim_df["sim_points"] = sim_df["sim_position"].map(self.points_system).fillna(0)

            # Sort by the true seasonal order, not raceId order, before calculating cumulative points
            sim_df['race_order'] = sim_df['raceId'].map(race_order_map)
            sim_df = sim_df.sort_values(['race_order', 'sim_position'])
            sim_df['cumulative_points'] = sim_df.groupby('driverId')['sim_points'].cumsum()

            for _, row in sim_df[['raceId', 'driverId', 'cumulative_points']].iterrows():
                key = (row['driverId'], row['raceId'])
                cumulative_point_sums[key] += float(row['cumulative_points'])
                cumulative_point_counts[key] += 1

            # SEASON STANDINGS
            standings = sim_df.groupby("driverId")["sim_points"].sum()
            champion = int(standings.idxmax())
            champion_counts[champion] = champion_counts.get(champion, 0) + 1

        # WIN PROBABILITIES
        win_probabilities = {
            int(driver): count / n_simulations
            for driver, count in champion_counts.items()
        }

        max_prob = max(win_probabilities.values())
        csi = 1 - max_prob

        # DRIVER NAME MAPPING
        driver_map = (
            df_year[["driverId", "driver_name"]]
            .drop_duplicates()
            .set_index("driverId")["driver_name"]
            .to_dict()
        )

        champion_counts_named = {
            driver_map.get(driver, str(driver)): int(count)
            for driver, count in champion_counts.items()
        }

        win_probabilities_named = {
            driver_map.get(driver, str(driver)): float(prob)
            for driver, prob in win_probabilities.items()
        }

        win_probabilities_named = dict(
            sorted(win_probabilities_named.items(), key=lambda x: x[1], reverse=True)
        )

        champion_counts_named = dict(
            sorted(champion_counts_named.items(), key=lambda x: x[1], reverse=True)
        )

        cumulative_points = []
        for driver_id, driver_name in driver_map.items():
            points_by_race = []
            for race_id in sorted(race_labels.keys(), key=lambda rid: race_labels[rid]['order']):
                key = (driver_id, race_id)
                total = cumulative_point_sums.get(key, 0.0)
                count = cumulative_point_counts.get(key, 0)
                avg_points = float(round(total / max(count, 1), 2))
                points_by_race.append({
                    "race_id": race_id,
                    "race_order": race_labels[race_id]['order'],
                    "race_label": race_labels[race_id]['label'],
                    "cumulative_points": avg_points
                })

            cumulative_points.append({
                "driver_id": int(driver_id),
                "driver_name": driver_name,
                "points_by_race": points_by_race
            })

        race_labels_sorted = [
            race_labels[rid]['label']
            for rid in sorted(race_labels.keys(), key=lambda rid: race_labels[rid]['order'])
        ]

        return {
            "year": int(year),
            "n_simulations": int(n_simulations),
            "win_probabilities": win_probabilities_named,
            "champion_counts": champion_counts_named,
            "csi": float(csi),
            "race_labels": race_labels_sorted,
            "cumulative_points": cumulative_points
        }

    def simulate_configs(self, year, configs, n_simulations=100):
        """
        Dummy counterfactual simulation for UI testing
        """
        results = []

        for config in configs:
            results.append({
                "race": config.get("race"),
                "driver": config.get("driver"),
                "dnf": float(config.get("dnf", 0)),
                "avg_points": float(np.round(np.random.uniform(5.0, 22.0), 1)),
                "win_chance": float(np.round(np.random.uniform(0.05, 0.65), 3))
            })

        return {
            "year": int(year) if year is not None else None,
            "n_simulations": int(n_simulations),
            "total_races": len(results),
            "race_results": results,
            "note": "Dummy simulation output for integration testing."
        }