import pandas as pd
import numpy as np

class MonteCarloSimulator:
    def __init__(self, df):
        """
        df: master_driver_race dataframe
        """
        self.df = df.copy()

        self.dnf_prob = 0.1  # tweak this val later if needed
        # scoring/points system for race e.g., first place = 25 pts
        self.points_system = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}
        
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

        # Separate finishers and DNFs
        finishers = df_race[df_race['is_dnf'] == False].copy()
        dnfs = df_race[df_race['is_dnf'] == True].copy()

        # Re-rank finishers
        finishers = finishers.sort_values('positionOrder')
        finishers['positionOrder'] = range(1, len(finishers) + 1)

        # points/scorring mapping
        points_map = {
            1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
            6: 8, 7: 6, 8: 4, 9: 2, 10: 1
        }

        finishers['points'] = finishers['positionOrder'].map(points_map).fillna(0)

        # DNF = 0pts
        dnfs['points'] = 0

        df_updated = pd.concat([finishers, dnfs])

        return df_updated

    def simulate_season(self, year, n_simulations=100, df_override=None, configs=None):
        # simulates the f1 season using monte carlo. 
        # inputs: year, n_simulations, df_override (optional pandas dataframe with the user's counterfactual modifications), configs (list of counterfactual configs)
        # returns dict with win probability, champion counts, CSI

        if configs is not None:
            return self.simulate_configs(year, configs, n_simulations)

        import numpy as np
        import pandas as pd

        if df_override is not None:
            df_year = df_override.copy()
        else:
            df_year = self.df[self.df['year'] == year].copy()

        champion_counts = {}

        for _ in range(n_simulations):
            sim_df = df_year.copy()

            # simulate & randomly assigned the dnfs 
            dnf_mask = np.random.rand(len(sim_df)) < self.dnf_prob
            sim_df.loc[dnf_mask, 'is_dnf'] = True

            # adjust positions for dnfs. dnfs go to the bottom rankings
            sim_df['sim_position'] = sim_df['positionOrder']
            sim_df.loc[sim_df['is_dnf'] == True, 'sim_position'] = 999

            # Re-rank for each race
            sim_df['sim_position'] = sim_df.groupby(['raceId'])['sim_position'] \
                .rank(method='first')
            # Re-calculate points based on re-rank
            sim_df['sim_points'] = sim_df['sim_position'].map(self.points_system).fillna(0)

            # get the season standings
            standings = sim_df.groupby('driverId')['sim_points'].sum()

            champion = int(standings.idxmax())

            champion_counts[champion] = champion_counts.get(champion, 0) + 1

        # GET WIN PROBABILITIES 
        win_probabilities = {
            int(driver): count / n_simulations
            for driver, count in champion_counts.items()
        }

        # GET CSI
        max_prob = max(win_probabilities.values())
        csi = 1 - max_prob

        # Map the driver id to the driver name
        driver_map = (
            df_year[['driverId', 'driver_name']]
            .drop_duplicates()
            .set_index('driverId')['driver_name']
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

        # Sort by win prob desc
        win_probabilities_named = dict(
            sorted(win_probabilities_named.items(), key=lambda x: x[1], reverse=True)
        )

        champion_counts_named = dict(
            sorted(champion_counts_named.items(), key=lambda x: x[1], reverse=True)
        )

        # OUTPUT
        return {
            "year": int(year),
            "n_simulations": int(n_simulations),
            "win_probabilities": win_probabilities_named,
            "champion_counts": champion_counts_named,
            "csi": float(csi)
        }

    def simulate_configs(self, year, configs, n_simulations=100):
        """
        Dummy counterfactual simulation for the selected race configurations.
        This returns randomly generated values for testing the submit and render path.
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