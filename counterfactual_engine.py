import pandas as pd
import numpy as np

POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}

class CounterfactualEngine:
    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()
        self.counterfactual = df.copy()

    def apply_change(self, row):
        race_id = row['raceId']
        driver_id = row['driverId']

        race_df = self.counterfactual[self.counterfactual['raceId'] == race_id].copy()

        if row['change_type'] == 'position':
            new_position = int(row['new_position'])

            target = race_df[race_df['driverId'] == driver_id]
            race_df = race_df[race_df['driverId'] != driver_id]

            race_df.loc[race_df['positionOrder'] >= new_position, 'positionOrder'] += 1

            target['positionOrder'] = new_position
            race_df = pd.concat([race_df, target])

        elif row['change_type'] == 'dnf':
            race_df['is_dnf'] = race_df.get('is_dnf', False)
            race_df.loc[race_df['driverId'] == driver_id, 'is_dnf'] = bool(row['is_dnf'])

        # Re-rank after any change
        finishers = race_df[~race_df.get('is_dnf', False)].copy()
        dnfs = race_df[race_df.get('is_dnf', False)].copy()

        finishers = finishers.sort_values('positionOrder')
        finishers['positionOrder'] = range(1, len(finishers) + 1)

        dnfs['positionOrder'] = np.nan

        updated = pd.concat([finishers, dnfs])

        self.counterfactual = self.counterfactual[self.counterfactual['raceId'] != race_id]
        self.counterfactual = pd.concat([self.counterfactual, updated])

    def apply_scenarios(self, scenarios_df):
        for _, row in scenarios_df.iterrows():
            self.apply_change(row)

    def get_data(self):
        return self.counterfactual.copy()
