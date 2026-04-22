import pandas as pd
import numpy as np

class CounterfactualEngine:
    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()
        self.counterfactual = df.copy()

        # Guardrails
        self.max_changes = 3
        self.max_changes_per_race = 1

    # Validation in scenario inputs

    def validate(self, scenarios_df):
        if scenarios_df is None or len(scenarios_df) == 0:
            return

        if len(scenarios_df) > self.max_changes:
            raise ValueError("Max 3 changes allowed per season")

        if scenarios_df['raceId'].duplicated().any():
            raise ValueError("Only 1 change allowed per race")

        if not scenarios_df['driverId'].notna().all():
            raise ValueError("Each scenario must include a valid driverId")

        dnf_probs = pd.to_numeric(scenarios_df['dnf_prob'], errors='coerce')
        if dnf_probs.isna().any() or not dnf_probs.between(0.0, 1.0).all():
            raise ValueError("dnf_prob must be a number between 0 and 1")

        time_delays = pd.to_numeric(scenarios_df['time_delay'], errors='coerce')
        if time_delays.isna().any() or not time_delays.between(0.0, 5.0).all():
            raise ValueError("time_delay must be a number between 0 and 5 seconds")

        scenarios_df['dnf_prob'] = dnf_probs
        scenarios_df['time_delay'] = time_delays

    # Helper methods

    def _get_race(self, race_id):
        return self.counterfactual[
            self.counterfactual['raceId'] == race_id
        ].copy()

    def _replace_race(self, race_id, race_df):
        self.counterfactual = self.counterfactual[
            self.counterfactual['raceId'] != race_id
        ]
        self.counterfactual = pd.concat([self.counterfactual, race_df])

    # Change types in races

    def _apply_dnf(self, race_df, driver_id, is_dnf):
        race_df = race_df.copy()
        race_df['is_dnf'] = race_df.get('is_dnf', False)

        race_df.loc[
            race_df['driverId'] == driver_id,
            'is_dnf'
        ] = bool(is_dnf)

        return race_df

    def _apply_pit_time(self, race_df, driver_id, delta_seconds):
        race_df = race_df.copy()

        if 'finishing_time_seconds' not in race_df.columns:
            raise ValueError("race_time column required")

        race_df['pit_time_delta'] = race_df.get('pit_time_delta', 0)

        race_df.loc[
            race_df['driverId'] == driver_id,
            'pit_time_delta'
        ] += float(delta_seconds)

        race_df['adjusted_time'] = (
            race_df['finishing_time_seconds'] + race_df['pit_time_delta']
        )

        return race_df

    def _apply_pit_count(self, race_df, driver_id, new_count, avg_stop_time=22):
        race_df = race_df.copy()

        if 'pit_stop_count' not in race_df.columns:
            raise ValueError("pit_stop_count column required")

        race_df['pit_time_delta'] = race_df.get('pit_time_delta', 0)

        original = race_df.loc[
            race_df['driverId'] == driver_id,
            'pit_stop_count'
        ].values[0]

        delta_stops = new_count - original
        avg_pit_time = race_df['total_pit_stop_duration_sec'] / original
        delta_time = delta_stops * avg_pit_time

        race_df.loc[
            race_df['driverId'] == driver_id,
            'pit_time_delta'
        ] += delta_time

        race_df['adjusted_time'] = (
            race_df['race_time'] + race_df['pit_time_delta']
        )

        return race_df

    def _apply_scenario(self, race_df, driver_id, dnf_prob, time_delay):
        race_df = race_df.copy()
        race_df['scenario_dnf_prob'] = race_df.get('scenario_dnf_prob', np.nan)
        race_df['scenario_time_delay'] = race_df.get('scenario_time_delay', np.nan)

        if pd.notna(dnf_prob):
            race_df.loc[
                race_df['driverId'] == driver_id,
                'scenario_dnf_prob'
            ] = float(dnf_prob)

        if pd.notna(time_delay):
            race_df.loc[
                race_df['driverId'] == driver_id,
                'scenario_time_delay'
            ] = float(time_delay)

        return race_df

    # Recompute race standings

    def recompute_race(self, race_df):
        race_df = race_df.copy()
        race_df['is_dnf'] = race_df.get('is_dnf', False)

        # Sort by time if available
        if 'adjusted_time' in race_df.columns:
            race_df = race_df.sort_values('adjusted_time')
        else:
            race_df = race_df.sort_values('positionOrder')

        finishers = race_df[~race_df['is_dnf']].copy()
        dnfs = race_df[race_df['is_dnf']].copy()

        # Assign positions
        finishers['positionOrder'] = range(1, len(finishers) + 1)
        dnfs['positionOrder'] = np.nan

        # Debug: gap to leader
        if 'adjusted_time' in finishers.columns:
            leader_time = finishers['adjusted_time'].min()
            finishers['gap_to_leader'] = finishers['adjusted_time'] - leader_time

        return pd.concat([finishers, dnfs])

    # Apply per-race scenario overrides

    def apply_change(self, row):
        race_id = row['raceId']
        driver_id = row['driverId']
        race_df = self._get_race(race_id)

        race_df = self._apply_scenario(
            race_df,
            driver_id,
            row.get('dnf_prob'),
            row.get('time_delay')
        )

        self._replace_race(race_id, race_df)

    # Applying scenarios based on user input

    def apply_scenarios(self, scenarios_df):
        if scenarios_df is None:
            return

        self.validate(scenarios_df)

        for _, row in scenarios_df.iterrows():
            self.apply_change(row)

    def get_counterfactual_data(self):
        return self.counterfactual.copy()
