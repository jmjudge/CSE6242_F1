import numpy as np
import pandas as pd
from src.monte_carlo import MonteCarloSimulator
from src.build_counterfactual_engine import CounterfactualEngine

class F1CounterfactualMonteCarlo:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def validate(self, scenarios_df):
        if scenarios_df is None or scenarios_df.empty:
            return

        if len(scenarios_df) > 5:
            raise ValueError("Max 5 changes allowed")

        if scenarios_df['raceId'].duplicated().any():
            raise ValueError("Only 1 change per race allowed")

        valid_types = {'position', 'dnf'}
        if not scenarios_df['change_type'].isin(valid_types).all():
            raise ValueError("Invalid change_type")

    def run(self, year, scenarios_df=None, n_simulations=1000):
        # Step 1: Validate
        self.validate(scenarios_df)

        # Step 2: Apply counterfactuals
        df_year = self.df[self.df['year'] == year].copy()

        if scenarios_df is not None and not scenarios_df.empty:
            cf_engine = CounterfactualEngine(df_year)
            cf_engine.apply_scenarios(scenarios_df)
            df_modified = cf_engine.get_data()
        else:
            df_modified = df_year

        # Step 3: Monte Carlo
        mc = MonteCarloSimulator(self.df)
        results = mc.simulate_season(
            year=year,
            n_simulations=n_simulations,
            df_override=df_modified
        )

        return results
