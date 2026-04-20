"""
build_master_dataset.py
Team 75 - CSE 6242 F1 Championship Analysis

Builds two clean datasets from raw Ergast CSV files:
  1. master_driver_race.csv     - one row per driver per race (26k rows)
  2. season_championship_summary.csv - final standings per driver per season

Usage:
    python build_master_dataset.py --data_dir ./data/raw --out_dir ./data/processed

Raw files required (all from Ergast F1 Kaggle dataset):
    races.csv, results.csv, drivers.csv, constructors.csv,
    driver_standings.csv, status.csv, pit_stops.csv
"""

import argparse
import os
import pandas as pd


# ---------------------------------------------------------------------------
# DNF classification
# ---------------------------------------------------------------------------

NON_DNF_PATTERNS = [
    "Finished",
    "Did not qualify",
    "Did not prequalify",
    "Disqualified",
    "Not classified",
    "Withdrew",
]


def classify_dnf(status_df: pd.DataFrame) -> set:
    """Return the set of statusIds that represent a true in-race DNF."""
    is_lapped = status_df["status"].str.match(r"^\+\d+ Lap")
    is_non_dnf = status_df["status"].isin(NON_DNF_PATTERNS)
    dnf_ids = status_df.loc[~is_lapped & ~is_non_dnf, "statusId"]
    return set(dnf_ids)


# ---------------------------------------------------------------------------
# Finishing time imputation
# ---------------------------------------------------------------------------

def impute_finishing_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute null finishing_time_seconds for classified finishers only.
    Strategy: max finishing time in that race + 0.5% gap per position delta.
    DNFs remain null since they're already blank.
    """
    df = df.copy()

    def impute_race(group):
        max_time = group["finishing_time_seconds"].max()
        if pd.isna(max_time):
            return group  # no times recorded for this race, skip

        last_known_pos = group.loc[group["finishing_time_seconds"].notna(), "positionOrder"].max()

        def estimate_time(row):
            pos_delta = row["positionOrder"] - last_known_pos
            return max_time * (1 + 0.001 * pos_delta)

        mask = group["finishing_time_seconds"].isna() & ~group["is_dnf"]
        group.loc[mask, "finishing_time_seconds"] = group.loc[mask].apply(estimate_time, axis=1)
        return group

    return df.groupby("raceId", group_keys=False).apply(impute_race)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_master(data_dir: str, out_dir: str) -> None:
    print("Loading raw CSVs...")
    races = pd.read_csv(os.path.join(data_dir, "races.csv"))
    results = pd.read_csv(os.path.join(data_dir, "results.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
    constructors = pd.read_csv(os.path.join(data_dir, "constructors.csv"))
    driver_standings = pd.read_csv(os.path.join(data_dir, "driver_standings.csv"))
    status = pd.read_csv(os.path.join(data_dir, "status.csv"))
    pit_stops = pd.read_csv(os.path.join(data_dir, "pit_stops.csv"))

    # -- Slim down columns before merging -----------------------------------
    races = races[["raceId", "year", "round", "name", "date"]].rename(
        columns={"name": "race_name"}
    )
    drivers = drivers[
        ["driverId", "driverRef", "forename", "surname", "nationality"]
    ]
    constructors = constructors[
        ["constructorId", "constructorRef", "name"]
    ].rename(columns={"name": "constructor_name", "constructorRef": "constructor_ref"})

    driver_standings = driver_standings.rename(
        columns={
            "points": "cumulative_points",
            "position": "championship_position",
            "positionText": "championship_position_text",
            "wins": "cumulative_wins",
        }
    )[[
        "raceId", "driverId", "cumulative_points",
        "championship_position", "championship_position_text", "cumulative_wins",
    ]]

    # -- DNF flag -----------------------------------------------------------
    dnf_status_ids = classify_dnf(status)

    # -- Build master -------------------------------------------------------
    print("Joining tables...")
    df = (
        results
        .merge(races, on="raceId")
        .merge(drivers, on="driverId")
        .merge(constructors, on="constructorId")
        .merge(status, on="statusId")
        .merge(driver_standings, on=["raceId", "driverId"], how="left")
    )

    df["driver_name"] = df["forename"] + " " + df["surname"]
    df["is_dnf"] = df["statusId"].isin(dnf_status_ids)

    # Replace Ergast null sentinel \N with proper NaN
    sentinel_cols = [
        "position", "milliseconds", "fastestLap",
        "rank", "fastestLapTime", "fastestLapSpeed",
    ]
    df[sentinel_cols] = df[sentinel_cols].replace("\\N", pd.NA)

    # Convert milliseconds to seconds
    df["finishing_time_seconds"] = pd.to_numeric(df["milliseconds"], errors="coerce") / 1000

    # -- Pit stop aggregation -----------------------------------------------
    pit_agg = pit_stops.groupby(["raceId", "driverId"]).agg(
        total_pit_stops=("stop", "count"),
        total_pit_stop_duration_sec=("duration", lambda x: pd.to_numeric(x, errors="coerce").sum())
    ).reset_index()

    df = df.merge(pit_agg, on=["raceId", "driverId"], how="left")

    # -- Select and order final columns ------------------------------------
    master = df[[
        "year", "round", "raceId", "race_name", "date",
        "driverId", "driverRef", "driver_name", "nationality",
        "constructorId", "constructor_ref", "constructor_name",
        "grid", "positionOrder", "positionText", "points",
        "laps", "finishing_time_seconds", "total_pit_stops",
        "total_pit_stop_duration_sec", "statusId", "status", "is_dnf",
        "championship_position", "championship_position_text",
        "cumulative_points", "cumulative_wins",
        "fastestLapTime", "fastestLapSpeed",
    ]].sort_values(["year", "round", "positionOrder"]).reset_index(drop=True)

    # -- Impute missing finishing times -------------------------------------
    master = impute_finishing_times(master)

    # -- Season championship summary ---------------------------------------
    season_summary = (
        master
        .sort_values("round")
        .groupby(["year", "driverId"])
        .last()
        .reset_index()
    )[[
        "year", "driverId", "driverRef", "driver_name", "nationality",
        "constructorId", "constructor_name",
        "championship_position", "cumulative_points", "cumulative_wins",
    ]].sort_values(["year", "championship_position"]).reset_index(drop=True)

    # -- Save ---------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    master.to_csv(os.path.join(out_dir, "master_driver_race.csv"), index=False)
    season_summary.to_csv(os.path.join(out_dir, "season_championship_summary.csv"), index=False)

    # -- Report -------------------------------------------------------------
    print(f"\n✓ master_driver_race.csv")
    print(f"  Rows: {len(master):,}  |  Columns: {len(master.columns)}")
    print(f"  Years: {master['year'].min()} – {master['year'].max()}")
    print(f"  Unique drivers: {master['driverId'].nunique()}")
    print(f"  Unique races:   {master['raceId'].nunique()}")
    print(f"  DNF rate:       {master['is_dnf'].mean()*100:.1f}%")

    print(f"\n✓ season_championship_summary.csv")
    print(f"  Rows: {len(season_summary):,}  |  Columns: {len(season_summary.columns)}")

    dnf_by_era = master.copy()
    dnf_by_era["era"] = pd.cut(
        dnf_by_era["year"],
        bins=[1949, 1979, 1999, 2009, 2024],
        labels=["1950-1979", "1980-1999", "2000-2009", "2010-2024"],
    )
    era_stats = dnf_by_era.groupby("era")["is_dnf"].mean().mul(100).round(1)
    print(f"\nDNF rate by era (useful for Monte Carlo calibration):")
    for era, rate in era_stats.items():
        print(f"  {era}: {rate}%")

    print(f"\nOutput saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build F1 master datasets.")
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--out_dir", default="./data/processed")
    args = parser.parse_args()
    build_master(args.data_dir, args.out_dir)