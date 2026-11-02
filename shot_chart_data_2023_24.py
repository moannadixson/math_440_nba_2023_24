"""
Download NBA 2023-24 shot-chart attempts and organize them by player.

Install:
    pip install nba_api pandas

Run:
    python shot_chart_data_2023_24.py

Outputs:
    nba_2023_24_shot_charts/all_shots_2023_24.csv
    nba_2023_24_shot_charts/player_shots/<player_id>_<player_name>.csv
    nba_2023_24_shot_charts/player_shot_summary_2023_24.csv

By default this downloads REGULAR-SEASON shots. Change SEASON_TYPE to
"Playoffs" if you want postseason shots instead.
"""

import re
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import shotchartdetail

SEASON = "2023-24"
SEASON_TYPE = "Regular Season"  # or "Playoffs"
OUTPUT_DIR = Path("nba_2023_24_shot_charts")
PLAYER_DIR = OUTPUT_DIR / "player_shots"


def safe_filename(value):
    """Remove characters that are awkward in Windows filenames."""
    value = re.sub(r'[<>:"/\\\\|?*]', "", str(value))
    return re.sub(r"\s+", "_", value.strip())


def download_all_shots():
    print(f"Requesting {SEASON} {SEASON_TYPE.lower()} shot-chart data...")

    # PlayerID=0 and TeamID=0 request league-wide shot data.
    # ContextMeasure='FGA' includes both made and missed field-goal attempts.
    response = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        season_nullable=SEASON,
        season_type_all_star=SEASON_TYPE,
        context_measure_simple="FGA",
        league_id="00",
        timeout=90,
    )

    frames = response.get_data_frames()
    if not frames or frames[0].empty:
        raise RuntimeError(
            "The NBA API returned no shot rows. Check the season settings, "
            "connection, and whether the NBA Stats endpoint is responding."
        )

    shots = frames[0].copy()
    return shots


def make_player_summary(shots):
    """Create one row per player with basic shot-chart features."""
    required = {"PLAYER_ID", "PLAYER_NAME", "SHOT_MADE_FLAG", "SHOT_DISTANCE"}
    missing = required.difference(shots.columns)
    if missing:
        raise ValueError(f"Shot data is missing expected columns: {sorted(missing)}")

    summary = (
        shots.groupby(["PLAYER_ID", "PLAYER_NAME"], dropna=False)
        .agg(
            FGA=("SHOT_ATTEMPTED_FLAG", "sum"),
            FGM=("SHOT_MADE_FLAG", "sum"),
            AVG_SHOT_DISTANCE=("SHOT_DISTANCE", "mean"),
            AVG_LOC_X=("LOC_X", "mean"),
            AVG_LOC_Y=("LOC_Y", "mean"),
        )
        .reset_index()
    )
    summary["FG_PCT"] = summary["FGM"] / summary["FGA"].replace(0, pd.NA)

    # Useful zone counts/rates for later feature engineering.
    if "SHOT_ZONE_BASIC" in shots.columns:
        zone_counts = pd.crosstab(
            [shots["PLAYER_ID"], shots["PLAYER_NAME"]],
            shots["SHOT_ZONE_BASIC"],
        ).reset_index()
        zone_counts.columns = [
            str(col).upper().replace(" ", "_") if isinstance(col, str) else col
            for col in zone_counts.columns
        ]
        summary = summary.merge(
            zone_counts, on=["PLAYER_ID", "PLAYER_NAME"], how="left"
        )

    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)

    shots = download_all_shots()

    # Keep IDs as strings in saved CSVs so leading zeros in GAME_ID are preserved.
    if "GAME_ID" in shots.columns:
        shots["GAME_ID"] = shots["GAME_ID"].astype(str)

    all_file = OUTPUT_DIR / "all_shots_2023_24.csv"
    shots.to_csv(all_file, index=False)
    print(f"Saved {len(shots):,} shot attempts to {all_file}")

    # One CSV per player. Each row is an individual field-goal attempt.
    for (player_id, player_name), player_shots in shots.groupby(
        ["PLAYER_ID", "PLAYER_NAME"], dropna=False
    ):
        filename = f"{player_id}_{safe_filename(player_name)}.csv"
        player_shots.to_csv(PLAYER_DIR / filename, index=False)

    summary = make_player_summary(shots)
    summary_file = OUTPUT_DIR / "player_shot_summary_2023_24.csv"
    summary.to_csv(summary_file, index=False)

    print(f"Saved {summary['PLAYER_ID'].nunique():,} player files in {PLAYER_DIR}")
    print(f"Saved player-level summary to {summary_file}")
    print("Done.")


if __name__ == "__main__":
    main()
