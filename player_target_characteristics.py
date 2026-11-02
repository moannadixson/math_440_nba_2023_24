
import pandas as pd
from pathlib import Path

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayerbiostats
)

SEASON = "2023-24"
TARGET_FILE = Path("player_targets_2023_24.csv")


def get_player_stats(measure_type):
    """Download one set of 2023-24 player statistics."""

    response = leaguedashplayerstats.LeagueDashPlayerStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense=measure_type
    )

    return response.get_data_frames()[0]


def main():

    # Load the blank target template created by your modeling script.
    targets = pd.read_csv(TARGET_FILE)

    if "PLAYER_ID" not in targets.columns:
        raise ValueError(
            "The target CSV needs a PLAYER_ID column to match players."
        )

    # Download basic and advanced statistics.
    print("Downloading basic player statistics...")
    basic = get_player_stats("Base")

    print("Downloading advanced player statistics...")
    advanced = get_player_stats("Advanced")

    print("Downloading player bio information...")
    bio_response = leaguedashplayerbiostats.LeagueDashPlayerBioStats(
        season=SEASON,
        season_type_all_star="Regular Season",
        per_mode_simple="PerGame"
    )
    bio = bio_response.get_data_frames()[0]

    # Keep one row per player. Prefer the total-season row
    # for players who played for multiple teams.
    def one_row_per_player(df):
        if "TEAM_ABBREVIATION" in df.columns:
            total_rows = df[df["TEAM_ABBREVIATION"] == "TOT"]
            if not total_rows.empty:
                df = pd.concat([
                    total_rows,
                    df[~df["PLAYER_ID"].isin(total_rows["PLAYER_ID"])]
                ])

        return df.drop_duplicates(subset="PLAYER_ID")

    basic = one_row_per_player(basic)
    advanced = one_row_per_player(advanced)
    bio = one_row_per_player(bio)

    # Select and rename the columns needed for the project.
    basic = basic[[
        "PLAYER_ID", "AGE", "FG_PCT", "FT_PCT",
        "PTS", "AST", "REB"
    ]]

    basic = basic.rename(columns={
        "AGE": "age",
        "FG_PCT": "fg_pct",
        "FT_PCT": "ft_pct",
        "PTS": "pts_per_game",
        "AST": "ast_per_game",
        "REB": "reb_per_game"
    })

    advanced_columns = [
        "PLAYER_ID", "OFF_RATING", "USG_PCT",
        "TS_PCT", "EFG_PCT"
    ]

    advanced = advanced[advanced_columns].rename(columns={
        "OFF_RATING": "off_rating",
        "USG_PCT": "usage_pct",
        "TS_PCT": "true_shooting_pct",
        "EFG_PCT": "effective_fg_pct"
    })

    bio = bio[[
        "PLAYER_ID", "PLAYER_HEIGHT_INCHES"
    ]].rename(columns={
        "PLAYER_HEIGHT_INCHES": "height_inches"
    })

    # Match the downloaded information to the players
    # already listed in the target template.
    targets["PLAYER_ID"] = pd.to_numeric(
        targets["PLAYER_ID"], errors="coerce"
    )

    for df in [basic, advanced, bio]:
        df["PLAYER_ID"] = pd.to_numeric(
            df["PLAYER_ID"], errors="coerce"
        )

    targets = targets.drop(
        columns=[
            "age", "height_inches", "fg_pct", "ft_pct",
            "pts_per_game", "ast_per_game", "reb_per_game",
            "off_rating", "usage_pct", "true_shooting_pct",
            "effective_fg_pct"
        ],
        errors="ignore"
    )

    targets = targets.merge(basic, on="PLAYER_ID", how="left")
    targets = targets.merge(advanced, on="PLAYER_ID", how="left")
    targets = targets.merge(bio, on="PLAYER_ID", how="left")
    # Remove players who have no matching basic statistical data.
    targets = targets.dropna(subset=["age", "pts_per_game"])

    # Remove duplicate player IDs.
    targets = targets.drop_duplicates(subset="PLAYER_ID")

    # These are not supplied by the endpoints used above.
    # Leave them blank until verified values are collected.
    for column in [
        "salary_usd",
        "wingspan_inches",
        "offensive_box_plus_minus"
    ]:
        if column not in targets.columns:
            targets[column] = pd.NA

    targets.to_csv(TARGET_FILE, index=False)

    print(f"\nUpdated target file: {TARGET_FILE}")
    print(f"Players in target file: {len(targets)}")
    print("\nMissing values by target:")
    print(targets.isna().sum())


if __name__ == "__main__":
    main()
