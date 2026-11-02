"""
Download NBA 2023-24 play-by-play and organize events by game and player.

Install:
    pip install nba_api pandas

Run:
    python play_by_play_data_2023_24.py

Outputs:
    nba_2023_24_play_by_play/all_events_2023_24.csv
    nba_2023_24_play_by_play/by_game/<game_id>.csv
    nba_2023_24_play_by_play/by_player/<player_id>_<player_name>.csv

By default this downloads REGULAR-SEASON games. Change SEASON_TYPE to
"Playoffs" for postseason games.
"""

import re
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

SEASON = "2023-24"
SEASON_TYPE = "Regular Season"  # or "Playoffs"
OUTPUT_DIR = Path("nba_2023_24_play_by_play")
GAME_DIR = OUTPUT_DIR / "by_game"
PLAYER_DIR = OUTPUT_DIR / "by_player"
REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 4


def safe_filename(value):
    value = re.sub(r'[<>:"/\\\\|?*]', "", str(value))
    return re.sub(r"\s+", "_", value.strip())


def get_season_game_ids():
    """Get the season's game IDs once, using team rows to avoid player duplicates."""
    print(f"Finding {SEASON} {SEASON_TYPE.lower()} game IDs...")
    finder = leaguegamefinder.LeagueGameFinder(
        player_or_team_abbreviation="T",
        season_nullable=SEASON,
        season_type_nullable=SEASON_TYPE,
        league_id_nullable="00",
        timeout=90,
    )
    games = finder.get_data_frames()[0]

    if games.empty or "GAME_ID" not in games.columns:
        raise RuntimeError("The NBA API returned no game IDs.")

    # LeagueGameFinder returns one row per team, so drop duplicate game IDs.
    return sorted(games["GAME_ID"].astype(str).drop_duplicates().tolist())


def download_one_game(game_id):
    """Fetch one game's V3 play-by-play, retrying temporary API failures."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = playbyplayv3.PlayByPlayV3(
                game_id=game_id,
                start_period=1,
                end_period=10,  # includes overtime periods
                timeout=90,
            )
            frames = response.get_data_frames()
            if not frames:
                raise RuntimeError("No play-by-play data frame was returned.")
            events = frames[0].copy()
            if "gameId" not in events.columns:
                events["gameId"] = game_id
            return events
        except Exception as exc:
            last_error = exc
            wait = attempt * 2
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed for {game_id}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    print(f"  Skipping game {game_id}; last error: {last_error}")
    return pd.DataFrame()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GAME_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_DIR.mkdir(parents=True, exist_ok=True)

    game_ids = get_season_game_ids()
    print(f"Found {len(game_ids)} games.")

    all_game_events = []
    for number, game_id in enumerate(game_ids, start=1):
        print(f"[{number}/{len(game_ids)}] Downloading game {game_id}...")
        events = download_one_game(game_id)

        if not events.empty:
            events["gameId"] = events["gameId"].astype(str)
            events.to_csv(GAME_DIR / f"{game_id}.csv", index=False)
            all_game_events.append(events)

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_game_events:
        raise RuntimeError(
            "No play-by-play was downloaded. The NBA Stats API may be blocking "
            "requests; try again later."
        )

    all_events = pd.concat(all_game_events, ignore_index=True)
    all_file = OUTPUT_DIR / "all_events_2023_24.csv"
    all_events.to_csv(all_file, index=False)
    print(f"Saved {len(all_events):,} events to {all_file}")

    # Each event is assigned to the player recorded as the event's personId.
    # This is the event's primary actor, not every player mentioned in its text.
    if "personId" in all_events.columns:
        player_events = all_events[
            all_events["personId"].notna() & (all_events["personId"] != 0)
        ].copy()

        name_col = "playerName" if "playerName" in player_events.columns else None
        if name_col:
            player_events["_player_name"] = player_events[name_col].fillna("Unknown")
        else:
            player_events["_player_name"] = "Unknown"

        for (player_id, player_name), rows in player_events.groupby(
            ["personId", "_player_name"], dropna=False
        ):
            filename = f"{player_id}_{safe_filename(player_name)}.csv"
            rows.drop(columns=["_player_name"]).to_csv(
                PLAYER_DIR / filename, index=False
            )

        print(
            f"Saved event-actor files for "
            f"{player_events['personId'].nunique():,} players in {PLAYER_DIR}"
        )
    else:
        print("Warning: personId was not present; player files were not created.")

    print(f"Saved per-game files in {GAME_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
