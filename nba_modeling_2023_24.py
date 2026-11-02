"""Train the six MATH 440 regression models on 2023-24 NBA data.

Expected folders/files (created by the two extraction scripts):
  nba_2023_24_shot_charts/all_shots_2023_24.csv
  nba_2023_24_play_by_play/all_events_2023_24.csv

Also create/fill in:
  player_targets_2023_24.csv

Run:
  pip install pandas numpy scikit-learn
  python nba_modeling_2023_24.py

This script builds player-level features from shot charts and play-by-play,
then trains all six required REGRESSION models separately for each numeric
trait with available labels. It does not treat position or All-Star status as
continuous numbers; those require classifiers, not the six regressors listed
in the project prompt.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR


SEASON = "2023-24"

SEASON_FILE = SEASON.replace("-", "_")

SHOT_FILE = Path("nba_2023_24_shot_charts") / f"all_shots_{SEASON_FILE}.csv"
PBP_FILE = Path("nba_2023_24_play_by_play") / f"all_events_{SEASON_FILE}.csv"
TARGET_FILE = Path(f"player_targets_{SEASON_FILE}.csv")

OUTPUT_DIR = Path("model_results_2023_24")
RANDOM_STATE = 42
TEST_SIZE = 0.25
MIN_PLAYERS = 20

# These are candidate numeric outcomes. Leave a cell blank if you do not have
# a trustworthy value for that player. Do not guess missing salary/wingspan.
TARGET_COLUMNS = [
    "height_inches",
    "age",
    "salary_usd",
    "wingspan_inches",
    "fg_pct",
    "ft_pct",
    "pts_per_game",
    "ast_per_game",
    "reb_per_game",
    "off_rating",
    "usage_pct",
    "true_shooting_pct",
    "effective_fg_pct",
    "offensive_box_plus_minus",
]


def clean_name(value):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")


def normalize_id(series):
    """Convert NBA IDs to nullable integer IDs, avoiding float merge issues."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def make_shot_features(shots):
    """Make one row per player using shot LOCATION/ATTEMPT features only.

    Made/missed flags are deliberately not used as predictors. That avoids
    directly giving a model the answer when FG% is one of the target traits.
    """
    if "PLAYER_ID" not in shots.columns:
        raise ValueError("Shot chart file needs a PLAYER_ID column.")

    shots = shots.copy()
    shots["PLAYER_ID"] = normalize_id(shots["PLAYER_ID"])
    shots = shots.dropna(subset=["PLAYER_ID"])

    agg = {}
    for col in ["SHOT_DISTANCE", "LOC_X", "LOC_Y"]:
        if col in shots.columns:
            agg[col] = "mean"
    if "SHOT_DISTANCE" in shots.columns:
        agg["SHOT_DISTANCE"] = ["mean", "std", "min", "max"]

    # Build the core summary without using shot-make information.
    summary = shots.groupby("PLAYER_ID").size().rename("shot_attempts").to_frame()
    for col, out_name in [
        ("SHOT_DISTANCE", "avg_shot_distance"),
        ("LOC_X", "avg_shot_x"),
        ("LOC_Y", "avg_shot_y"),
    ]:
        if col in shots.columns:
            summary[out_name] = shots.groupby("PLAYER_ID")[col].mean()
    if "SHOT_DISTANCE" in shots.columns:
        summary["shot_distance_sd"] = shots.groupby("PLAYER_ID")["SHOT_DISTANCE"].std()

    # Attempt proportions by NBA shot-zone category.
    for col, prefix in [
        ("SHOT_ZONE_BASIC", "zone"),
        ("SHOT_TYPE", "shot_type"),
        ("SHOT_ZONE_AREA", "area"),
    ]:
        if col in shots.columns:
            counts = pd.crosstab(shots["PLAYER_ID"], shots[col], normalize="index")
            counts.columns = [
                f"{prefix}_{clean_name(c).lower()}_rate" for c in counts.columns
            ]
            summary = summary.join(counts, how="left")

    if "PLAYER_NAME" in shots.columns:
        names = shots.groupby("PLAYER_ID")["PLAYER_NAME"].first()
        summary["PLAYER_NAME"] = names
    summary = summary.replace([np.inf, -np.inf], np.nan).reset_index()
    return summary


def make_pbp_features(events):
    """Create event-actor counts/rates from the V3 personId field.

    Important limitation: personId identifies the event's primary actor. It
    does not identify every player involved in every event, so these are
    actor-event features, not complete player participation statistics.
    """
    if "personId" not in events.columns:
        raise ValueError(
            "Play-by-play file has no personId column. Inspect its columns and "
            "map the player-ID field before running this script."
        )
    events = events.copy()
    events["PLAYER_ID"] = normalize_id(events["personId"])
    events = events.dropna(subset=["PLAYER_ID"])
    events = events[events["PLAYER_ID"] != 0]
    if events.empty:
        raise ValueError("No play-by-play events with a usable personId were found.")

    features = events.groupby("PLAYER_ID").size().rename("pbp_actor_events").to_frame()
    if "playerName" in events.columns:
        names = events.groupby("PLAYER_ID")["playerName"].first()
        features["PLAYER_NAME"] = names
    for col, prefix in [("actionType", "action"), ("subType", "subtype")]:
        if col in events.columns:
            counts = pd.crosstab(
                events["PLAYER_ID"],
                events[col].fillna("unknown")
            )

            counts.columns = [
                f"{prefix}_{clean_name(c).lower()}_count"
                for c in counts.columns
            ]

            # Combine duplicate columns if different categories
            # produce the same cleaned name.
            counts = counts.T.groupby(level=0).sum().T

            features = features.join(counts, how="left")
    # Include event proportions as well as raw counts.
    count_cols = [c for c in features.columns if c.endswith("_count")]
    for col in count_cols:
        features[col.replace("_count", "_rate")] = (
            features[col] / features["pbp_actor_events"].replace(0, np.nan)
        )
    return features.replace([np.inf, -np.inf], np.nan).reset_index()


def create_target_template(shot_features, pbp_features):
    """Create a blank target-label CSV once; user fills trusted target values."""
    id_frames = []
    for frame in [shot_features, pbp_features]:
        cols = ["PLAYER_ID"] + (["PLAYER_NAME"] if "PLAYER_NAME" in frame.columns else [])
        id_frames.append(frame[cols])
    ids = pd.concat(id_frames, ignore_index=True).drop_duplicates("PLAYER_ID")
    targets = ids.copy()
    for col in TARGET_COLUMNS:
        targets[col] = np.nan
    targets["position"] = ""       # categorical; not used by these regressors
    targets["all_star_2024"] = ""  # categorical 0/1 label; not used by these regressors
    targets.to_csv(TARGET_FILE, index=False)
    print(f"Created blank target template: {TARGET_FILE}")
    print("Fill the numeric target columns with verified 2023-24 player values, save, and rerun.")


def build_models(n_train):
    """Return the six models required by the project prompt."""
    # KNN needs a smaller neighbor count when the training set is small.
    neighbors = max(1, min(5, n_train - 1))
    return {
        "LinearRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]),
        "DecisionTreeRegressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, min_samples_leaf=3)),
        ]),
        "LinearSVR": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearSVR(C=1.0, max_iter=20000, random_state=RANDOM_STATE)),
        ]),
        "KNeighborsRegressor": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(n_neighbors=neighbors, weights="distance")),
        
        ])
    }


def evaluate_dataset(feature_df, targets, source_name):
    merged = feature_df.merge(targets, on="PLAYER_ID", how="inner")
    feature_cols = [
        c for c in feature_df.columns
        if c != "PLAYER_ID" and pd.api.types.is_numeric_dtype(feature_df[c])
    ]
    if not feature_cols:
        print(f"No numeric features found for {source_name}; skipping.")
        return []

    results = []
    for target in TARGET_COLUMNS:
        if target not in merged.columns:
            continue
        usable = merged.dropna(subset=[target]).copy()
        usable[target] = pd.to_numeric(usable[target], errors="coerce")
        usable = usable.dropna(subset=[target])
        if len(usable) < MIN_PLAYERS:
            print(f"Skipping {source_name} -> {target}: only {len(usable)} labeled players.")
            continue

        X = usable[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = usable[target].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        models = build_models(len(X_train))
        for model_name, model in models.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                import joblib

                model_dir = OUTPUT_DIR / "saved_models" / source_name / target
                model_dir.mkdir(parents=True, exist_ok=True)

                model_file = model_dir / f"{model_name}.joblib"
                joblib.dump(model, model_file)

                # Save actual and predicted values for later graphs.
                prediction_dir = OUTPUT_DIR / "predictions" / source_name / target
                prediction_dir.mkdir(parents=True, exist_ok=True)

                prediction_df = pd.DataFrame({
                    "PLAYER_ID": usable.loc[X_test.index, "PLAYER_ID"].values,
                    "actual": y_test.values,
                    "predicted": predictions
                })

                if "PLAYER_NAME" in usable.columns:
                    prediction_df["PLAYER_NAME"] = (
                        usable.loc[X_test.index, "PLAYER_NAME"].values
                    )

                prediction_df.to_csv(
                    prediction_dir / f"{model_name}_predictions.csv",
                    index=False
                )

                # Save the nearest neighbors for the KNN model.
                if model_name == "KNeighborsRegressor":
                    knn = model.named_steps["model"]
                    first_test = X_test.iloc[[0]]
                    first_test_scaled = model[:-1].transform(first_test)
                    distances, indices = knn.kneighbors(first_test_scaled)

                    training_rows = usable.loc[X_train.index]
                    nearest_rows = training_rows.iloc[indices[0]]
                    neighbors_df = nearest_rows[["PLAYER_ID", target]].copy()
                    if "PLAYER_NAME" in training_rows.columns:
                        neighbors_df["PLAYER_NAME"] = nearest_rows["PLAYER_NAME"].values

                    neighbors_df["distance"] = distances[0]
                    neighbors_df["test_player_actual"] = y_test.iloc[0]
                    neighbors_df["test_player_prediction"] = predictions[0]
                    neighbors_df.to_csv(
                        prediction_dir / "knn_nearest_neighbors.csv",
                        index=False
                    )

                results.append({
                    "data_source": source_name,
                    "target": target,
                    "model": model_name,
                    "n_players": len(usable),
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "MSE": mean_squared_error(y_test, predictions),
                    "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
                    "MAE": mean_absolute_error(y_test, predictions),
                    "R2": r2_score(y_test, predictions) if len(y_test) > 1 else np.nan,
                })
                print(
                    f"{source_name:10s} | {target:24s} | {model_name:25s} "
                    f"MSE={results[-1]['MSE']:.4f}  R2={results[-1]['R2']:.4f}"
                )
            except Exception as exc:
                print(f"Model failed: {source_name}, {target}, {model_name}: {exc}")
    return results


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    if not SHOT_FILE.exists():
        raise FileNotFoundError(f"Missing {SHOT_FILE}. Run shot_chart_data_2023_24.py first.")
    if not PBP_FILE.exists():
        raise FileNotFoundError(f"Missing {PBP_FILE}. Run play_by_play_data_2023_24.py first.")

    print("Loading shot-chart and play-by-play files...")
    shots = pd.read_csv(SHOT_FILE)
    events = pd.read_csv(PBP_FILE, low_memory=False)
    shot_features = make_shot_features(shots)
    pbp_features = make_pbp_features(events)

    if not TARGET_FILE.exists():
        create_target_template(shot_features, pbp_features)
        return

    targets = pd.read_csv(TARGET_FILE)
    if "PLAYER_ID" not in targets.columns:
        raise ValueError(f"{TARGET_FILE} must contain PLAYER_ID as its first column.")
    targets["PLAYER_ID"] = normalize_id(targets["PLAYER_ID"])
    for col in TARGET_COLUMNS:
        if col in targets.columns:
            targets[col] = pd.to_numeric(targets[col], errors="coerce")

    print("\nTraining models on SHOT-CHART features...")
    results = evaluate_dataset(shot_features, targets, "shot_chart")
    print("\nTraining models on PLAY-BY-PLAY features...")
    results += evaluate_dataset(pbp_features, targets, "play_by_play")

    if not results:
        print("No models were evaluated. Check that target columns contain enough numeric labels.")
        return

    results_df = pd.DataFrame(results).sort_values(["target", "data_source", "MSE"])
    output_file = OUTPUT_DIR / "model_comparison_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved full model comparison to {output_file}")
    print("\nBest model per data source and target (lowest test MSE):")
    best = results_df.loc[results_df.groupby(["data_source", "target"])["MSE"].idxmin()]
    print(best[["data_source", "target", "model", "n_players", "MSE", "RMSE", "MAE", "R2"]].to_string(index=False))

    print("\nReminder: position and All-Star status are categorical outcomes. The six required models are regressors;")
    print("do not report regression on category codes as a valid classification result.")


if __name__ == "__main__":
    main()
