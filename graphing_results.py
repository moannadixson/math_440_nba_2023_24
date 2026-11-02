
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import plot_tree


# --------------------------------------------------
# Folders
# --------------------------------------------------

RESULTS_DIR = Path("model_results_2023_24")
MODEL_DIR = RESULTS_DIR / "saved_models"
PREDICTION_DIR = RESULTS_DIR / "predictions"
GRAPH_DIR = RESULTS_DIR / "visualizations"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# General prediction graph
# --------------------------------------------------

def plot_actual_vs_predicted(predictions, model_name, source, target, output_dir):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predictions["actual"],
        predictions["predicted"]
    )

    # Perfect predictions would fall on this diagonal.
    minimum = min(
        predictions["actual"].min(),
        predictions["predicted"].min()
    )

    maximum = max(
        predictions["actual"].max(),
        predictions["predicted"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{model_name}: {target}\n{source}")

    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png")
    plt.close()


# --------------------------------------------------
# Decision Tree visualization
# --------------------------------------------------

def plot_decision_tree(model, feature_names, source, target, output_dir):

    tree = model.named_steps["model"]

    plt.figure(figsize=(24, 12))

    plot_tree(
        tree,
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        max_depth=3,
        precision=2,
        fontsize=8
    )

    plt.title(
        f"Decision Tree Regressor\n{target} - {source}"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "decision_tree.png", dpi=200)
    plt.close()


# --------------------------------------------------
# Linear Regression visualization
# --------------------------------------------------

def plot_linear_regression(model, feature_names, source, target, output_dir):

    regression = model.named_steps["model"]

    coefficients = pd.Series(
        regression.coef_,
        index=feature_names
    ).sort_values()

    plt.figure(figsize=(10, 7))

    coefficients.plot(kind="barh")

    plt.xlabel("Coefficient")
    plt.ylabel("Input Feature")
    plt.title(
        f"Linear Regression Coefficients\n{target} - {source}"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "linear_regression_coefficients.png")
    plt.close()


# --------------------------------------------------
# Linear SVR visualization
# --------------------------------------------------

def plot_linear_svr(model, feature_names, source, target, output_dir):

    svr = model.named_steps["model"]

    # These coefficients correspond to standardized input features.
    coefficients = pd.Series(
        svr.coef_.ravel(),
        index=feature_names
    ).sort_values()

    plt.figure(figsize=(10, 7))

    coefficients.plot(kind="barh")

    plt.xlabel("Coefficient (standardized feature scale)")
    plt.ylabel("Input Feature")
    plt.title(
        f"Linear SVR Coefficients\n{target} - {source}"
    )

    plt.tight_layout()
    plt.savefig(output_dir / "linear_svr_coefficients.png")
    plt.close()


# --------------------------------------------------
# KNN visualization
# --------------------------------------------------

def plot_knn_neighbors(neighbor_file, save_folder, target):

    neighbors = pd.read_csv(neighbor_file)

    # Identify the target column.
    metadata_columns = {
        "PLAYER_ID",
        "PLAYER_NAME",
        "distance",
        "test_player_actual",
        "test_player_prediction"
    }

    target_columns = [
        column for column in neighbors.columns
        if column not in metadata_columns
    ]

    if not target_columns:
        print(f"Could not find target column in {neighbor_file}")
        return

    target_column = (
        target if target in neighbors.columns
        else target_columns[0]
    )

    # Label each neighboring player.
    if "PLAYER_NAME" in neighbors.columns:
        labels = neighbors["PLAYER_NAME"].astype(str).tolist()
    elif "PLAYER_ID" in neighbors.columns:
        labels = neighbors["PLAYER_ID"].astype(str).tolist()
    else:
        labels = [
            f"Neighbor {i + 1}"
            for i in range(len(neighbors))
        ]

    neighbor_values = neighbors[target_column]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, neighbor_values)

    # Show the KNN prediction.
    if "test_player_prediction" in neighbors.columns:
        prediction = neighbors["test_player_prediction"].iloc[0]

        plt.axhline(
            prediction,
            linestyle="--",
            label=f"KNN prediction: {prediction:.2f}"
        )

    # Show the test player's actual value.
    if "test_player_actual" in neighbors.columns:
        actual = neighbors["test_player_actual"].iloc[0]

        plt.axhline(
            actual,
            linestyle=":",
            label=f"Actual value: {actual:.2f}"
        )

    plt.xlabel("Nearest training players")
    plt.ylabel(f"Actual {target}")
    plt.title(f"KNN: Nearest Neighbors for {target}")
    plt.xticks(rotation=35, ha="right")

    # Display each neighbor's distance.
    if "distance" in neighbors.columns:
        for index, (value, distance) in enumerate(
            zip(neighbor_values, neighbors["distance"])
        ):
            plt.annotate(
                f"d={distance:.3f}",
                (index, value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=8
            )

    plt.legend()
    plt.tight_layout()

    save_folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        save_folder / "KNeighborsRegressor_nearest_neighbors.png",
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Run visualizations for every saved model
# --------------------------------------------------

def main():

    model_files = list(MODEL_DIR.glob("*/*/*.joblib"))

    if not model_files:
        print("No saved models found. Run nba_modeling_2023_24.py first.")
        return

    for model_file in model_files:

        source = model_file.parents[1].name
        target = model_file.parent.name
        model_name = model_file.stem

        print(f"Creating graphs: {source} | {target} | {model_name}")

        model = joblib.load(model_file)

        output_dir = GRAPH_DIR / source / target / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        prediction_file = (
            PREDICTION_DIR
            / source
            / target
            / f"{model_name}_predictions.csv"
        )

        if prediction_file.exists():

            predictions = pd.read_csv(prediction_file)

            plot_actual_vs_predicted(
                predictions,
                model_name,
                source,
                target,
                output_dir
            )

        feature_names = model.named_steps["imputer"].feature_names_in_

        if model_name == "DecisionTreeRegressor":

            plot_decision_tree(
                model,
                feature_names,
                source,
                target,
                output_dir
            )

        elif model_name == "LinearRegression":

            plot_linear_regression(
                model,
                feature_names,
                source,
                target,
                output_dir
            )

        elif model_name == "LinearSVR":

            plot_linear_svr(
                model,
                feature_names,
                source,
                target,
                output_dir
            )

        elif model_name == "KNeighborsRegressor":

            neighbors_file = (
                PREDICTION_DIR
                / source
                / target
                / "knn_nearest_neighbors.csv"
            )

            plot_knn_neighbors(
                neighbors_file,
                output_dir,
                target
            )

    print(f"\nVisualizations saved in: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
