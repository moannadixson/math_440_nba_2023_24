MATH 440 NBA MODELING STARTER PACKAGE (2023-24)

FILES
- shot_chart_data_2023_24.py: downloads 2023-24 shot-chart rows and creates a player-level summary.
- play_by_play_data_2023_24.py: downloads 2023-24 play-by-play events.
- nba_modeling_2023_24.py: creates player-level features and trains all six required regression models.

INSTALL
Open a terminal in this folder and run:
    python -m pip install nba_api pandas numpy scikit-learn

RUN IN THIS ORDER
1. Download shot charts:
    python shot_chart_data_2023_24.py
2. Download play-by-play (this can take a while because it requests games individually):
    python play_by_play_data_2023_24.py
3. Create the target-label template:
    python nba_modeling_2023_24.py
   The first run creates player_targets_2023_24.csv and stops.
4. Fill the numeric target columns in player_targets_2023_24.csv with verified data, save it, and run:
    python nba_modeling_2023_24.py
5. Find the comparison output at:
    model_results_2023_24/model_comparison_results.csv

TARGET COLUMNS
height_inches, age, salary_usd, wingspan_inches, fg_pct, ft_pct, pts_per_game,
ast_per_game, reb_per_game, off_rating, usage_pct, true_shooting_pct,
effective_fg_pct, offensive_box_plus_minus

Use one consistent definition and season for each target. Example: age should be
age on a stated date during the 2023-24 season; per-game stats and advanced stats
should be 2023-24 regular-season values. Leave unavailable/unverified values blank.
Salary and wingspan usually require a separate reliable data source; this package
does not invent or automatically retrieve them.

IMPORTANT LIMITATIONS
- All six required algorithms (LinearRegression, DecisionTreeRegressor, LinearSVR,
  KNeighborsRegressor, MLPRegressor, GaussianProcessRegressor) are regressors.
  Position and All-Star status are categories, so these models do not directly
  solve those classification tasks. You would need classifier versions for that.
- Shot-chart predictors intentionally exclude made/missed shot flags and FG% so
  FG% is not simply copied into the input features. The shot-chart model uses
  attempt locations and shot-zone proportions.
- Play-by-play features count events for the event's primary actor (personId).
  This is not a complete record of every player involved in every event.
- This is a one-season, player-level sample. Report the number of labeled players
  for each target. Small samples can make complex models unstable.
- Results are test-set estimates from one fixed random split, not proof of
  general performance. Compare models using the same split and discuss limitations.
