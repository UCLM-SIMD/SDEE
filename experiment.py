import argparse
from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.custom_kfold import CustomKFold
from src.sdee_statistics import calculate_nmae
from src.train_dataset import get_X_y
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector


def run_experiment(tol=None, direction="forward"):
    # load dataset heads:
    dataset_iteration_30 = pd.read_csv("datasets/apache_iteration_30_features.csv")

    # setup model:
    X, y = get_X_y(dataset_iteration_30)

    # train model
    random_forest = RandomForestRegressor(
        n_estimators=500, max_depth=7, random_state=42
    )

    nmae_scorer = make_scorer(calculate_nmae, greater_is_better=False)

    # print(f"Sequential Forward Search:")
    sfs_pipeline = Pipeline(
        [
            (
                "feature_selection",
                SequentialFeatureSelector(
                    random_forest,
                    direction=direction,
                    n_features_to_select="auto",  # 0.2,
                    tol=tol,
                    cv=CustomKFold(n_splits=2, shuffle=False),
                    n_jobs=-1,
                    scoring=nmae_scorer,
                ),
            ),
            ("regressor", random_forest),
        ]
    )

    start_time = time.time()
    # nmae_values = cross_val_score(
    #    sfs_pipeline,
    #    X,
    #    y,
    #    cv=CustomKFold(n_splits=10, shuffle=False),
    #    scoring=nmae_scorer,
    # )

    result = []
    nmae_values = []
    feature_counts = defaultdict(int)
    for fold_idx, (train_idx, test_idx) in enumerate(
        CustomKFold(n_splits=10, shuffle=False).split(X)
    ):
        result_iteration = {}
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        sfs_pipeline.fit(X_train_fold, y_train_fold)
        score = sfs_pipeline.score(X_test_fold, y_test_fold)
        nmae_values.append(score)

        feature_selector = sfs_pipeline.named_steps["feature_selection"]

        selected_feature_indices = np.array(feature_selector.get_support())

        result_iteration["nmae"] = score
        for index, column_name in enumerate(X.columns):
            if index in selected_feature_indices:
                feature_counts[column_name] += 1
        result_iteration["features"] = feature_counts

        result.append(str(result_iteration))

    execution_time = time.time() - start_time

    execution_result = {
        "config": f"tol={tol} direction={direction}",
        "nmae_avg": np.mean(nmae_values),
        "time(s)": execution_time,
        "folds": result,
    }
    print(json.dumps(execution_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the experiment with specific parameters."
    )
    parser.add_argument("tol", type=float, help="Tolerance value for the experiment")
    parser.add_argument("direction", type=str, help="Direction forward or backward")

    # Parse arguments
    args = parser.parse_args()
    tol = args.tol
    direction = args.direction
    # print(f"starting SFS with tol={tol} + direction={direction}")
    run_experiment(tol, direction)
    # print("finished")
