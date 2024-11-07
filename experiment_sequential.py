import argparse
from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from src.custom_kfold import CustomKFold
from src.get_folds_results import get_fold_results
from src.sdee_statistics import calculate_mae, calculate_nmae
from src.train_dataset import get_X_y
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector


def parse_value(value):
    try:
        if value == "None":
            return None
        # Try to convert to float first
        float_value = float(value)
        # Check if it can be represented as an integer
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except ValueError:
        # Return as string if it's neither int nor float
        return value


def run_experiment(dataset, tol=None, direction="forward", n_features_to_select="auto"):
    dataset_iteration_30 = pd.read_csv(f"datasets/{dataset}.csv")
    X, y = get_X_y(dataset_iteration_30)

    mae_scorer = make_scorer(calculate_mae, greater_is_better=False)
    cv = CustomKFold(n_splits=10, shuffle=False)

    random_forest = RandomForestRegressor(
        n_estimators=500, max_depth=7, random_state=42
    )

    sfs_pipeline = Pipeline(
        [
            (
                "feature_selection",
                SequentialFeatureSelector(
                    random_forest,
                    direction=direction,
                    n_features_to_select=n_features_to_select,
                    tol=tol,
                    cv=CustomKFold(n_splits=2, shuffle=False),
                    n_jobs=-1,
                    scoring=mae_scorer,
                ),
            ),
            ("regressor", random_forest),
        ]
    )

    execution_result = get_fold_results(
        sfs_pipeline,
        cv=cv,
        X=X,
        y=y,
        config_text=f"tol={tol} direction={direction} n_features_to_select={n_features_to_select}",
        dataset=dataset
    )
    print(json.dumps(execution_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the experiment with specific parameters."
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("tol", help="Tolerance value for the experiment")
    parser.add_argument("direction", type=str,
                        help="Direction forward or backward")
    parser.add_argument("n_features_to_select", help="n_features_to_select")

    args = parser.parse_args()
    dataset = args.dataset
    tol = parse_value(args.tol)
    direction = args.direction
    n_features_to_select = parse_value(args.n_features_to_select)

    run_experiment(dataset, tol, direction, n_features_to_select)
