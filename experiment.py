import argparse
from collections import defaultdict
import json
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from src.custom_kfold import CustomKFold
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

    start_time = time.time()

    result = []
    nmae_values = []
    mae_values = []
    feature_counts = defaultdict(int)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        sfs_pipeline.fit(X_train_fold, y_train_fold)
        y_pred = sfs_pipeline.predict(X_test_fold)
        mae = calculate_mae(y_test_fold, y_pred)
        mae_values.append(mae)
        nmae = calculate_nmae(y_test_fold, y_pred)
        nmae_values.append(nmae)

        feature_selector = sfs_pipeline.named_steps["feature_selection"]
        selected_feature_indices = np.array(feature_selector.get_support())
        feature_list = []
        for index, column_name in enumerate(X.columns):
            if selected_feature_indices[index]:
                feature_list.append(column_name)
                feature_counts[column_name] += 1

        result.append({"mae": mae, "nmae": nmae, "features": feature_list})

    execution_time = time.time() - start_time

    execution_result = {
        "config": f"tol={tol} direction={direction} n_features_to_select={n_features_to_select}",
        "dataset": dataset,
        "mae_avg": np.mean(mae_values),
        "nmae_avg": np.mean(nmae_values),
        "time(s)": execution_time,
        "folds": result,
        "features_frequency": feature_counts,
    }
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
