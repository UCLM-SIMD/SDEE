import argparse
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.custom_kfold import CustomKFold
from src.get_folds_results import get_fold_results
from src.train_dataset import get_X_y
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile


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


def run_experiment(dataset, percentile):
    dataset_iteration_30 = pd.read_csv(
        f"datasets/{dataset}.csv")
    X, y = get_X_y(dataset_iteration_30)

    cv = CustomKFold(n_splits=10, shuffle=False)

    random_forest = RandomForestRegressor(
        n_estimators=500, max_depth=7, random_state=42
    )

    def custom_scorer(X, y):
        random_forest.fit(X, y)
        return random_forest.feature_importances_

    sfs_pipeline = Pipeline(
        [
            (
                "feature_selection",
                SelectPercentile(percentile=percentile,
                                 score_func=custom_scorer),
            ),
            ("regressor", random_forest),
        ]
    )

    execution_result = get_fold_results(
        sfs_pipeline,
        cv=cv,
        X=X,
        y=y,
        config_text=f"SelectPercentile {percentile}%",
        dataset=dataset
    )
    print(json.dumps(execution_result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select Percentile based of RF feature importance. Run the experiment with specific parameters."
    )
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("percentile", help="Percentile of features to select")

    args = parser.parse_args()
    dataset = args.dataset
    percentile = parse_value(args.percentile)

    run_experiment(dataset, percentile)
