import time
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.custom_kfold import CustomKFold
from src.sdee_statistics import calculate_nmae
from src.train_dataset import get_X_y
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import mutual_info_regression


def run_experiment():
    # load dataset heads:
    dataset_iteration_30 = pd.read_csv("datasets/apache_iteration_30_features.csv")
    dataset_iteration_30.head(1)

    # setup model:
    X, y = get_X_y(dataset_iteration_30)

    # train model
    random_forest = RandomForestRegressor(
        n_estimators=500, max_depth=7, random_state=42
    )

    nmae_scorer = make_scorer(calculate_nmae, greater_is_better=False)

    print(f"Sequential Forward Search:")
    sfs_pipeline = Pipeline(
        [
            (
                "feature_selection",
                SequentialFeatureSelector(
                    random_forest,
                    n_features_to_select=0.2,
                    cv=CustomKFold(n_splits=2, shuffle=False),
                    n_jobs=-1,
                    scoring=nmae_scorer,
                ),
            ),
            ("regressor", random_forest),
        ]
    )

    start_time = time.time()
    nmae_values = cross_val_score(
        sfs_pipeline,
        X,
        y,
        cv=CustomKFold(n_splits=10, shuffle=False),
        scoring=nmae_scorer,
    )
    execution_time = time.time() - start_time

    print(f"Time(s):{execution_time}\n")
    print(f"NMAE:{nmae_values}\n")


if __name__ == "__main__":
    print("starting")
    run_experiment()
    print("finished")
