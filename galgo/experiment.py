import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from custom_kfold import CustomKFold
from sdee_statistics import calculate_nmae
from train_dataset import get_X_y, train_model_cv
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import (
    SelectPercentile,
    f_regression,
    mutual_info_regression,
)
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

    custom_kfold = CustomKFold(n_splits=10, shuffle=False)

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
                    cv=custom_kfold,
                    n_jobs=-1,
                    scoring=nmae_scorer,
                ),
            ),
            ("regressor", random_forest),
        ]
    )

    start_time = time.time()
    nmae_values = cross_val_score(
        sfs_pipeline, X, y, cv=custom_kfold, scoring=nmae_scorer
    )
    execution_time = time.time() - start_time

    with open("results_sdee.txt", "w") as file:
        file.write(f"Time(s):{execution_time}\n")
        file.write(f"NMAE:{nmae_values}\n")


if __name__ == "__main__":
    run_experiment()
