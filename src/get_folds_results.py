from collections import defaultdict
import time

import numpy as np
from src.sdee_statistics import calculate_mae, calculate_nmae


def get_fold_results(pipeline, cv, X, y, config_text):
    """
    The pipeline has to have a "feature_selection" step
    returns: {
                "config": "a config with params",
                "mae_avg": 1.0,
                "nmae_avg": 1.0,
                "time(s)": 1.0,
                "folds": [
                    {
                        "mae": 4.369074879645989,
                        "nmae": 0.24272638220255496,
                        "features": ["vel_starttime"]
                    }
                ],
                "features_frequency": {
                    "vel_starttime": 1
                }
            }
    """
    result = []
    nmae_values = []
    mae_values = []
    feature_counts = defaultdict(int)
    start_time = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train_fold, y_train_fold)
        y_pred = pipeline.predict(X_test_fold)
        mae = calculate_mae(y_test_fold, y_pred)
        mae_values.append(mae)
        nmae = calculate_nmae(y_test_fold, y_pred)
        nmae_values.append(nmae)

        feature_selector = pipeline.named_steps["feature_selection"]
        selected_feature_indices = np.array(feature_selector.get_support())
        feature_list = []
        for index, column_name in enumerate(X.columns):
            if selected_feature_indices[index]:
                feature_list.append(column_name)
                feature_counts[column_name] += 1

        result.append({"mae": mae, "nmae": nmae, "features": feature_list})

    execution_time = time.time() - start_time

    execution_result = {
        "config": config_text,
        "mae_avg": np.mean(mae_values),
        "nmae_avg": np.mean(nmae_values),
        "time(s)": execution_time,
        "folds": result,
        "features_frequency": feature_counts,
    }
    return execution_result
