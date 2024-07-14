from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats


def calculate_nmae(y_test: list, y_pred: list) -> float:
    mae = calculate_mae(y_test, y_pred)
    iqr = calculate_iqr(y_test)
    nmae = (
        mae / iqr
    )  # iqr should be the same for all the samples, by taking the iqr of the whole dataset
    return nmae


def calculate_nmae_pre_calculated_iqr(y_test: list, y_pred: list, iqr: float) -> float:
    mae = calculate_mae(y_test, y_pred)
    nmae = mae / iqr
    return nmae


def calculate_mae(y_test: list, y_pred: list) -> float:
    return mean_absolute_error(y_test, y_pred)


def calculate_iqr(y_test: list) -> float:
    return stats.iqr(y_test, interpolation="linear")  # default linear


def calculate_mean_squared_error(y_test: list, y_pred: list) -> float:
    return mean_squared_error(y_test, y_pred)
