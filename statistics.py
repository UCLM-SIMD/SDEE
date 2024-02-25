from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats


def calculate_nmae(y_true: list, y_pred: list) -> float:
    mae = calculate_mae(y_true, y_pred)
    iqr = calculate_iqr(y_true)
    nmae = mae / iqr
    return nmae


def calculate_mae(y_test: list, y_pred: list) -> float:
    return mean_absolute_error(y_test, y_pred)


def calculate_iqr(y_true: list) -> float:
    return stats.iqr(y_true, interpolation='midpoint')  # default linear


def calculate_mean_squared_error(y_true: list, y_pred: list) -> float:
    return mean_squared_error(y_true, y_pred)
