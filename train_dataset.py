import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

from statistics import calculate_mae, calculate_nmae, calculate_mean_squared_error
import joblib


def train_dataset(iterations_filename: str, output_filename: str):
    # Step 1: Load and Prepare the Dataset
    df = pd.read_csv(iterations_filename)
    # filter out columns that are not features
    df = df.drop(columns=['boardid', 'sprintid', 'name', 'planday'])

    # Step 2: Split the Dataset into Features and Target Variable
    target_variable_column = 'vel_diff'
    X = df.drop(target_variable_column, axis=1)  # Features
    y = df[target_variable_column]  # Target variable

    # Step 3: Split the Dataset into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Step 4: Define and Train the Gradient Boosting Regressor
    n_estimators = 100
    param_grid = {
        'n_estimators': [n_estimators],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    grid_search = GridSearchCV(GradientBoostingRegressor(), param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best model: {best_model}")

    best_parameters = grid_search.best_params_
    print(f"Best parameters: {best_parameters}")

    best_score = grid_search.best_score_
    print(f"Best score: {best_score}")

    # Step 5: Evaluate the Model
    predictions = best_model.predict(X_test)
    print(f"Predictions:\n"
          f"Min: {min(predictions)}\n"
          f"Max: {max(predictions)}\n"
          f"Mean: {predictions.mean()}\n"
          f"Median: {pd.Series(predictions).median()}\n"
          f"Std: {predictions.std()}\n"
          )
    mse = calculate_mean_squared_error(y_test, predictions)
    mae = calculate_mae(y_test, predictions)
    nmae = calculate_nmae(y_test, predictions)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Normalized Mean Absolute Error (NMAE): {nmae}')

    # Step 6: Store the Model
    best_model_filename = f"{output_filename}.pkl"
    joblib.dump(best_model, best_model_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model using the given dataset')
    parser.add_argument('--iterations', type=str, help='Iterations dataset filename')
    parser.add_argument('--output', type=str, help='Output dataset filename without extension',
                        default='output')
    args = parser.parse_args()
    print(f"Reading iterations from: {args.iterations}")
    train_dataset(args.iterations, args.output)
    print(f"Model trained and stored in {args.output}")
