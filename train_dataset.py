import argparse
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from custom_kfold import CustomKFold
from statistics import calculate_mae, calculate_nmae, calculate_mean_squared_error
import matplotlib.pyplot as plt
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def find_best(iterations_filename: str):
    # Load and Prepare the Dataset
    df = pd.read_csv(iterations_filename)
    # filter out columns that are not features
    df = df.drop(columns=['boardid', 'sprintid', 'name', 'planday'])

    # Split the Dataset into Features and Target Variable
    target_variable_column = 'vel_diff'
    X = df.drop(target_variable_column, axis=1)  # Features
    y = df[target_variable_column]  # Target variable

    # Define and Train the Gradient Boosting Regressor
    folds = CustomKFold(n_splits=10, shuffle=False)
    nmae_scorer = make_scorer(calculate_nmae, greater_is_better=False)

    # models to try:
    gbm = GridSearchCV(GradientBoostingRegressor(), param_grid={
        'n_estimators': [100],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
    }, cv=folds, scoring=nmae_scorer)
    _find_best(gbm, X, y)

    random_forest = GridSearchCV(RandomForestRegressor(), param_grid={
        'n_estimators': [500],
        'max_depth': [3, 5, 7],
    }, cv=folds, scoring=nmae_scorer)
    _find_best(random_forest, X, y)

    # train_dnn(X, iterations_filename, y)


def train_dnn(X, iterations_filename, y):
    # DNN with dropout
    # Assuming you have your dataset and labels
    print(f"Training DNN with dropout for {iterations_filename}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Build a neural network with dropout
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    # Train the model
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32,
              validation_split=0.2)  # Evaluate the model on the test set
    y_pred = model.predict(X_test_scaled)
    nmae = calculate_nmae(y_test, y_pred)
    print(f'Normalized Mean Absolute Error (NMAE): {nmae}')


def _find_best(grid_search: GridSearchCV, X, y):
    print(f"Finding best model for {grid_search.estimator.__class__.__name__}")
    # Split the Dataset into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

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
    return best_model


def train_best(iterations_filename: str, output_filename: str):
    # Step 1: Load and Prepare the Dataset
    df = pd.read_csv(iterations_filename)
    # filter out columns that are not features
    df = df.drop(columns=['boardid', 'sprintid', 'name', 'planday'])

    # Step 2: Split the Dataset into Features and Target Variable
    target_variable_column = 'vel_diff'
    X = df.drop(target_variable_column, axis=1)  # Features
    y = df[target_variable_column]  # Target variable

    # Assuming your data is in X and y
    n_folds = 10
    kf = CustomKFold(n_splits=n_folds, shuffle=False)

    nmae_values = []
    models = [GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5),
              RandomForestRegressor(n_estimators=500, max_depth=7)]
    for train_index, test_index in kf.split(X):
        for model in models:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Initialize and train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate NMAE for this fold
            nmae = calculate_nmae(y_test, y_pred)
            nmae_values.append(nmae)
    print(f"avg nmae: {sum(nmae_values) / len(nmae_values)}")
    plt.boxplot(nmae_values)
    plt.savefig(f"{output_filename}_boxplot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model using the given dataset')
    parser.add_argument('--iterations', type=str, help='Iterations dataset filename')
    parser.add_argument('--output', type=str, help='Output dataset filename without extension',
                        default='output')
    args = parser.parse_args()
    print(f"Reading iterations from: {args.iterations}")
    # find_best(args.iterations)
    train_best(args.iterations, args.output)
    print(f"Model trained and stored in {args.output}")
