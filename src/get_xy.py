def get_X_y(df):
    df = df.drop(columns=["boardid", "sprintid", "name", "planday"])
    target_variable_column = "vel_diff"
    X = df.drop(target_variable_column, axis=1)  # Features
    y = df[target_variable_column]  # Target variable
    return X, y
