from matplotlib import pyplot as plt


def plot_feature_frequency_per_fold(config_result: dict):
    frequencies = config_result["features_frequency"]
    sorted_features = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:]
    features, importances = zip(*sorted_features)

    plt.figure(figsize=(10, 10))
    plt.barh(features, importances, color="skyblue")
    plt.xlabel("Frequency")
    plt.title(
        f"Feature Frequency for {config_result['config']} and MAE={config_result['mae_avg']}"
    )
    plt.gca().invert_yaxis()
    plt.show()
    return plt
