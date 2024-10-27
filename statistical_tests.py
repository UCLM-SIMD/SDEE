import json
import argparse
from scipy.stats import wilcoxon, friedmanchisquare
import numpy as np
from statsmodels.stats.multitest import multipletests
from src.VD_A import VD_A

from scipy.stats import rankdata


def apply_holm_bonferroni(p_values, alpha=0.05):
    """
    Apply the Holm-Bonferroni correction to the given p-values.
    Returns a list of adjusted p-values and significant results.
    """
    # Sort p-values and keep track of their original indices
    sorted_indices = np.argsort(p_values)
    sorted_p_values = p_values[sorted_indices]
    m = len(p_values)
    adjusted_p_values = np.zeros(m)

    # Apply Holm-Bonferroni adjustment
    for i, p in enumerate(sorted_p_values):
        adjusted_p_values[i] = p * (m / (i + 1))

    # Check for significance
    significant_results = []
    for i in range(m):
        if adjusted_p_values[i] <= alpha:
            significant_results.append((sorted_indices[i], adjusted_p_values[i]))

    return adjusted_p_values, significant_results


def load_results(filename, dataset):
    with open(filename, "r") as file:
        configs = json.load(file)
        filtered_configs = [
            config for config in configs if config["dataset"] == dataset]
        return filtered_configs


def friedman_test(data):
    """
    Perform the Friedman test to compare MAE values across all configurations in the dataset.
    """
    mae_values_all_configs = [[fold["mae"] for fold in config["folds"]] for config in data]
    stat, p_value = friedmanchisquare(*mae_values_all_configs)
    return stat, p_value


def is_statistically_different(config, config_without_fss):
    """
    Perform the Wilcoxon test between two configurations.
    """
    mae_values1 = [fold["mae"] for fold in config["folds"]]
    mae_values2 = [fold["mae"] for fold in config_without_fss["folds"]]
    stat, p_value = wilcoxon(mae_values1, mae_values2, alternative='two-sided')
    return p_value < 0.05, p_value


def compute_a12(config, config_without_fss):
    """
    Compute Vargha and Delaney's Â12 effect size between two configurations.
    """
    mae_values1 = [fold["mae"] for fold in config["folds"]]
    mae_values2 = [fold["mae"] for fold in config_without_fss["folds"]]
    a12_stat, _ = VD_A(mae_values1, mae_values2)
    return a12_stat


def main(filename, dataset):
    data = load_results(filename, dataset)
    print(f"Dataset: {dataset}")

    # Find the full feature set configuration
    config_without_fss = [config for config in data if "100%" in config["config"]][0]

    data = sorted(
        data, key=lambda x: int(x["config"].split("%")[0].split(" ")[1]), reverse=True
    )

    # First, perform the Friedman test on the dataset
    stat, p_value = friedman_test(data)
    print(f"Friedman Test - Statistic: {stat}, p-value: {p_value:.4f}")

    if p_value < 0.05:
        print("Statistically significant differences found. Proceeding with pairwise Wilcoxon tests.")
        # Loop through each configuration and compare against the full feature set (config_without_fss)
        wilcoxon_p_values = []
        config_names = []
        for config in data:
            if config["config"] == config_without_fss["config"]:
                print(f"Config: {config['config']}, Mean MAE: {config['mae_avg']:.4f}")
                continue

            # Perform Wilcoxon test
            different, wilcoxon_p_value = is_statistically_different(config, config_without_fss)

            wilcoxon_p_values.append(wilcoxon_p_value)
            config_names.append(config['config'])

            # Compute A12 statistic
            a12_stat = compute_a12(config, config_without_fss)

            print(
                f"Config: {config['config']}, Mean MAE: {config['mae_avg']:.4f} "
                f"(p={wilcoxon_p_value:.4f}, A12={a12_stat:.3f})"
            )

        # Apply Holm-Bonferroni correction
        adjusted_p_values = multipletests(wilcoxon_p_values, method='holm')[1]

        print("\nHolm-Bonferroni Adjusted p-values:")
        for config_name, adjusted_p in zip(config_names, adjusted_p_values):
            print(f"Config: {config_name}, Adjusted p-value: {adjusted_p:.4f}")

        print("\nSignificant Results after Holm-Bonferroni Adjustment:")
        for config_name, adjusted_p in zip(config_names, adjusted_p_values):
            if adjusted_p < 0.05:
                print(f"Config: {config_name}, Significant (Adjusted p-value: {adjusted_p:.4f})")


    else:
        print("No statistically significant differences found with the Friedman test.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform Friedman test, Wilcoxon tests, and compute A12")
    parser.add_argument("filename", type=str, help="file name")
    args = parser.parse_args()
    filename = args.filename
    for dataset in [
        "apache_iteration_30_features",
        "jboss_iteration_30_features",
        "jira_iteration_30_features",
        "mongodb_iteration_30_features",
        "spring_iteration_30_features",
    ]:
        main(filename, dataset)
        print("==========================================")
        print("\n\n")
