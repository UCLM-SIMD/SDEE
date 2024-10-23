import argparse
import itertools
import json
import numpy as np
from scipy.stats import wilcoxon
import pandas as pd


def load_results(filename, dataset):
    with open(filename, "r") as file:
        configs = json.load(file)
        filtered_configs = [
            config for config in configs if config["dataset"] == dataset]
        return filtered_configs


def is_statistically_different(config, config_without_fss):
    mae_values1 = [fold["mae"] for fold in config["folds"]]
    mae_values2 = [fold["mae"] for fold in config_without_fss["folds"]]
    stat, p_value = wilcoxon(mae_values1, mae_values2, alternative='two-sided')
    return p_value < 0.05
    # for comp in comparisons:
    #     if (comp['config1'] == config["config"] or comp['config2'] == config["config"]):
    #         if comp['p_value'] > 0.05:
    #             return False
    # return True


def main(filename, dataset):
    data = load_results(filename, dataset)
    #mae_dict = extract_mae_values(data)
    # results_df = compare_configs(data)
    # print(results_df)

    # comparisons = compare_configs(data)

    print(f"Dataset: {dataset}")
    config_without_fss = [config for config in data if "100%" in config["config"]][0]
    for config in data:
        if config["config"] == config_without_fss["config"]:
            print(
                f"Config: {config['config']}, Mean MAE: {config['mae_avg']:.4f}")
            continue
        different = is_statistically_different(config, config_without_fss)
        symbol = '*' if different else ''
        print(
            f"Config: {config['config']}, Mean MAE: {config['mae_avg']:.4f} {symbol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="perform wilcoxon tests")
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
