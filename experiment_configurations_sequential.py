if __name__ == "__main__":
    datasets = [
        "apache_iteration_30_features",
        "jboss_iteration_30_features",
        "jira_iteration_30_features",
        "mongodb_iteration_30_features",
        "spring_iteration_30_features",
    ]
    tols = [1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.01, 0.001]
    directions = ["forward", "backward"]
    n_features_to_select = [0.2, "auto"]

    filename = "configs.txt"
    print(
        f"Generating {len(datasets)*len(tols)*len(directions)*len(n_features_to_select)} configs in {filename}"
    )
    with open(filename, "w") as f:
        for dataset in datasets:
            for direct in directions:
                for tol in tols:
                    for n_features in n_features_to_select:
                        f.write(
                            f"{dataset} {tol:.10f} {direct} {n_features}\n")

    print("finished")
