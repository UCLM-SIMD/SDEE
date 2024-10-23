if __name__ == "__main__":
    datasets = [
        "apache_iteration_30_features",
        "jboss_iteration_30_features",
        "jira_iteration_30_features",
        "mongodb_iteration_30_features",
        "spring_iteration_30_features",
    ]
    percentiles = [1, 5, 10, 20, 40, 60, 80, 100]

    filename = "configs.txt"
    print(
        f"Generating {len(datasets) * len(percentiles)} configs in {filename}"
    )
    with open(filename, "w") as f:
        for dataset in datasets:
            for percentile in percentiles:
                f.write(f"{dataset} {percentile}\n")

    print("finished")
