if __name__ == "__main__":
    tols = [1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.01, 0.001]
    directions = ["forward", "backward"]
    n_features_to_select = [0.2, "auto"]

    filename = "configs.txt"
    print(
        f"Generating {len(tols)*len(directions)*len(n_features_to_select)} configs in {filename}"
    )
    with open(filename, "w") as f:
        for direct in directions:
            for tol in tols:
                for n_features in n_features_to_select:
                    f.write(f"{tol:.10f} {direct} {n_features}\n")

    print("finished")
