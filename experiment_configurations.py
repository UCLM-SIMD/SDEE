if __name__ == "__main__":
    print(f"Generate configs")
    tols = [1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.01, 0.001]
    directions = ["forward", "backward"]
    with open("configs.txt", "w") as f:
        for direct in directions:
            for tol in tols:
                f.write(f"{tol:.10f} {direct}\n")

    print("finished")
