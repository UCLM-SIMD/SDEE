if __name__ == "__main__":
    print(f"Generate configs")
    tols = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    directions = ["forward", "backward"]
    with open("configs.txt", "w") as f:
        for direct in directions:
            for tol in tols:
                f.write(f"{tol:.10f} {direct}\n")

    print("finished")
