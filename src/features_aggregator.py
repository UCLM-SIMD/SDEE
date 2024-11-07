import pandas as pd
import argparse

FIELDS = {
    "type": str,  # Issue type
    "priority": str,  # Issue priority
    "no_comment": int,  # The number of comments
    "no_affectversion": int,  # The number of versions for which an issue has been found
    "no_fixversion": int,  # The number of versions for which an issue was or will be ﬁxed
    "no_issuelink": int,  # The number of dependencies of an issues
    "no_blocking": int,  # The number of issues that block this issue for being resolved
    "no_blockedby": int,  # The number of issues that are blocked by this issue
    "no_fixversion_change": int,  # The number of times in which a ﬁx version was changed
    "no_priority_change": int,  # The number of times an issue’s priority was changed
    "no_des_change": int,  # The number of times in which an issue description was changed
    # The read ability index (Gunning Fog [21]) indicates the complexity level
    "gunning_fog": str,
    # of a description which is encoded to easy and hard
}

AGGREGATIONS = {
    "min": lambda col: col.agg("min"),
    "max": lambda col: col.agg("max"),
    "mean": lambda col: col.agg("mean"),
    "median": lambda col: col.agg("median"),
    "std": lambda col: col.agg("std"),
    "var": lambda col: col.agg("var"),
    "range": lambda col: col.max() - col.min(),
}

ISSUE_COLS = 14
ITERATION_COLS = 19


def aggregate_features(
    iterations_filename: str, issues_filename: str, output_iterations_filename: str
):
    df_iterations = pd.read_csv(
        iterations_filename,
        dtype={field: FIELDS[field] for field in FIELDS},
        encoding="latin1",
        delimiter=",",
        on_bad_lines=lambda bad_line: handle_bad_line(
            bad_line, cols=ITERATION_COLS),
        engine="python",
    )
    df_issues = pd.read_csv(
        issues_filename,
        encoding="latin1",
        delimiter=",",
        on_bad_lines=lambda bad_line: handle_bad_line(
            bad_line, cols=ISSUE_COLS),
        engine="python",
    )

    print(f"Nº Iterations: {df_iterations.shape[0]}")
    print(f"Nº Issues: {df_issues.shape[0]}")

    fields_to_aggregate = list(FIELDS.keys())
    print("Aggregating features. Progress:[", end="")

    # Iterate through each row in the first dataset
    for index, row in df_iterations.iterrows():
        if index % 10 == 0:
            print("#", end="")
        board_id = row["boardid"]
        sprint_id = row["sprintid"]

        # Filter rows in the second dataset that have the same iteration and board id
        iteration_issues = df_issues[
            (df_issues["boardid"] == board_id) & (
                df_issues["sprintid"] == sprint_id)
        ]

        # Perform aggregations:
        for field in fields_to_aggregate:
            for agg_key, agg_fun in AGGREGATIONS.items():
                if FIELDS[field] == int:
                    df_iterations.at[index, f"{field}_{agg_key}"] = agg_fun(
                        iteration_issues[field]
                    )
                elif FIELDS[field] == str:
                    # for each different value of the field, count the number of occurrences (frequency)
                    counts = iteration_issues[field].value_counts()
                    for value in counts.index:
                        df_iterations.at[index,
                                         f"{field}_freq_{value}"] = counts[value]

    # Fill NaN values with 0 for categorical features
    df_iterations.fillna(0, inplace=True)

    print("]")
    # store dataset
    df_iterations.to_csv(output_iterations_filename, index=False)


def handle_bad_line(bad_line, cols):
    if len(bad_line) > cols:
        # Join the problematic parts and replace commas with semicolons
        text_column = ",".join(bad_line[2: len(bad_line) - (cols - 3)]).replace(
            ",", ""
        )

        fixed_row = bad_line[:2] + [text_column] + bad_line[-(cols - 3):]
        return fixed_row
    return bad_line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adds statistical features to the iterations of a dataset"
    )
    parser.add_argument("--iterations", type=str,
                        help="Iterations dataset filename")
    parser.add_argument("--issues", type=str, help="Issues dataset filename")
    parser.add_argument(
        "--output_iterations",
        type=str,
        help="Output iterations filename",
        default="output.csv",
    )
    args = parser.parse_args()
    print(
        f"Reading iterations from: {args.iterations} and issues from: {args.issues}")
    aggregate_features(args.iterations, args.issues, args.output_iterations)
    print(
        f"Features were successfully aggregated and stored in: {args.output_iterations}"
    )
