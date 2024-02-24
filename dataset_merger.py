import pandas as pd
import json
import argparse


def merge_datasets(iterations_filename: str, issues_filename: str, output_filename: str):
    df_iterations = pd.read_csv(iterations_filename)
    df_issues = pd.read_csv(issues_filename)

    json_dataset = {
        "iterations": df_iterations.to_dict(orient='records')
    }

    for iteration in json_dataset['iterations']:
        iteration['issues'] = df_issues[df_issues['sprintid'] == iteration['sprintid']].to_dict(orient='records')

    # store the dataset in a json file
    with open(output_filename, 'w') as outfile:
        json.dump(json_dataset['iterations'], outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merges an iterations and an issues dataset into a single json file')
    parser.add_argument('--iterations', type=str, help='Iterations dataset filename')
    parser.add_argument('--issues', type=str, help='Issues dataset filename')
    parser.add_argument('--output', type=str, help='Output dataset filename', default='output.json')
    args = parser.parse_args()
    print(f"Reading iterations from: {args.iterations} and issues from: {args.issues}")
    merge_datasets(args.iterations, args.issues, args.output)
    print(f"Datasets were successfully merged and stored in: {args.output}")
