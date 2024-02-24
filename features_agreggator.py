import pandas as pd
import os
import json
import argparse


def aggregate_features(dataset_filename: str, output_filename: str):
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), dataset_filename))

    # store dataset
    df.to_json(output_filename, orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adds statistical features to the iterations of a dataset')
    parser.add_argument('--dataset', type=str, help='Dataset filename')
    parser.add_argument('--output', type=str, help='Output dataset filename', default=None)
    args = parser.parse_args()
    print(f"Reading dataset from: {args.dataset}")
    if args.output is None:
        args.output = os.path.splitext(args.dataset)[0] + "_features.json"
    aggregate_features(args.dataset, args.output)
    print(f"Datasets were successfully merged and stored in: {args.output}")
