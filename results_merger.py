import argparse
import glob
import json
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge result files")
    parser.add_argument("file_prefix", type=str, help="file prefix")
    args = parser.parse_args()
    file_prefix = args.file_prefix

    outputs_directory = os.path.join(os.getcwd(), f"{file_prefix}/outputs")
    pattern = os.path.join(outputs_directory, "*")
    files = glob.glob(pattern)

    merged_json_list = []

    print(f"Merging {len(files)} configs...")
    for file_path in files:
        with open(file_path, "r") as file:
            file_content = file.read()
            if not file_content.strip():
                print("File empty. Skipping")
                continue
            content = json.loads(file_content)
            print(f"{file_path}")
            merged_json_list.append(content)

    output_file_path = os.path.join(outputs_directory, "merged_output.json")
    with open(output_file_path, "w") as output_file:
        json.dump(merged_json_list, output_file, indent=4)

    print("finished")
