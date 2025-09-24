# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Last update: Lucia Licakova, 2025-09-08

import json, os
import argparse


def main(split):
    """
    Convert icot text data to JSON format.
    Args:
        split (str): The dataset split (e.g., train, test, valid).
    """
    
    # Ensure full path
    data_dir = os.path.join(os.getcwd(), "data")
    input_path = os.path.join(data_dir, f"gsm_{split}.txt")
    output_path = os.path.join(data_dir, f"gsm_{split}.json")
    
    with open(input_path, encoding="utf-8") as f:
        data = f.readlines()
    data = [
        {
            "question": d.split("||")[0],
            "steps": d.split("||")[1].split("##")[0].strip().split(" "),
            "answer": d.split("##")[-1].strip(),
        }
        for d in data
    ]
    json.dump(data, open(output_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert icot text data to JSON format."
    )
    parser.add_argument(
        "split", type=str, help="The dataset split (e.g., train, test, valid)."
    )
    args = parser.parse_args()
    main(args.split)
