#!/usr/bin/env python
"""

Last update: Lucia Licakova, 2025-09-03

Prepare a small GSM8K dataset sample for Coconut training on CPU.
- Downloads the Hugging Face "gsm8k" dataset (train/test).
- Extracts a small subset (default: 200 train, 100 val).
- Splits long answers into a list of reasoning steps which Coconut expects.
- Extracts the final numeric answer.
- Saves everything into JSON files compatible with Coconut training.

Usage (from the repo root on Windows):
    python scripts\prepare_gsm8k_mini.py --train_n 200 --val_n 100 --outdir data

After running, you will find two files in the 'data' folder:
    - gsm8k_mini_train.json
    - gsm8k_mini_val.json

Each entry looks like:
{
  "question": "Jane has 3 apples...",
  "answer": "12",
  "steps": [
    "Jane starts with 3 apples.",
    "She buys 9 more.",
    "Now she has 12."
  ]
}
"""
import json, re, random, argparse, os
from datasets import load_dataset

def split_steps(text):
    # take text before final "#### answer"
    text = text.split("####")[0]
    text = re.sub(r'\s+', ' ', text).strip()
    # split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    # remove leading/trailing spaces from non-empty strings
    steps = [p.strip() for p in parts if len(p.strip()) > 0]
    # keep the first 8 steps, if no steps were found, fall back to a dummy step
    return steps[:8] if steps else ["Think."]

def extract_final(ans):
    # extract the final numeric answer from the #### marker
    m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', ans)
    # if not found, fall back to the last token in the answer string
    return m.group(1) if m else ans.strip().split()[-1]

def convert(split="train", n=200, seed=123):
    """
    Load a split of the GSM8K dataset and convert it into Coconut format.
    - split: "train" or "test".
    - n: number of examples to keep.
    - seed: random seed for reproducibility.
    Returns a list of dicts with {question, answer, steps}.
    """
    # load the main configuration of the GSM8K dataset from Hugging Face
    # the data is stored in Parquet format
    ds = load_dataset("gsm8k", "main", split=split)
    items = []
    rng = random.Random(seed)
    # pick a random subset of n indices
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    idxs = idxs[:n]
    for i in idxs:
        q = ds[i]["question"].strip()
        a = ds[i]["answer"].strip()
        item = {
            "question": q,
            "answer": extract_final(a),
            "steps": split_steps(a)
        }
        items.append(item)
    return items

def main():
    """
    Entry point:
    - Parse command line arguments.
    - Generate train/validation subsets.
    - Save them as JSON files.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_n", type=int, default=200, help="Number of training samples to keep")
    ap.add_argument("--val_n", type=int, default=100, help="Number of validation samples to keep")
    ap.add_argument("--outdir", type=str, default="data", help="Where to save the JSON output")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    # build train and validation sets
    train = convert("train", args.train_n, 123)
    val = convert("test", args.val_n, 456)
    # save to json
    with open(os.path.join(args.outdir, "gsm8k_mini_train.json"), "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "gsm8k_mini_val.json"), "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
