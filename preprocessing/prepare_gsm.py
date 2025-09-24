import sys
import os
import urllib.request
import subprocess

os.makedirs("data", exist_ok=True)

urls = {
    "train": "https://media.githubusercontent.com/media/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/train.txt",
    "valid": "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/valid.txt",
    "test":  "https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k/test.txt",
}

for split, url in urls.items():
    out_file = f"data/gsm_{split}.txt"
    print(f"Downloading {split}...")
    urllib.request.urlretrieve(url, out_file)

    # Make sure it uses the same Python interpreter
    subprocess.run([sys.executable, os.path.join("preprocessing", "gsm_icot.py"), split], check=True)
    os.remove(out_file)

print("GSM8K preprocessing done. JSON files are in data/")
