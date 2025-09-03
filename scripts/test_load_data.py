import json
import os

# folder where the script is
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "..", "data")
train_file = os.path.join(data_dir, "gsm8k_mini_train.json")
val_file = os.path.join(data_dir, "gsm8k_mini_val.json")

# load and inspect
with open(train_file, "r", encoding="utf-8") as f:
    train = json.load(f)
with open(val_file, "r", encoding="utf-8") as f:
    val = json.load(f)

print(f"Train items: {len(train)}")
print(f"Val items: {len(val)}")
print("Sample train item:")
print(train[0])
