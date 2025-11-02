from datasets import load_dataset
import pandas as pd

# Load the full dataset (no split = everything)
dataset = load_dataset("Ateeqq/AI-and-Human-Generated-Text")

# Merge all splits (train/test/etc.) into a single list of examples
df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset.keys()], ignore_index=True)

# Remove 'title' column if it exists
if "title" in df.columns:
    df = df.drop(columns=["title"])

# Rename 'label' to 'generated'
if "label" in df.columns:
    df = df.rename(columns={"label": "generated"})

# Convert 0 → False and 1 → True
df["generated"] = df["generated"].map({0: False, 1: True})

# Save the entire dataset as test_set.csv
df.to_csv("test_set.csv", index=False)

print("✅ All data saved as test_set.csv")
