import pandas as pd
from tqdm import tqdm

# ====== CONFIGURATION ======
parent_file = "/Users/tysali/Downloads/AI_Human.csv"
train_files = {
    "small": "subset_sm.csv",
    "medium": "subset_med.csv",
    "large": "subset_lg.csv"
}
text_column = "text"      # name of the text column
label_column = "generated"    # name of the label column (0.0 = human, 1.0 = AI)
limit_per_class = 10000   # target per class for test set
# ===========================

# === STEP 1: Load datasets ===
print("📥 Loading datasets...")
parent_df = pd.read_csv(parent_file)
train_dfs = {name: pd.read_csv(path) for name, path in tqdm(train_files.items(), desc="Loading training sets")}
print("✅ All datasets loaded.\n")

# === STEP 2: Clean duplicates ===
print("🧹 Removing duplicates...")
parent_df = parent_df.drop_duplicates(subset=[text_column])
train_dfs = {name: df.drop_duplicates(subset=[text_column]) for name, df in train_dfs.items()}

# === STEP 3: Combine all training data ===
print("🔗 Combining all training data...")
train_combined = pd.concat(train_dfs.values(), ignore_index=True)
train_texts = set()
for text in tqdm(train_combined[text_column].astype(str), desc="Building training text set"):
    train_texts.add(text)

# === STEP 4: Remove overlaps ===
print("🚫 Removing training overlaps from parent dataset...")
parent_df[text_column] = parent_df[text_column].astype(str)
mask = ~parent_df[text_column].isin(train_texts)
available_for_test = parent_df[mask]

removed_count = len(parent_df) - len(available_for_test)
print(f"✅ Removed {removed_count} overlapping entries.")
print(f"Remaining available data: {len(available_for_test)}\n")

# === STEP 5: Check label distribution before sampling ===
label_counts = available_for_test[label_column].value_counts(dropna=False)
print("📊 Available samples by label:")
print(label_counts)

if 0.0 not in label_counts or 1.0 not in label_counts:
    raise ValueError("❌ Missing one or both label classes (0.0 or 1.0) in remaining data — cannot balance test set.")

if label_counts[0.0] == 0 or label_counts[1.0] == 0:
    raise ValueError("❌ One of the label classes has zero available samples after overlap removal.")

# === STEP 6: Sample up to 10k of each ===
print(f"\n🎯 Sampling up to {limit_per_class} per class...")
human_df = available_for_test[available_for_test[label_column] == 0.0]
ai_df = available_for_test[available_for_test[label_column] == 1.0]

sampled_human = human_df.sample(
    n=min(limit_per_class, len(human_df)), random_state=42
)
sampled_ai = ai_df.sample(
    n=min(limit_per_class, len(ai_df)), random_state=42
)

# Combine and shuffle
test_df = pd.concat([sampled_human, sampled_ai], ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === STEP 7: Convert labels to Boolean ===
test_df[label_column] = test_df[label_column].map({0.0: False, 1.0: True})

# === STEP 8: Save final test set ===
if len(test_df) == 0:
    raise ValueError("❌ No samples available for test set after filtering overlaps.")
else:
    test_df.to_csv("test_set_clean.csv", index=False)

# === STEP 9: Summary ===
print("\n=== Cleaning Summary ===")
print(f"Original parent size: {len(parent_df)}")
print(f"Removed (training overlap): {removed_count}")
print(f"Final test set size: {len(test_df)}")
print(f" - Human (False): {len(test_df[test_df[label_column] == False])}")
print(f" - AI (True): {len(test_df[test_df[label_column] == True])}")
print("==========================")
print("💾 Saved new test set as: test_set_clean.csv\n")
