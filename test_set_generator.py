import pandas as pd
from tqdm import tqdm

# ====== CONFIGURATION ======
parent_file = "/Users/tysali/Downloads/AI_Human.csv"
train_files = {
    "small": "subset_sm.csv",
    "medium": "subset_med.csv",
    "large": "subset_lg.csv"
}
text_column = "text"      # name of your text column
label_column = "generated"    # name of your label column (0 = human, 1 = AI-generated)
limit_per_class = 10000   # entries per class for test set
# ===========================

# === STEP 1: Load datasets ===
print("📥 Loading datasets...")
parent_df = pd.read_csv(parent_file)
train_dfs = {name: pd.read_csv(path) for name, path in tqdm(train_files.items(), desc="Loading training sets")}
print("✅ All datasets loaded.\n")

# === STEP 2: Remove duplicates (safety step) ===
print("🧹 Removing duplicates...")
parent_df = parent_df.drop_duplicates(subset=[text_column])
train_dfs = {name: df.drop_duplicates(subset=[text_column]) for name, df in train_dfs.items()}

# === STEP 3: Combine all training data ===
print("🔗 Combining all training data...")
train_combined = pd.concat(train_dfs.values(), ignore_index=True)
train_texts = set()
for text in tqdm(train_combined[text_column].astype(str), desc="Building training text set"):
    train_texts.add(text)

# === STEP 4: Filter parent to exclude anything in training ===
print("🚫 Removing training overlaps from parent dataset...")
parent_df[text_column] = parent_df[text_column].astype(str)
mask = ~parent_df[text_column].isin(train_texts)
available_for_test = parent_df[mask]

print(f"✅ Removed {len(parent_df) - len(available_for_test)} overlapping entries.")
print(f"Remaining available data: {len(available_for_test)}\n")

# === STEP 5: Balance test set (10,000 per class) ===
print(f"📊 Selecting up to {limit_per_class} entries per class for the test set...")
human_df = available_for_test[available_for_test[label_column] == 0]
ai_df = available_for_test[available_for_test[label_column] == 1]

sampled_human = human_df.sample(
    n=min(limit_per_class, len(human_df)), random_state=42
)
sampled_ai = ai_df.sample(
    n=min(limit_per_class, len(ai_df)), random_state=42
)

# Combine and shuffle
test_df = pd.concat([sampled_human, sampled_ai], ignore_index=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# === STEP 6: Save clean test set ===
test_df.to_csv("test_set_clean.csv", index=False)

# === STEP 7: Summary ===
print("\n=== Cleaning Summary ===")
print(f"Original parent size: {len(parent_df)}")
print(f"Removed (training overlap): {len(parent_df) - len(available_for_test)}")
print(f"Final test set size: {len(test_df)}")
print(f" - Human (label 0): {len(test_df[test_df[label_column] == 0])}")
print(f" - AI (label 1): {len(test_df[test_df[label_column] == 1])}")
print("==========================")
print("💾 Saved new test set as: test_set_clean.csv\n")
