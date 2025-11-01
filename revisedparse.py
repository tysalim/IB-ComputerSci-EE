import pandas as pd

# === CONFIG ===
input_file = "/Users/tysali/Downloads/AI_Human.csv"
output_file = "balanced_subset.csv"
n_samples_per_class = 10000
chunksize = 50000

collected = {True: [], False: []}

def normalize_generated_column(series):
    """Convert string/boolean/integer labels to True or False."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "ai": True, "1.0": True,
            "false": False, "human": False, "0.0": False
        })
    )

# --- Read CSV in chunks ---
total_rows = 0
for chunk in pd.read_csv(input_file, chunksize=chunksize):
    total_rows += len(chunk)

    # Skip if column missing
    if "generated" not in chunk.columns:
        print("⚠️ 'generated' column not found in this chunk — skipping.")
        continue

    # Drop rows without 'generated'
    chunk = chunk.dropna(subset=["generated"])
    if chunk.empty:
        continue

    # Normalize and filter
    chunk["generated"] = normalize_generated_column(chunk["generated"])
    chunk = chunk[chunk["generated"].isin([True, False])]

    # Add per class
    for label in [True, False]:
        subset = chunk[chunk["generated"] == label]
        if not subset.empty:
            collected[label].append(subset)

    # Show progress
    count_true = sum(len(df) for df in collected[True])
    count_false = sum(len(df) for df in collected[False])
    print(f"📊 Progress: {count_true} AI, {count_false} Human collected...")

    # Early stop
    if count_true >= n_samples_per_class and count_false >= n_samples_per_class:
        print("✅ Enough samples collected, stopping early.")
        break

# --- Safe concatenation ---
def safe_concat(label):
    if not collected[label]:
        print(f"⚠️ No samples found for label={label}! Check your data formatting.")
        return pd.DataFrame(columns=["text", "generated"])
    return pd.concat(collected[label], ignore_index=True)

df_true = safe_concat(True)
df_false = safe_concat(False)

# --- Sample safely ---
def safe_sample(df, n):
    if len(df) == 0:
        return df
    if len(df) < n:
        print(f"⚠️ Only {len(df)} samples available, using all of them.")
        return df
    return df.sample(n=n, random_state=42)

df_true = safe_sample(df_true, n_samples_per_class)
df_false = safe_sample(df_false, n_samples_per_class)

# --- Combine & Save ---
final_df = pd.concat([df_true, df_false]).sample(frac=1, random_state=42).reset_index(drop=True)
final_df.to_csv(output_file, index=False)

print(f"\n✅ Final counts:")
print(final_df["generated"].value_counts(dropna=False))
print(f"💾 Saved to '{output_file}'")
