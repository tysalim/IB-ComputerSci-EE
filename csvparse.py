import pandas as pd

# === CONFIG ===
input_file = "/Users/tysali/Downloads/AI_Human.csv"        # Path to your 1GB CSV
output_file = "balanced_subset.csv" # Output file
n_samples_per_class = 10000         # Number of samples to extract per class
chunksize = 50000                   # Number of rows to read per chunk

# Storage for partial data
collected = {True: [], False: []}

# Read the CSV file in chunks
for chunk in pd.read_csv(input_file, chunksize=chunksize):
    # Normalize boolean column (in case values are strings like "True"/"False")
    if chunk["generated"].dtype == object:
        chunk["generated"] = chunk["generated"].astype(str).str.lower().map({"true": True, "false": False})

    for label in [True, False]:
        subset = chunk[chunk["generated"] == label]
        if not subset.empty:
            collected[label].append(subset)

    # Check if we have enough samples for both classes
    enough_true = sum(len(df) for df in collected[True]) >= n_samples_per_class
    enough_false = sum(len(df) for df in collected[False]) >= n_samples_per_class

    if enough_true and enough_false:
        break

# Concatenate partial data
df_true = pd.concat(collected[True]).sample(n=n_samples_per_class, random_state=42)
df_false = pd.concat(collected[False]).sample(n=n_samples_per_class, random_state=42)

# Combine and shuffle
final_df = pd.concat([df_true, df_false]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save subset to new CSV
final_df.to_csv(output_file, index=False)

print(f"✅ Extracted {n_samples_per_class} AI-generated and {n_samples_per_class} human-generated texts.")
print(f"💾 Saved to {output_file}")
