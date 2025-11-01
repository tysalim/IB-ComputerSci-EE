import pandas as pd
import random
import sys
import types
import nltk
from tqdm import tqdm

# === PATCH for textaugment import bug ===
if 'textblob.translate' not in sys.modules:
    sys.modules['textblob.translate'] = types.ModuleType('textblob.translate')
    sys.modules['textblob.translate'].NotTranslated = Exception

# === IMPORT after patch ===
from textaugment import Translate

# === DOWNLOAD NLTK DATASETS (quietly) ===
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# === CONFIG ===
input_csv = "subset_sm.csv"        # Input file
output_csv = "subset_sm_rtt.csv"   # Output file
num_aug_per_row = 1                      # Number of augmented versions per text
random.seed(42)

# === LOAD DATA ===
df = pd.read_csv(input_csv)
df = df.dropna(subset=["text", "generated"])  # Remove rows missing data

src = "en" # source language of the sentence
to = "es" # target language

# === INIT AUGMENTER ===
augmenter = Translate(src, to)

# === AUGMENTATION LOOP WITH PROGRESS ===
augmented_rows = []
total_to_augment = len(df) * num_aug_per_row
successful_augs = 0
failed_augs = 0


print(f"🚀 Starting augmentation on {len(df)} rows ({total_to_augment} total augmentations)...\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting"):
    original_text = row["text"]
    label = row["generated"]

    # Add original text
    augmented_rows.append({"text": original_text, "generated": label})

    # Generate random augmentations
    for _ in range(num_aug_per_row):
        try:
            aug_text = augmenter.augment(original_text)
            augmented_rows.append({"text": aug_text, "generated": label})
            successful_augs += 1
        except Exception as e:
            failed_augs += 1
            print(f"⚠️ Augmentation failed for index {idx}: {e}")

# === BUILD NEW DATAFRAME ===
aug_df = pd.DataFrame(augmented_rows)

# === SHUFFLE AND SAVE ===
aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)
aug_df.to_csv(output_csv, index=False)

# === SUMMARY ===
print("\n✅ Augmentation complete!")
print(f"Total rows in final dataset: {len(aug_df)}")
print(f"Successful augmentations: {successful_augs}/{total_to_augment}")
print(f"Failed augmentations: {failed_augs}")
print(f"💾 Saved to: {output_csv}")
