import os
import csv
from sklearn.model_selection import train_test_split

def split_spectral_data(root_dir, data_dir, test_size=0.2, val_size=0.1):
    # Load the data
    patches_dir = os.path.join(root_dir, data_dir)
    print(patches_dir)
    all_files = []
    for root, dirs, files in os.walk(patches_dir):
        for file in files:
            if file.endswith(".tif"):
                all_files.append(os.path.relpath(os.path.join(root, file), root_dir))

    # Split the data into train, validation, and test sets
    train_files, test_files = train_test_split(all_files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size / (1 - test_size), random_state=42)

    # Save the splits into CSV files
    splits_dir = os.path.join(root_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    with open(os.path.join(splits_dir, "train.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(train_files)

    with open(os.path.join(splits_dir, "val.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(val_files)

    with open(os.path.join(splits_dir, "test.csv"), "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(test_files)

    print(f"Data split into train, validation, and test sets and saved to {splits_dir}")

# Example usage
root_dir = "/data/behnood/spectral_earth"
split_spectral_data(root_dir, data_dir="spectral_earth50K", test_size=0.2, val_size=0.1)