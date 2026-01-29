# =========================================
# DigiCow Challenge - Data Exploration Report
# =========================================

import pandas as pd
import numpy as np

# ------------------------
# Load datasets
# ------------------------
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

print("=" * 60)
print("DIGICOW FARMER TRAINING ADOPTION - DATA EXPLORATION REPORT")
print("=" * 60)

# ------------------------
# Basic shape information
# ------------------------
print("\n📊 DATASET SHAPES:")
print(f"Training set: {train.shape[0]:,} samples, {train.shape[1]} features")
print(f"Test set: {test.shape[0]:,} samples, {test.shape[1]} features")

# ------------------------
# Target variable distribution
# ------------------------
print("\n🎯 TARGET VARIABLE DISTRIBUTION:")
target_counts = train['adopted_within_07_days'].value_counts()
target_pct = train['adopted_within_07_days'].value_counts(normalize=True) * 100

for val, count in target_counts.items():
    pct = target_pct[val]
    label = "Adopted" if val == 1 else "Not Adopted"
    print(f"  {label} ({val}): {count:,} samples ({pct:.1f}%)")

# ------------------------
# Feature types
# ------------------------
print("\n📋 FEATURE TYPES:")

# Identify categorical and numeric features
categorical_features = train.select_dtypes(include=['object']).columns.tolist()
numeric_features = train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove ID and target from feature lists
if 'ID' in categorical_features:
    categorical_features.remove('ID')
if 'adopted_within_07_days' in numeric_features:
    numeric_features.remove('adopted_within_07_days')

print(f"Categorical features ({len(categorical_features)}):")
for feat in categorical_features:
    print(f"  - {feat}")

print(f"\nNumeric features ({len(numeric_features)}):")
for feat in numeric_features:
    print(f"  - {feat}")

# ------------------------
# Missing values
# ------------------------
print("\n❌ MISSING VALUES:")

train_missing = train.isnull().sum()
test_missing = test.isnull().sum()

print("Training set missing values:")
missing_train = train_missing[train_missing > 0]
if len(missing_train) > 0:
    for col, count in missing_train.items():
        pct = (count / len(train)) * 100
        print(f"  - {col}: {count:,} ({pct:.1f}%)")
else:
    print("  No missing values")

print("\nTest set missing values:")
missing_test = test_missing[test_missing > 0]
if len(missing_test) > 0:
    for col, count in missing_test.items():
        pct = (count / len(test)) * 100
        print(f"  - {col}: {count:,} ({pct:.1f}%)")
else:
    print("  No missing values")

# ------------------------
# Numeric feature statistics
# ------------------------
print("\n📈 NUMERIC FEATURES STATISTICS:")

numeric_stats = train[numeric_features].describe().round(3)
print(numeric_stats)

# ------------------------
# Categorical feature unique values
# ------------------------
print("\n🏷️  CATEGORICAL FEATURES - UNIQUE VALUES:")

for feat in categorical_features:
    train_unique = train[feat].nunique()
    test_unique = test[feat].nunique()
    print(f"  - {feat}: {train_unique} unique values (train), {test_unique} unique values (test)")
    
    # Show top categories for features with reasonable number of unique values
    if train_unique <= 10:
        top_categories = train[feat].value_counts().head(5)
        print(f"    Top categories: {dict(top_categories)}")

# ------------------------
# First few rows preview
# ------------------------
print("\n👀 FIRST 3 ROWS OF TRAINING DATA:")
print(train.head(3).to_string())

print("\n👀 FIRST 3 ROWS OF TEST DATA:")
print(test.head(3).to_string())

# ------------------------
# Summary
# ------------------------
print("\n" + "=" * 60)
print("SUMMARY:")
print(f"- Total samples: {train.shape[0] + test.shape[0]:,}")
print(f"- Training features: {train.shape[1] - 2}")  # Excluding ID and target
print(f"- Test features: {test.shape[1] - 1}")      # Excluding ID
print(f"- Target imbalance: {target_pct[1]:.1f}% adoption rate")
print(f"- Missing data: {'Yes' if len(missing_train) > 0 or len(missing_test) > 0 else 'No'}")
print("=" * 60)
