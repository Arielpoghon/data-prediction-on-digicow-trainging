# =========================================
# DigiCow Challenge - Advanced Feature Engineering
# =========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ------------------------
# Load data
# ------------------------
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

target = 'adopted_within_07_days'

print("=" * 60)
print("DIGICOW - ADVANCED FEATURE ENGINEERING")
print("=" * 60)

# ------------------------
# 1️⃣ Basic Feature Engineering
# ------------------------
print("\n🔧 1️⃣ BASIC FEATURE ENGINEERING:")

# Topics count feature
train['num_topics'] = train['topics_list'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
test['num_topics'] = test['topics_list'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
print("✅ Created num_topics feature")

# Date features
for df in [train, test]:
    df['first_training_date'] = pd.to_datetime(df['first_training_date'])
    df['train_dayofweek'] = df['first_training_date'].dt.dayofweek
    df['train_month'] = df['first_training_date'].dt.month
    df['train_is_weekend'] = (df['train_dayofweek'] >= 5).astype(int)
    df['train_quarter'] = df['first_training_date'].dt.quarter
    df['train_dayofyear'] = df['first_training_date'].dt.dayofyear
print("✅ Created date features")

# ------------------------
# 2️⃣ Missing Value Handling
# ------------------------
print("\n🔧 2️⃣ MISSING VALUE HANDLING:")

# Fill missing values appropriately
for df in [train, test]:
    # For days_to_second_training: fill with large value indicating no second training
    df['days_to_second_training'] = df['days_to_second_training'].fillna(999)
    
    # For has_second_training: fill with 0 (no second training)
    df['has_second_training'] = df['has_second_training'].fillna(0)
    
    # For numeric features: fill with median
    numeric_cols = ['belong_to_cooperative', 'num_trainings_30d', 'num_trainings_60d', 
                   'num_total_trainings', 'num_repeat_trainings', 'num_unique_trainers']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical features: fill with 'Unknown'
    categorical_cols = ['gender', 'age', 'registration', 'county', 'subcounty', 'ward', 'trainer']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

print("✅ Handled missing values")

# ------------------------
# 3️⃣ Target Encoding for High-Cardinality Features
# ------------------------
print("\n🔧 3️⃣ TARGET ENCODING:")

# Target encode trainer (high cardinality)
trainer_mean = train.groupby('trainer')[target].mean()
overall_mean = train[target].mean()

train['trainer_encoded'] = train['trainer'].map(trainer_mean).fillna(overall_mean)
test['trainer_encoded'] = test['trainer'].map(trainer_mean).fillna(overall_mean)
print("✅ Target encoded trainer feature")

# Target encode county (moderate cardinality)
county_mean = train.groupby('county')[target].mean()
train['county_encoded'] = train['county'].map(county_mean).fillna(overall_mean)
test['county_encoded'] = test['county'].map(county_mean).fillna(overall_mean)
print("✅ Target encoded county feature")

# ------------------------
# 4️⃣ Interaction Features
# ------------------------
print("\n🔧 4️⃣ INTERACTION FEATURES:")

for df in [train, test]:
    # Training intensity interactions
    df['training_intensity'] = df['num_trainings_30d'] * df['num_trainings_60d']
    df['training_frequency'] = df['num_total_trainings'] / (df['days_to_second_training'] + 1)  # +1 to avoid division by zero
    
    # Demographic interactions
    df['age_gender_interaction'] = df['age'] + '_' + df['gender']
    df['registration_cooperative'] = df['registration'] + '_' + df['belong_to_cooperative'].astype(str)
    
    # Trainer effectiveness interactions
    df['trainer_county_interaction'] = df['trainer'] + '_' + df['county']
    
    # Time-based interactions
    df['training_month_intensity'] = df['train_month'] * df['num_trainings_30d']
    df['weekend_training_effect'] = df['train_is_weekend'] * df['num_trainings_30d']

print("✅ Created interaction features")

# ------------------------
# 5️⃣ Aggregated Features
# ------------------------
print("\n🔧 5️⃣ AGGREGATED FEATURES:")

# Trainer-level aggregations
trainer_stats = train.groupby('trainer').agg({
    'num_total_trainings': ['mean', 'std'],
    'adopted_within_07_days': ['mean', 'count']
}).round(3)

trainer_stats.columns = ['trainer_avg_trainings', 'trainer_std_trainings', 'trainer_adoption_rate', 'trainer_farmers_count']
trainer_stats = trainer_stats.reset_index()

# Merge back to train and test
train = train.merge(trainer_stats, on='trainer', how='left')
test = test.merge(trainer_stats, on='trainer', how='left')

# Fill missing values for new trainer stats
for col in ['trainer_avg_trainings', 'trainer_std_trainings', 'trainer_adoption_rate', 'trainer_farmers_count']:
    train[col] = train[col].fillna(train[col].median())
    test[col] = test[col].fillna(train[col].median())

print("✅ Created trainer aggregation features")

# ------------------------
# 6️⃣ Feature Scaling
# ------------------------
print("\n🔧 6️⃣ FEATURE SCALING:")

# Identify numerical features for scaling
numerical_features = [
    'belong_to_cooperative', 'num_trainings_30d', 'num_trainings_60d',
    'num_total_trainings', 'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training', 'num_topics',
    'train_dayofweek', 'train_month', 'train_is_weekend', 'train_quarter',
    'train_dayofyear', 'trainer_encoded', 'county_encoded',
    'training_intensity', 'training_frequency', 'training_month_intensity',
    'weekend_training_effect', 'trainer_avg_trainings', 'trainer_std_trainings',
    'trainer_adoption_rate', 'trainer_farmers_count'
]

# Filter to only existing columns
numerical_features = [col for col in numerical_features if col in train.columns]

# Scale numerical features
scaler = StandardScaler()
train[numerical_features] = scaler.fit_transform(train[numerical_features])
test[numerical_features] = scaler.transform(test[numerical_features])

print("✅ Scaled numerical features")

# ------------------------
# 7️⃣ Categorical Encoding
# ------------------------
print("\n🔧 7️⃣ CATEGORICAL ENCODING:")

# Low cardinality features for one-hot encoding
low_cardinality_features = ['gender', 'age', 'registration']
low_cardinality_features = [col for col in low_cardinality_features if col in train.columns]

# One-hot encode low cardinality features
train_encoded = pd.get_dummies(train, columns=low_cardinality_features, drop_first=True)
test_encoded = pd.get_dummies(test, columns=low_cardinality_features, drop_first=True)

# Ensure test has same columns as train
for col in train_encoded.columns:
    if col not in test_encoded.columns and col != target:
        test_encoded[col] = 0

# Align columns
test_encoded = test_encoded[train_encoded.drop(columns=[target]).columns]

print("✅ One-hot encoded categorical features")

# ------------------------
# 8️⃣ Final Feature Selection
# ------------------------
print("\n🔧 8️⃣ FINAL FEATURE SELECTION:")

# Remove columns that shouldn't be used for training
exclude_cols = ['ID', 'first_training_date', 'topics_list', 'county', 'subcounty', 'ward', 
                'trainer', 'age_gender_interaction', 'registration_cooperative', 
                'trainer_county_interaction']

feature_cols = [col for col in train_encoded.columns if col not in exclude_cols + [target]]

print(f"✅ Selected {len(feature_cols)} features for modeling")

# ------------------------
# 9️⃣ Prepare Final Datasets
# ------------------------
print("\n🔧 9️⃣ PREPARING FINAL DATASETS:")

# Final training and test matrices
X_train_final = train_encoded[feature_cols]
y_train_final = train_encoded[target]
X_test_final = test_encoded[feature_cols]

# Train/validation split for model evaluation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_final, y_train_final, test_size=0.2, random_state=42, stratify=y_train_final
)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test_final.shape}")

# ------------------------
# 🔟 Save Processed Data
# ------------------------
print("\n🔧 🔟 SAVING PROCESSED DATA:")

# Save processed datasets
train_processed = pd.concat([X_train, y_train], axis=1)
val_processed = pd.concat([X_val, y_val], axis=1)

train_processed.to_csv("train_processed.csv", index=False)
val_processed.to_csv("val_processed.csv", index=False)
X_test_final.to_csv("test_processed.csv", index=False)

# Save feature list
with open("feature_list.txt", "w") as f:
    for feature in feature_cols:
        f.write(f"{feature}\n")

print("✅ Saved processed datasets")

# ------------------------
# 📊 Feature Importance Summary
# ------------------------
print("\n📊 FEATURE ENGINEERING SUMMARY:")
print("=" * 60)

print(f"\n📈 Dataset Shapes:")
print(f"- Original training: {train.shape}")
print(f"- Processed training: {X_train_final.shape}")
print(f"- Processed test: {X_test_final.shape}")

print(f"\n🔧 Feature Categories:")
print(f"- Numerical features: {len([col for col in feature_cols if col in numerical_features])}")
print(f"- One-hot encoded: {len([col for col in feature_cols if any(col.startswith(prefix) for prefix in ['gender_', 'age_', 'registration_'])])}")
print(f"- Target encoded: {len([col for col in feature_cols if 'encoded' in col])}")
print(f"- Interaction features: {len([col for col in feature_cols if any(x in col for x in ['intensity', 'frequency', 'interaction', 'effect'])])}")
print(f"- Aggregated features: {len([col for col in feature_cols if 'trainer_' in col and col not in ['trainer_encoded']])}")

print(f"\n🎯 Target Distribution:")
print(f"- Training adoption rate: {y_train_final.mean():.3f}")
print(f"- Validation adoption rate: {y_val.mean():.3f}")

print(f"\n💾 Files Created:")
print(f"- train_processed.csv")
print(f"- val_processed.csv") 
print(f"- test_processed.csv")
print(f"- feature_list.txt")

print("\n✅ Feature engineering complete! Ready for model training.")
print("=" * 60)
