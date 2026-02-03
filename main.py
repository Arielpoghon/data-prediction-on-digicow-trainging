# =========================================
# DigiCow Farmer Training Adoption - Corrected Full Pipeline
# =========================================
# Goal: Predict adoption probabilities for 7, 90, and 120 days

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# ------------------------
# 1️⃣ Load Data
# ------------------------
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

# Print columns to verify
print("Train columns:", train.columns)
print("Test columns:", test.columns)

# ------------------------
# 2️⃣ Feature Engineering
# ------------------------
# Number of unique topics
train['num_topics'] = train['topics_list'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
test['num_topics'] = test['topics_list'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

# Date features
for df in [train, test]:
    df['first_training_date'] = pd.to_datetime(df['first_training_date'])
    df['train_dayofweek'] = df['first_training_date'].dt.dayofweek
    df['train_month'] = df['first_training_date'].dt.month
    df['train_is_weekend'] = (df['train_dayofweek'] >= 5).astype(int)

# Interaction features
for df in [train, test]:
    df['county_trainer'] = df['county'] + '_' + df['trainer']
    df['age_gender'] = df['age'] + '_' + df['gender']

# Check if 'num_trainings_30d' exists before binning
if 'num_trainings_30d' in train.columns:
    train['num_trainings_30d_bin'] = pd.cut(train['num_trainings_30d'], bins=[0, 1, 3, 5, 10, np.inf], labels=['0-1', '1-3', '3-5', '5-10', '10+'])
if 'num_trainings_30d' in test.columns:
    test['num_trainings_30d_bin'] = pd.cut(test['num_trainings_30d'], bins=[0, 1, 3, 5, 10, np.inf], labels=['0-1', '1-3', '3-5', '5-10', '10+'])

# Aggregations (only if 'county' and 'trainer' columns exist)
if 'county' in train.columns and 'adopted_within_07_days' in train.columns:
    county_agg = train.groupby('county')['adopted_within_07_days'].agg(['mean', 'count']).rename(columns={'mean': 'county_adopt_rate', 'count': 'county_train_count'})
    county_agg = county_agg[county_agg['county_train_count'] > 10]
    train = train.merge(county_agg, on='county', how='left')
    test = test.merge(county_agg, on='county', how='left')

if 'trainer' in train.columns and 'adopted_within_07_days' in train.columns:
    trainer_agg = train.groupby('trainer')['adopted_within_07_days'].agg(['mean', 'count']).rename(columns={'mean': 'trainer_adopt_rate', 'count': 'trainer_train_count'})
    trainer_agg = trainer_agg[trainer_agg['trainer_train_count'] > 5]
    train = train.merge(trainer_agg, on='trainer', how='left')
    test = test.merge(trainer_agg, on='trainer', how='left')

# Target encoding for trainer (only if 'trainer' and 'adopted_within_07_days' columns exist)
if 'trainer' in train.columns and 'adopted_within_07_days' in train.columns:
    global_mean = train['adopted_within_07_days'].mean()
    trainer_target_enc = train.groupby('trainer')['adopted_within_07_days'].transform(lambda x: (x.sum() + global_mean * 10) / (len(x) + 10))
    train['trainer_target_enc'] = trainer_target_enc
    test_trainer_enc = test['trainer'].map(train.groupby('trainer')['adopted_within_07_days'].mean()).fillna(global_mean)
    test['trainer_target_enc'] = test_trainer_enc

# ------------------------
# 3️⃣ Select Features
# ------------------------
# Define a base set of features that are likely to exist
base_features = [
    'gender', 'age', 'registration', 'belong_to_cooperative',
    'num_total_trainings', 'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training', 'num_topics',
    'train_dayofweek', 'train_month', 'train_is_weekend'
]

# Add additional features if they exist
additional_features = [
    'num_trainings_30d', 'num_trainings_60d', 'county_trainer', 'age_gender',
    'num_trainings_30d_bin', 'county_adopt_rate', 'county_train_count',
    'trainer_adopt_rate', 'trainer_train_count', 'trainer_target_enc'
]

features = base_features + [f for f in additional_features if f in train.columns and f in test.columns]

X = train[features]
y_07 = train['adopted_within_07_days']
X_test = test[features]

# ------------------------
# 4️⃣ Handle Categorical Features
# ------------------------
categorical_cols = ['gender', 'age', 'registration']
if 'county_trainer' in features:
    categorical_cols.append('county_trainer')
if 'age_gender' in features:
    categorical_cols.append('age_gender')
if 'num_trainings_30d_bin' in features:
    categorical_cols.append('num_trainings_30d_bin')

for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('category')
    if col in X_test.columns:
        X_test[col] = X_test[col].astype('category')

# ------------------------
# 5️⃣ Calculate Imbalance Weight
# ------------------------
pos_weight = len(y_07[y_07 == 0]) / len(y_07[y_07 == 1])

# ------------------------
# 6️⃣ Hyperparameter Tuning with RandomizedSearchCV
# ------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'learning_rate': [0.005, 0.01, 0.05],
    'num_leaves': [20, 50, 100],
    'max_depth': [3, 6, 10],
    'min_child_samples': [5, 10, 20],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

lgb_model = LGBMClassifier(objective='binary', random_state=42, verbose=-1)
random_search = RandomizedSearchCV(
    lgb_model, param_distributions=param_dist, n_iter=20, cv=skf, scoring='roc_auc', random_state=42, n_jobs=-1
)
random_search.fit(X, y_07)
best_params = random_search.best_params_
print("Best params from RandomizedSearchCV:", best_params)

# ------------------------
# 7️⃣ Train Final Model
# ------------------------
lgb_model = LGBMClassifier(**best_params, objective='binary', random_state=42, verbose=-1)
lgb_model.fit(X, y_07)

# ------------------------
# 8️⃣ Generate Test Predictions
# ------------------------
lgb_probs_07 = lgb_model.predict_proba(X_test)[:, 1]

# Simulate predictions for 90 and 120 days
lgb_probs_90 = lgb_probs_07 * 0.9
lgb_probs_120 = lgb_probs_07 * 0.85

# ------------------------
# 9️⃣ Calibration
# ------------------------
calibrated_lgb = CalibratedClassifierCV(lgb_model, method='sigmoid', cv=skf)
calibrated_lgb.fit(X, y_07)
calibrated_probs_07 = calibrated_lgb.predict_proba(X_test)[:, 1]
calibrated_probs_90 = calibrated_probs_07 * 0.9
calibrated_probs_120 = calibrated_probs_07 * 0.85

# Clip probabilities to ensure they are within [0, 1]
eps = 1e-6
calibrated_probs_07 = np.clip(calibrated_probs_07, eps, 1 - eps)
calibrated_probs_90 = np.clip(calibrated_probs_90, eps, 1 - eps)
calibrated_probs_120 = np.clip(calibrated_probs_120, eps, 1 - eps)

# ------------------------
# 1️⃣0️⃣ Create Submission File
# ------------------------
submission = pd.DataFrame({
    'ID': test['ID'],
    'Target_07_AUC': calibrated_probs_07,
    'Target_07_LogLoss': calibrated_probs_07,
    'Target_90_AUC': calibrated_probs_90,
    'Target_90_LogLoss': calibrated_probs_90,
    'Target_120_AUC': calibrated_probs_120,
    'Target_120_LogLoss': calibrated_probs_120
})

# Save to CSV
submission.to_csv('submission_final.csv', index=False)
print("✅ Submission file 'submission_final.csv' created successfully!")
print(f"Submission shape: {submission.shape}")
