# =========================================
# DigiCow Farmer Training Adoption - Final Pipeline
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import lightgbm as lgb

# ------------------------
# 1️⃣ Load data
# ------------------------
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

target = 'adopted_within_07_days'

# ------------------------
# 2️⃣ Feature Engineering
# ------------------------
# Topics count
train['num_topics'] = train['topics_list'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
test['num_topics'] = test['topics_list'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

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

# ------------------------
# 3️⃣ Select features
# ------------------------
features = [
    'gender', 'age', 'registration', 'belong_to_cooperative',
    'num_trainings_30d', 'num_trainings_60d', 'num_total_trainings',
    'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training', 'num_topics',
    'trainer', 'train_dayofweek', 'train_month', 'train_is_weekend',
    'county_trainer', 'age_gender'
]

X = train[features]
y = train[target]
X_test = test[features]

# ------------------------
# 4️⃣ Handle categorical features
# ------------------------
categorical_cols = ['gender', 'age', 'registration', 'trainer', 'county_trainer', 'age_gender']

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ------------------------
# 5️⃣ Train/validation split (stratified)
# ------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------
# 6️⃣ Train LightGBM model
# ------------------------
model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

# Fit with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# ------------------------
# 7️⃣ Validation evaluation
# ------------------------
y_val_pred = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation ROC-AUC: {roc_auc:.4f}")

# ------------------------
# 8️⃣ Test predictions and submission
# ------------------------
test_probs = model.predict_proba(X_test)[:, 1]

# Clip extreme probabilities to avoid Log Loss issues
eps = 1e-6
test_probs = np.clip(test_probs, eps, 1 - eps)

submission = pd.DataFrame({
    'ID': test['ID'],
    'Target_AUC': test_probs,
    'Target_LogLoss': test_probs
})

submission.to_csv("submission_lightgbm_final.csv", index=False)
print("✅ Submission file 'submission_lightgbm_final.csv' created successfully!")
print(f"Submission shape: {submission.shape}")
print(f"Probability range: [{test_probs.min():.6f}, {test_probs.max():.6f}]")
