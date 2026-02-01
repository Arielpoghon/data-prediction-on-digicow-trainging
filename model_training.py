# =========================================
# DigiCow Farmer Training Adoption - Enhanced Pipeline
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import TargetEncoder
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

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
    df['train_dayofyear'] = df['first_training_date'].dt.dayofyear

# Interaction features
for df in [train, test]:
    df['county_trainer'] = df['county'] + '_' + df['trainer']
    df['age_gender'] = df['age'] + '_' + df['gender']
    df['cooperative_age'] = df['belong_to_cooperative'].astype(str) + '_' + df['age']

# Target encoding for high-cardinality categorical features
categorical_cols = ['trainer', 'county_trainer', 'age_gender', 'cooperative_age']
for col in categorical_cols:
    encoder = TargetEncoder(target_type='binary', random_state=42)
    train[col + '_encoded'] = encoder.fit_transform(train[col], train[target])
    test[col + '_encoded'] = encoder.transform(test[col])

# ------------------------
# 3️⃣ Select features
# ------------------------
features = [
    'gender', 'age', 'registration', 'belong_to_cooperative',
    'num_trainings_30d', 'num_trainings_60d', 'num_total_trainings',
    'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training', 'num_topics',
    'trainer_encoded', 'county_trainer_encoded', 'age_gender_encoded',
    'cooperative_age_encoded', 'train_dayofweek', 'train_month',
    'train_is_weekend', 'train_dayofyear'
]

X = train[features]
y = train[target]
X_test = test[features]

# ------------------------
# 4️⃣ Handle categorical features
# ------------------------
categorical_cols = ['gender', 'age', 'registration']

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
# 6️⃣ Hyperparameter Tuning with RandomizedSearchCV
# ------------------------
param_dist = {
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'num_leaves': [32, 64, 128],
    'n_estimators': [500, 1000, 2000],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

search = RandomizedSearchCV(
    LGBMClassifier(objective='binary', random_state=42, verbose=-1),
    param_distributions=param_dist,
    n_iter=20,
    scoring='roc_auc',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

print(f"Best parameters: {search.best_params_}")
print(f"Best validation ROC-AUC: {search.best_score_:.4f}")

# ------------------------
# 7️⃣ Probability Calibration for Log Loss
# ------------------------
calibrated_model = CalibratedClassifierCV(
    best_model,
    method='isotonic',
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)
calibrated_model.fit(X_train, y_train)

# ------------------------
# 8️⃣ Validation evaluation
# ------------------------
y_val_pred = calibrated_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_pred)
logloss = log_loss(y_val, y_val_pred)

print(f"Validation ROC-AUC: {roc_auc:.4f}")
print(f"Validation Log Loss: {logloss:.4f}")

# ------------------------
# 9️⃣ Test predictions and submission
# ------------------------
test_probs = calibrated_model.predict_proba(X_test)[:, 1]

# Clip extreme probabilities to avoid Log Loss issues
eps = 1e-6
test_probs = np.clip(test_probs, eps, 1 - eps)

submission = pd.DataFrame({
    'ID': test['ID'],
    'Target_AUC': test_probs,
    'Target_LogLoss': test_probs
})

submission.to_csv("submission_lightgbm_calibrated.csv", index=False)
print("✅ Submission file 'submission_lightgbm_calibrated.csv' created successfully!")
print(f"Submission shape: {submission.shape}")
print(f"Probability range: [{test_probs.min():.6f}, {test_probs.max():.6f}]")
