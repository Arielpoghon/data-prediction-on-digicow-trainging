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

# Load Data
train = pd.read_csv("data/Train.csv")
test = pd.read_csv("data/Test.csv")

# Feature Engineering
train['num_topics'] = train['topics_list'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
test['num_topics'] = test['topics_list'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

for df in [train, test]:
    df['first_training_date'] = pd.to_datetime(df['first_training_date'])
    df['train_dayofweek'] = df['first_training_date'].dt.dayofweek
    df['train_month'] = df['first_training_date'].dt.month
    df['train_is_weekend'] = (df['train_dayofweek'] >= 5).astype(int)
    df['county_trainer'] = df['county'] + '_' + df['trainer']
    df['age_gender'] = df['age'] + '_' + df['gender']

    # Additional time-based features
    if 'days_to_second_training' in df.columns:
        df['training_frequency'] = df['num_total_trainings'] / (df['days_to_second_training'].fillna(0) + 1)

# Aggregated features
county_adopt_rate_07 = train.groupby('county')['adopted_within_07_days'].mean().rename('county_adopt_rate_07')
county_adopt_rate_90 = train.groupby('county')['adopted_within_90_days'].mean().rename('county_adopt_rate_90')
county_adopt_rate_120 = train.groupby('county')['adopted_within_120_days'].mean().rename('county_adopt_rate_120')

trainer_adopt_rate_07 = train.groupby('trainer')['adopted_within_07_days'].mean().rename('trainer_adopt_rate_07')
trainer_adopt_rate_90 = train.groupby('trainer')['adopted_within_90_days'].mean().rename('trainer_adopt_rate_90')
trainer_adopt_rate_120 = train.groupby('trainer')['adopted_within_120_days'].mean().rename('trainer_adopt_rate_120')

train = train.join(county_adopt_rate_07, on='county')
train = train.join(county_adopt_rate_90, on='county')
train = train.join(county_adopt_rate_120, on='county')
train = train.join(trainer_adopt_rate_07, on='trainer')
train = train.join(trainer_adopt_rate_90, on='trainer')
train = train.join(trainer_adopt_rate_120, on='trainer')

test = test.join(county_adopt_rate_07, on='county')
test = test.join(county_adopt_rate_90, on='county')
test = test.join(county_adopt_rate_120, on='county')
test = test.join(trainer_adopt_rate_07, on='trainer')
test = test.join(trainer_adopt_rate_90, on='trainer')
test = test.join(trainer_adopt_rate_120, on='trainer')

# Select Features
features = [
    'gender', 'age', 'registration', 'belong_to_cooperative',
    'num_total_trainings', 'num_repeat_trainings', 'days_to_second_training',
    'num_unique_trainers', 'has_second_training', 'num_topics',
    'train_dayofweek', 'train_month', 'train_is_weekend', 'county_trainer', 'age_gender',
    'county_adopt_rate_07', 'county_adopt_rate_90', 'county_adopt_rate_120',
    'trainer_adopt_rate_07', 'trainer_adopt_rate_90', 'trainer_adopt_rate_120'
]

if 'training_frequency' in train.columns:
    features.append('training_frequency')

X = train[features]
X_test = test[features]

# Define targets
y_07 = train['adopted_within_07_days']
y_90 = train['adopted_within_90_days']
y_120 = train['adopted_within_120_days']

# Handle Categorical Features
categorical_cols = ['gender', 'age', 'registration', 'county_trainer', 'age_gender']

for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Hyperparameter Tuning with RandomizedSearchCV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'num_leaves': [20, 50, 100, 200],
    'max_depth': [3, 6, 10, 15],
    'min_child_samples': [5, 10, 20, 30],
    'n_estimators': [500, 1000, 1500, 2000],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

def train_model(X, y):
    lgb_model = LGBMClassifier(objective='binary', random_state=42, verbose=-1)
    random_search = RandomizedSearchCV(
        lgb_model, param_distributions=param_dist, n_iter=30, cv=skf, scoring='roc_auc', random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    best_params = random_search.best_params_
    print("Best params:", best_params)
    best_model = LGBMClassifier(**best_params, objective='binary', random_state=42, verbose=-1)
    best_model.fit(X, y)
    return best_model

# Train models for each time window
model_07 = train_model(X, y_07)
model_90 = train_model(X, y_90)
model_120 = train_model(X, y_120)

# Generate Test Predictions
probs_07 = model_07.predict_proba(X_test)[:, 1]
probs_90 = model_90.predict_proba(X_test)[:, 1]
probs_120 = model_120.predict_proba(X_test)[:, 1]

# Calibration
calibrated_07 = CalibratedClassifierCV(model_07, method='sigmoid', cv=skf)
calibrated_07.fit(X, y_07)
calibrated_probs_07 = calibrated_07.predict_proba(X_test)[:, 1]

calibrated_90 = CalibratedClassifierCV(model_90, method='sigmoid', cv=skf)
calibrated_90.fit(X, y_90)
calibrated_probs_90 = calibrated_90.predict_proba(X_test)[:, 1]

calibrated_120 = CalibratedClassifierCV(model_120, method='sigmoid', cv=skf)
calibrated_120.fit(X, y_120)
calibrated_probs_120 = calibrated_120.predict_proba(X_test)[:, 1]

# Clip probabilities to ensure they are within [0, 1]
eps = 1e-6
calibrated_probs_07 = np.clip(calibrated_probs_07, eps, 1 - eps)
calibrated_probs_90 = np.clip(calibrated_probs_90, eps, 1 - eps)
calibrated_probs_120 = np.clip(calibrated_probs_120, eps, 1 - eps)

# Create Submission File
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
submission.to_csv('submission_final_v3.csv', index=False)
print("✅ Submission file 'submission_final_v3.csv' created successfully!")
print(f"Submission shape: {submission.shape}")
