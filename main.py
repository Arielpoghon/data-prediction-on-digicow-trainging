import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility (required by Zindi rules)
np.random.seed(42)

# Define data directory (relative path to match your file location)
DATA_DIR = './data'

def load_csv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found at '{filepath}'. Check the path and ensure the file exists.")
        raise

print("=== Advanced DigiCow Solution (XGBoost + TF-IDF) ===")
print("Loading data...")

# Load data
train = load_csv('Train.csv')
test = load_csv('Test.csv')
sample_sub = load_csv('SampleSubmission.csv')

# Define targets
targets = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']

# Enhanced preprocessing with TF-IDF
def preprocess(df):
    # Feature engineering
    if 'topics_list' in df.columns:
        df['topics_count'] = df['topics_list'].fillna('').str.split(',').str.len()
        # TF-IDF on topics
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        topics_tfidf = vectorizer.fit_transform(df['topics_list'].fillna(''))
        tfidf_df = pd.DataFrame(topics_tfidf.toarray(), columns=[f'tfidf_{i}' for i in range(10)])
        df = pd.concat([df, tfidf_df], axis=1)
    if 'first_training_date' in df.columns:
        df['first_training_date'] = pd.to_datetime(df['first_training_date'], errors='coerce')
        df['training_month'] = df['first_training_date'].dt.month
        df['training_dayofweek'] = df['first_training_date'].dt.dayofweek
    if 'num_repeat_trainings' in df.columns and 'num_total_trainings' in df.columns:
        df['repeat_ratio'] = df['num_repeat_trainings'] / (df['num_total_trainings'] + 1)
    
    # Impute and encode
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    for col in cat_cols:
        if col not in ['ID', 'topics_list']:  # Skip ID and processed topics
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Robust drop: Only drop columns that exist
    columns_to_drop = ['ID']
    if 'topics_list' in df.columns:
        columns_to_drop.append('topics_list')
    if 'first_training_date' in df.columns:
        columns_to_drop.append('first_training_date')
    df = df.drop(columns_to_drop, axis=1, errors='ignore')  # errors='ignore' prevents KeyError
    
    for t in targets:
        if t in df.columns:
            df = df.drop(t, axis=1)
    
    return df

train_processed = preprocess(train.copy())
test_processed = preprocess(test.copy())

# Train models with tuning
models = {}
predictions = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target in targets:
    print(f"Training for {target}")
    X = train_processed
    y = train[target]
    
    # Hyperparameter tuning
    param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100, 200]
    }
    xgb = XGBClassifier(scale_pos_weight=(len(y) - sum(y)) / sum(y), random_state=42)
    grid_search = GridSearchCV(xgb, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X, y)
    
    model = grid_search.best_estimator_
    print(f"  Best params: {grid_search.best_params_}")
    
    # CV scores
    cv_scores = []
    for train_idx, val_idx in skf.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
        auc = roc_auc_score(y.iloc[val_idx], y_pred_proba)
        cv_scores.append(auc)
    print(f"  CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    # Final fit
    model.fit(X, y)
    models[target] = model
    predictions[target] = model.predict_proba(test_processed)[:, 1]

# Prepare submission
sub = sample_sub.copy()
sub['Target_07_AUC'] = predictions['adopted_within_07_days']
sub['Target_07_LogLoss'] = predictions['adopted_within_07_days']
sub['Target_90_AUC'] = predictions['adopted_within_90_days']
sub['Target_90_LogLoss'] = predictions['adopted_within_90_days']
sub['Target_120_AUC'] = predictions['adopted_within_120_days']
sub['Target_120_LogLoss'] = predictions['adopted_within_120_days']

# Save submission
sub.to_csv('submission.csv', index=False)
print("Submission saved as submission.csv")
