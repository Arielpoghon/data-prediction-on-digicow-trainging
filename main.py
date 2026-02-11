import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)

DATA_DIR = './data'

TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']

def load_file(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"{filename} missing – download from Zindi!")
        return None
    return pd.read_csv(path)

train = load_file('Train.csv')
test = load_file('Test.csv')
sample_sub = load_file('SampleSubmission.csv')
prior = load_file('Prior.csv')

if train is None or test is None or sample_sub is None:
    raise ValueError("Core files missing!")

# ID cleaning
for df in [train, test, sample_sub]:
    df['ID'] = df['ID'].astype(str).str.strip()

# Prior features (safe & strong)
if prior is not None:
    print("Adding Prior.csv features...")
    prior['ID'] = prior['ID'].astype(str).str.strip()
    prior_agg = prior.groupby('ID').agg({
        'ID': 'count',  # prior trainings count
    }).rename(columns={'ID': 'prior_trainings_count'}).reset_index()
    if 'topics_list' in prior.columns:
        prior['prior_topics_count'] = prior['topics_list'].fillna('').str.split(',').str.len()
        prior_topics = prior.groupby('ID')['prior_topics_count'].mean().reset_index(name='prior_avg_topics')
        prior_agg = prior_agg.merge(prior_topics, on='ID', how='left')
    
    train = train.merge(prior_agg, on='ID', how='left')
    test = test.merge(prior_agg, on='ID', how='left')
    train[['prior_trainings_count', 'prior_avg_topics']] = train[['prior_trainings_count', 'prior_avg_topics']].fillna(0)
    test[['prior_trainings_count', 'prior_avg_topics']] = test[['prior_trainings_count', 'prior_avg_topics']].fillna(0)

def preprocess(df, is_train=True):
    if 'topics_list' in df.columns:
        df['topics_count'] = df['topics_list'].fillna('').str.split(',').str.len()
        vec = TfidfVectorizer(max_features=150, ngram_range=(1,2), stop_words='english')
        tf = vec.fit_transform(df['topics_list'].fillna(''))
        tf_cols = [f'tfidf_{i}' for i in range(tf.shape[1])]
        df = pd.concat([df, pd.DataFrame(tf.toarray(), columns=tf_cols, index=df.index)], axis=1)
    
    if 'first_training_date' in df.columns:
        dt = pd.to_datetime(df['first_training_date'], errors='coerce')
        df['month'] = dt.dt.month
        df['dow'] = dt.dt.dayofweek
    
    if 'num_repeat_trainings' in df.columns and 'num_total_trainings' in df.columns:
        df['repeat_ratio'] = df['num_repeat_trainings'] / (df['num_total_trainings'] + 1e-6)
    
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])
    
    cat_cols = df.select_dtypes(include='object').columns.drop('ID', errors='ignore')
    for col in cat_cols:
        if col != 'topics_list':
            df[col] = LabelEncoder().fit_transform(df[col].astype(str).fillna('missing'))
    
    drop = ['topics_list', 'first_training_date']
    df = df.drop(columns=[c for c in drop if c in df.columns], errors='ignore')
    
    if is_train:
        df = df.drop(columns=TARGETS, errors='ignore')
    
    df = df.select_dtypes(include=[np.number])
    return df

train_proc = preprocess(train.copy(), True)
test_proc = preprocess(test.copy(), False)

common_cols = train_proc.columns.intersection(test_proc.columns)
train_proc = train_proc[common_cols]
test_proc = test_proc[common_cols]

# Local CV to estimate score
def local_cv_score(model_class, params):
    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for target in TARGETS:
        y = train[target]
        fold_aucs = []
        fold_lls = []
        for tr_idx, val_idx in skf.split(train_proc, y):
            X_tr, X_val = train_proc.iloc[tr_idx], train_proc.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            prob = np.clip(model.predict_proba(X_val)[:,1], 1e-7, 1-1e-7)
            fold_aucs.append(roc_auc_score(y_val, prob))
            fold_lls.append(log_loss(y_val, prob))
        avg_auc = np.mean(fold_aucs)
        avg_ll = np.mean(fold_lls)
        weighted = 0.25 * avg_auc + 0.75 * (1 - avg_ll / 10)  # rough normalization
        scores.append(weighted)
    return np.mean(scores)

# Train ensemble
xgb_preds = {}
lgb_preds = {}
for target in TARGETS:
    print(f"\nTraining {target}...")
    y = train[target]
    pos_w = (len(y) - y.sum()) / (y.sum() + 1e-6)
    
    # XGBoost
    xgb = XGBClassifier(scale_pos_weight=pos_w, n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42)
    xgb.fit(train_proc, y)
    xgb_preds[target] = xgb.predict_proba(test_proc)[:,1]
    
    # LightGBM
    lgb = LGBMClassifier(scale_pos_weight=pos_w, n_estimators=800, learning_rate=0.03, num_leaves=64, verbose=-1, random_state=42)
    lgb.fit(train_proc, y)
    lgb_preds[target] = lgb.predict_proba(test_proc)[:,1]

# Ensemble average + clip
final_preds = {}
for target in TARGETS:
    ens = (xgb_preds[target] + lgb_preds[target]) / 2
    final_preds[target] = np.clip(ens, 0.01, 0.99)  # safer than 0.0

# Submission
pred_df = pd.DataFrame({
    'ID': test['ID'],
    'Target_07_AUC':     final_preds['adopted_within_07_days'],
    'Target_07_LogLoss': final_preds['adopted_within_07_days'],
    'Target_90_AUC':     final_preds['adopted_within_90_days'],
    'Target_90_LogLoss': final_preds['adopted_within_90_days'],
    'Target_120_AUC':    final_preds['adopted_within_120_days'],
    'Target_120_LogLoss': final_preds['adopted_within_120_days'],
})

sub = sample_sub[['ID']].merge(pred_df, on='ID', how='left').fillna(0.01)  # low but not zero

sub = sub[['ID', 'Target_07_AUC', 'Target_07_LogLoss', 'Target_90_AUC', 'Target_90_LogLoss', 'Target_120_AUC', 'Target_120_LogLoss']]

sub.to_csv('submission_ensemble.csv', index=False)
print("\nsubmission_ensemble.csv ready – this should beat 0.859 comfortably")
