import os, re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = './data'
TARGETS = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']

# --------------------- HELPERS ---------------------
def parse_list(s):
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return []
    s = re.sub(r'[\[\]\'\"]', '', s)
    return [item.strip() for item in s.split(',') if item.strip()]

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, 'Train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
    prior = pd.read_csv(os.path.join(DATA_DIR, 'Prior.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'SampleSubmission.csv'))
    return train, test, prior, sample

# --------------------- PRIOR FEATURES ---------------------
def add_prior_features(train, test, prior):
    for df in [train, test, prior]:
        if 'training_day' in df.columns:
            df['training_day'] = pd.to_datetime(df['training_day'], errors='coerce')

        if 'topics_list' in df.columns:
            df['topics_parsed'] = df['topics_list'].apply(parse_list)
        if 'trainer' in df.columns:
            df['trainer_parsed'] = df['trainer'].apply(parse_list)

    prior_hist = prior[['farmer_name', 'training_day',
                        'adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days',
                        'has_topic_trained_on']].copy()

    for name, df in [('train', train), ('test', test)]:
        curr = df[['ID', 'farmer_name', 'training_day']].copy()
        merged = curr.merge(prior_hist, on='farmer_name', how='left', suffixes=('', '_prior'))
        past = merged[merged['training_day_prior'] < merged['training_day']]
        agg = past.groupby('ID').agg(
            num_prior=('training_day_prior', 'count'),
            prior_adopt_07=('adopted_within_07_days', 'mean'),
            prior_adopt_90=('adopted_within_90_days', 'mean'),
            prior_adopt_120=('adopted_within_120_days', 'mean'),
            prior_has_topic=('has_topic_trained_on', 'mean'),
            last_prior=('training_day_prior', 'max')
        ).reset_index()
        agg = agg.merge(curr[['ID', 'training_day']], on='ID')
        agg['days_since_last_prior'] = (agg['training_day'] - agg['last_prior']).dt.days
        agg['log_days_since_last_prior'] = np.log1p(agg['days_since_last_prior'])
        agg = agg.drop(columns=['last_prior', 'training_day', 'days_since_last_prior'])
        df = df.merge(agg, on='ID', how='left')

        for c in ['num_prior', 'prior_adopt_07', 'prior_adopt_90', 'prior_adopt_120', 'prior_has_topic', 'log_days_since_last_prior']:
            df[c] = df[c].fillna(0)

        if name == 'train':
            train = df
        else:
            test = df

    # Trainer stats
    trainer_stats = prior.groupby('trainer').agg({
        'adopted_within_07_days': 'mean',
        'adopted_within_90_days': 'mean',
        'adopted_within_120_days': 'mean'
    }).rename(columns={
        'adopted_within_07_days': 'trainer_07_rate',
        'adopted_within_90_days': 'trainer_90_rate',
        'adopted_within_120_days': 'trainer_120_rate'
    })
    for df in [train, test]:
        for days in ['07', '90', '120']:
            col = f'trainer_{days}_rate'
            df[col] = df['trainer_parsed'].apply(
                lambda trainers: np.mean([trainer_stats.loc[t, col] for t in trainers if t in trainer_stats.index]) if trainers else 0
            )

    # Topic stats
    for tgt in TARGETS:
        exploded = prior.explode('topics_parsed')
        topic_stats = exploded.groupby('topics_parsed')[tgt].mean().to_dict()
        period = tgt.split('_')[-2]
        col = f'topic_{period}_rate'
        for df in [train, test]:
            df[col] = df['topics_parsed'].apply(
                lambda topics: np.mean([topic_stats.get(t,0) for t in topics]) if topics else 0
            )
            # Trainer-topic interaction
            df[f'trainer_topic_{period}'] = df[col] * df[f'trainer_{period}_rate']

    return train, test

# --------------------- PREPROCESS ---------------------
def preprocess(train, test):
    for df in [train, test]:
        df['topics_text'] = df['topics_parsed'].apply(lambda x: ' '.join(x) if x else '')

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=150, ngram_range=(1,2), stop_words='english', min_df=3)
    tf_train = vec.fit_transform(train['topics_text'])
    tf_test = vec.transform(test['topics_text'])
    tf_cols = [f'tfidf_{i}' for i in range(tf_train.shape[1])]

    train = pd.concat([train, pd.DataFrame(tf_train.toarray(), columns=tf_cols, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame(tf_test.toarray(), columns=tf_cols, index=test.index)], axis=1)

    # TFIDF aggregates
    for df in [train, test]:
        tf_cols_df = [c for c in df.columns if c.startswith('tfidf_')]
        df['tfidf_mean'] = df[tf_cols_df].mean(axis=1)
        df['tfidf_std'] = df[tf_cols_df].std(axis=1)
        df['tfidf_sum'] = df[tf_cols_df].sum(axis=1)

    # Date features
    for df in [train, test]:
        if 'training_day' in df.columns:
            dt = df['training_day']
            df['month'] = dt.dt.month
            df['dow'] = dt.dt.dayofweek
            df['year'] = dt.dt.year

    # Frequency encoding with log
    for col in ['county', 'subcounty', 'ward']:
        if col in train.columns:
            freq = np.log1p(train[col].value_counts(normalize=True))
            train[f'{col}_freq'] = train[col].map(freq)
            test[f'{col}_freq'] = test[col].map(freq).fillna(0)

    # Label encoding
    cat_cols = ['gender','age','registration','group_name','belong_to_cooperative','has_topic_trained_on']
    for col in cat_cols:
        if col in train.columns:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]]).astype(str).fillna('missing')
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str).fillna('missing'))
            test[col] = le.transform(test[col].astype(str).fillna('missing'))

    # Drop non-numeric
    drop = ['ID','farmer_name','training_day','trainer','topics_list','topics_parsed','trainer_parsed','topics_text',
            'county','subcounty','ward']
    train_proc = train.drop(columns=[c for c in drop if c in train.columns], errors='ignore')
    test_proc = test.drop(columns=[c for c in drop if c in test.columns], errors='ignore')

    # Remove targets from features
    for tgt in TARGETS:
        if tgt in train_proc.columns:
            train_proc = train_proc.drop(columns=[tgt])

    # Align
    common_cols = train_proc.columns.intersection(test_proc.columns)
    train_proc = train_proc[common_cols]
    test_proc  = test_proc[common_cols]

    # Impute
    imputer = SimpleImputer(strategy='median')
    train_proc = pd.DataFrame(imputer.fit_transform(train_proc), columns=train_proc.columns)
    test_proc  = pd.DataFrame(imputer.transform(test_proc), columns=test_proc.columns)

    return train_proc, test_proc

# --------------------- STACKING ---------------------
def train_stack_model(train_proc, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((train_proc.shape[0], 3)) # 3 base models
    base_models = [
        XGBClassifier(n_estimators=800, max_depth=7, learning_rate=0.03,
                      subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
                      eval_metric='logloss', random_state=42, tree_method='hist'),
        LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=64,
                       subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42),
        CatBoostClassifier(iterations=800, learning_rate=0.03, depth=7,
                           verbose=0, random_state=42)
    ]

    for i, model in enumerate(base_models):
        oof_fold = np.zeros(train_proc.shape[0])
        for tr, val in skf.split(train_proc, y):
            model.fit(train_proc.iloc[tr], y.iloc[tr])
            oof_fold[val] = model.predict_proba(train_proc.iloc[val])[:,1]
        oof_preds[:, i] = oof_fold

    meta = LogisticRegression()
    meta.fit(oof_preds, y)
    return base_models, meta

def predict_stack(base_models, meta, test_proc):
    base_preds = np.column_stack([m.predict_proba(test_proc)[:,1] for m in base_models])
    final_pred = meta.predict_proba(base_preds)[:,1]
    return np.clip(final_pred, 0.01, 0.99)

# --------------------- MAIN ---------------------
def main():
    train, test, prior, sample = load_data()
    print("Building prior features...")
    train, test = add_prior_features(train, test, prior)

    print("Preprocessing...")
    train_proc, test_proc = preprocess(train, test)
    print(f"Final features: {train_proc.shape[1]}")

    submission_preds = {}
    for tgt in TARGETS:
        print(f"Training {tgt} with stacking...")
        y = train[tgt]
        base_models, meta_model = train_stack_model(train_proc, y)
        submission_preds[tgt] = predict_stack(base_models, meta_model, test_proc)

    sub = sample[['ID']].copy()
    sub['Target_07_AUC'] = submission_preds['adopted_within_07_days']
    sub['Target_07_LogLoss'] = submission_preds['adopted_within_07_days']
    sub['Target_90_AUC'] = submission_preds['adopted_within_90_days']
    sub['Target_90_LogLoss'] = submission_preds['adopted_within_90_days']
    sub['Target_120_AUC'] = submission_preds['adopted_within_120_days']
    sub['Target_120_LogLoss'] = submission_preds['adopted_within_120_days']

    sub.to_csv('submission_final.csv', index=False)
    print("\nsubmission_final.csv saved → submit this now!")

if __name__ == "__main__":
    main()
