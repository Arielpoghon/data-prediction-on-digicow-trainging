"""
DigiCow — Tomorrow's final submission
Back to proven 0.9498 foundation + one clean addition:
farmer_name personal history from Prior (correctly computed, no session noise)
"""
import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from scipy.stats import rankdata

warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = './data'
TARGETS  = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
N_FOLDS  = 7
SEED     = 42

def parse_list(s):
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return []
    s = re.sub(r"[\[\]'\"]", '', s)
    return [item.strip() for item in s.split(',') if item.strip()]

def bsmooth(mean_s, count_s, gm, k=10):
    return (mean_s * count_s + gm * k) / (count_s + k)

def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'Train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
    prior  = pd.read_csv(os.path.join(DATA_DIR, 'Prior.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'SampleSubmission.csv'))
    print("Sample columns:", list(sample.columns))
    print(f"Train:{train.shape} Test:{test.shape} Prior:{prior.shape}")
    return train, test, prior, sample

def add_prior_features(train, test, prior):
    for df in [train, test, prior]:
        for dc in ['training_day', 'first_training_date']:
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors='coerce')
        if 'topics_list' in df.columns:
            df['topics_parsed'] = df['topics_list'].apply(parse_list)
        if 'trainer' in df.columns:
            df['trainer_parsed'] = df['trainer'].apply(parse_list)

    k         = 10
    global07  = prior['adopted_within_07_days'].mean()
    global90  = prior['adopted_within_90_days'].mean()
    global120 = prior['adopted_within_120_days'].mean()
    print(f"  Base rates 07:{global07:.4f} 90:{global90:.4f} 120:{global120:.4f}")

    prior_hist = prior[['farmer_name','training_day',
                         'adopted_within_07_days','adopted_within_90_days',
                         'adopted_within_120_days','has_topic_trained_on',
                         'topics_parsed']].copy()

    # ── per-farmer history (ALL prior rows, not just past dates) ─────────────
    # For train: use strictly past dates to avoid leakage
    # For test:  use ALL prior dates (test is May 2025, prior is all 2024 = all past)
    for name, df in [('train', train), ('test', test)]:
        curr   = df[['ID','farmer_name','training_day']].copy()
        merged = curr.merge(prior_hist, on='farmer_name', how='left',
                            suffixes=('','_prior'))

        if name == 'train':
            # strict past only for train
            past = merged[merged['training_day_prior'] < merged['training_day']].copy()
        else:
            # for test (May 2025), ALL prior rows are in the past (prior is 2024)
            past = merged[merged['training_day_prior'] < merged['training_day']].copy()

        agg = past.groupby('ID').agg(
            num_prior           = ('training_day_prior','count'),
            prior_adopt_07      = ('adopted_within_07_days','mean'),
            prior_adopt_90      = ('adopted_within_90_days','mean'),
            prior_adopt_120     = ('adopted_within_120_days','mean'),
            prior_adopt_07_std  = ('adopted_within_07_days','std'),
            prior_adopt_90_std  = ('adopted_within_90_days','std'),
            prior_adopt_120_std = ('adopted_within_120_days','std'),
            prior_adopt_07_sum  = ('adopted_within_07_days','sum'),
            prior_adopt_90_sum  = ('adopted_within_90_days','sum'),
            prior_adopt_120_sum = ('adopted_within_120_days','sum'),
            prior_has_topic     = ('has_topic_trained_on','mean'),
            last_prior          = ('training_day_prior','max'),
            first_prior         = ('training_day_prior','min'),
        ).reset_index()

        # topic repeat ratio
        curr_topics = df[['ID','topics_parsed']].set_index('ID')
        past_topics = past.groupby('ID')['topics_parsed'].apply(
            lambda x: set(t for lst in x for t in lst))
        def topic_overlap(row):
            fid  = row['ID']
            cur  = set(curr_topics.loc[fid,'topics_parsed']) if fid in curr_topics.index else set()
            hist = past_topics.get(fid, set())
            return len(cur & hist) / len(cur) if cur else 0.0
        agg['topic_repeat_ratio'] = agg.apply(topic_overlap, axis=1)

        # last-3-session momentum
        recent  = past.sort_values('training_day_prior').groupby('ID').tail(3)
        rec_agg = recent.groupby('ID').agg(
            recent_adopt_07  = ('adopted_within_07_days','mean'),
            recent_adopt_90  = ('adopted_within_90_days','mean'),
            recent_adopt_120 = ('adopted_within_120_days','mean'),
        ).reset_index()
        agg = agg.merge(rec_agg, on='ID', how='left')

        # time + frequency
        agg = agg.merge(curr[['ID','training_day']], on='ID')
        agg['days_since_last']  = (agg['training_day'] - agg['last_prior']).dt.days
        agg['days_since_first'] = (agg['training_day'] - agg['first_prior']).dt.days
        agg['log_days_since_last']  = np.log1p(agg['days_since_last'])
        agg['log_days_since_first'] = np.log1p(agg['days_since_first'])
        agg['training_freq'] = (
            agg['num_prior'] / (agg['days_since_first'].replace(0,np.nan)/30)
        ).fillna(0)
        agg.drop(columns=['last_prior','first_prior','training_day',
                           'days_since_last','days_since_first'], inplace=True)

        df = df.merge(agg, on='ID', how='left')
        for c in [col for col in agg.columns if col != 'ID']:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        if name == 'train': train = df
        else:               test  = df

    n_tr = (train['num_prior'] > 0).sum()
    n_te = (test['num_prior'] > 0).sum()
    print(f"  Farmers with history — train:{n_tr}/{len(train)}  test:{n_te}/{len(test)}")

    # ── trainer rates (Bayesian-smoothed from Prior) ──────────────────────
    prior['trainer_parsed'] = prior['trainer'].apply(parse_list)
    prior_exp_tr = prior.explode('trainer_parsed')
    tr_stats = prior_exp_tr.groupby('trainer_parsed').agg(
        tr07 = ('adopted_within_07_days','mean'),
        tr90 = ('adopted_within_90_days','mean'),
        tr120= ('adopted_within_120_days','mean'),
        tr_n = ('adopted_within_07_days','count'),
    )
    for col, gm in [('tr07',global07),('tr90',global90),('tr120',global120)]:
        tr_stats[col] = bsmooth(tr_stats[col], tr_stats['tr_n'], gm, k)

    for df in [train, test]:
        for attr, gm, cname in [('tr07',global07,'trainer_07_rate'),
                                  ('tr90',global90,'trainer_90_rate'),
                                  ('tr120',global120,'trainer_120_rate')]:
            df[cname] = df['trainer_parsed'].apply(
                lambda ts: np.mean([tr_stats.loc[t,attr]
                                    for t in ts if t in tr_stats.index]) if ts else gm)
        df['trainer_07_max'] = df['trainer_parsed'].apply(
            lambda ts: max([tr_stats.loc[t,'tr07']
                            for t in ts if t in tr_stats.index], default=global07))

    # ── topic rates (Bayesian-smoothed from Prior) ────────────────────────
    prior['topics_parsed'] = prior['topics_list'].apply(parse_list)
    prior_exp_tp = prior.explode('topics_parsed')

    for tgt in TARGETS:
        period = tgt.split('_')[-2]
        gm     = prior[tgt].mean()
        tp_agg = prior_exp_tp.groupby('topics_parsed')[tgt].agg(['mean','count'])
        tp_agg['smooth'] = bsmooth(tp_agg['mean'], tp_agg['count'], gm, k)
        tp_map = tp_agg['smooth'].to_dict()
        for df in [train, test]:
            df[f'topic_{period}_rate'] = df['topics_parsed'].apply(
                lambda ts: np.mean([tp_map.get(t,gm) for t in ts]) if ts else gm)
            df[f'topic_{period}_max']  = df['topics_parsed'].apply(
                lambda ts: max([tp_map.get(t,gm) for t in ts], default=gm) if ts else gm)
            df[f'trainer_topic_{period}'] = (df[f'topic_{period}_rate'] *
                                              df[f'trainer_{period}_rate'])

    # ── group rates (Bayesian-smoothed from Prior) ────────────────────────
    grp = prior.groupby('group_name').agg(
        g07=('adopted_within_07_days','mean'),
        g90=('adopted_within_90_days','mean'),
        g120=('adopted_within_120_days','mean'),
        g_n=('adopted_within_07_days','count'),
    ).reset_index()
    for col,gm in [('g07',global07),('g90',global90),('g120',global120)]:
        grp[col] = bsmooth(grp[col], grp['g_n'], gm, k)
    grp.rename(columns={'g07':'group_07_rate','g90':'group_90_rate','g120':'group_120_rate'},
               inplace=True)
    for df in [train, test]:
        tmp = df[['group_name']].merge(
            grp[['group_name','group_07_rate','group_90_rate','group_120_rate']],
            on='group_name', how='left')
        df['group_07_rate']  = tmp['group_07_rate'].fillna(global07).values
        df['group_90_rate']  = tmp['group_90_rate'].fillna(global90).values
        df['group_120_rate'] = tmp['group_120_rate'].fillna(global120).values

    return train, test

def preprocess(train, test):
    from sklearn.feature_extraction.text import TfidfVectorizer

    for df in [train, test]:
        df['topics_text']  = df['topics_parsed'].apply(lambda x: ' '.join(x) if x else '')
        df['num_topics']   = df['topics_parsed'].apply(len)
        df['num_trainers'] = df['trainer_parsed'].apply(len)

    vec = TfidfVectorizer(max_features=150, ngram_range=(1,2), stop_words='english', min_df=2)
    tf_tr = vec.fit_transform(train['topics_text'])
    tf_te = vec.transform(test['topics_text'])
    tf_cols = [f'tfidf_{i}' for i in range(tf_tr.shape[1])]
    train = pd.concat([train, pd.DataFrame(tf_tr.toarray(), columns=tf_cols, index=train.index)], axis=1)
    test  = pd.concat([test,  pd.DataFrame(tf_te.toarray(), columns=tf_cols, index=test.index)],  axis=1)
    for df in [train, test]:
        tfc = [c for c in df.columns if c.startswith('tfidf_')]
        df['tfidf_mean'] = df[tfc].mean(axis=1)
        df['tfidf_max']  = df[tfc].max(axis=1)
        df['tfidf_sum']  = df[tfc].sum(axis=1)

    for df in [train, test]:
        for dc, pfx in [('training_day','td'),('first_training_date','ftd')]:
            if dc in df.columns:
                dt = df[dc]
                df[f'{pfx}_month']   = dt.dt.month
                df[f'{pfx}_dow']     = dt.dt.dayofweek
                df[f'{pfx}_year']    = dt.dt.year
                df[f'{pfx}_quarter'] = dt.dt.quarter
                df[f'{pfx}_weeknum'] = dt.dt.isocalendar().week.astype(int)
        if 'training_day' in df.columns and 'first_training_date' in df.columns:
            df['days_since_first_training']     = (df['training_day'] - df['first_training_date']).dt.days.fillna(0)
            df['log_days_since_first_training'] = np.log1p(df['days_since_first_training'])

    for col in ['county','subcounty','ward']:
        if col in train.columns:
            freq = np.log1p(train[col].value_counts(normalize=True))
            train[f'{col}_freq'] = train[col].map(freq).fillna(0)
            test[f'{col}_freq']  = test[col].map(freq).fillna(0)

    for col in ['gender','age','registration','group_name',
                'belong_to_cooperative','has_topic_trained_on']:
        if col in train.columns:
            le = LabelEncoder()
            combined = pd.concat([train[col], test[col]]).astype(str).fillna('missing')
            le.fit(combined)
            train[col] = le.transform(train[col].astype(str).fillna('missing'))
            test[col]  = le.transform(test[col].astype(str).fillna('missing'))

    for df in [train, test]:
        for period in ['07','90','120']:
            pa  = df.get(f'prior_adopt_{period}', pd.Series(0, index=df.index))
            tr  = df.get(f'trainer_{period}_rate', pd.Series(0, index=df.index))
            tp  = df.get(f'topic_{period}_rate', pd.Series(0, index=df.index))
            tpx = df.get(f'topic_{period}_max', pd.Series(0, index=df.index))
            grp = df.get(f'group_{period}_rate', pd.Series(0, index=df.index))
            df[f'age_prior_{period}']         = df['age'] * pa
            df[f'gender_topic_{period}']      = df['gender'] * tp
            df[f'num_prior_topic_{period}']   = df['num_prior'] * tp
            df[f'trainer_topic_max_{period}'] = tr * tpx
            df[f'group_topic_{period}']       = grp * tp
            df[f'group_x_trainer_{period}']   = grp * tr
        df['num_prior_x_topics']  = df['num_prior'] * df['num_topics']
        df['is_repeat_topic']     = (df.get('topic_repeat_ratio',0) > 0).astype(int)
        df['adopt_consistency']   = df.get('prior_adopt_90',0) - df.get('prior_adopt_07',0)
        df['adopt_accel']         = df.get('prior_adopt_120',0) - df.get('prior_adopt_90',0)

    drop = ['ID','farmer_name','training_day','first_training_date',
            'trainer','topics_list','topics_parsed','trainer_parsed',
            'topics_text','county','subcounty','ward','group_name']
    train_proc = train.drop(columns=[c for c in drop if c in train.columns], errors='ignore')
    test_proc  = test.drop( columns=[c for c in drop if c in test.columns],  errors='ignore')
    for tgt in TARGETS:
        if tgt in train_proc.columns:
            train_proc = train_proc.drop(columns=[tgt])

    common     = train_proc.columns.intersection(test_proc.columns)
    train_proc = train_proc[common]
    test_proc  = test_proc[common]

    imputer    = SimpleImputer(strategy='median')
    train_proc = pd.DataFrame(imputer.fit_transform(train_proc), columns=train_proc.columns)
    test_proc  = pd.DataFrame(imputer.transform(test_proc),      columns=test_proc.columns)
    print(f"  Feature count: {train_proc.shape[1]}")
    return train_proc, test_proc

def train_stack_model(X, y, tgt_name):
    pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    base_models = [
        XGBClassifier(
            n_estimators=1600, max_depth=5, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7, colsample_bylevel=0.8,
            min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='logloss', random_state=SEED, tree_method='hist',
            scale_pos_weight=pos_weight, n_jobs=-1,
        ),
        LGBMClassifier(
            n_estimators=1800, learning_rate=0.02, num_leaves=50,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
            random_state=SEED, class_weight='balanced', n_jobs=-1,
        ),
        CatBoostClassifier(
            iterations=1600, learning_rate=0.02, depth=6,
            l2_leaf_reg=5, bagging_temperature=0.5,
            verbose=0, random_state=SEED,
            class_weights=[1, float(pos_weight)],
        ),
        XGBClassifier(
            n_estimators=1200, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            min_child_weight=10, reg_alpha=0.5,
            eval_metric='logloss', random_state=SEED+1, tree_method='hist',
            scale_pos_weight=pos_weight, n_jobs=-1,
        ),
        LGBMClassifier(
            n_estimators=2000, learning_rate=0.015, num_leaves=80,
            subsample=0.75, colsample_bytree=0.65, min_child_samples=15,
            reg_alpha=0.05, reg_lambda=2.0, verbose=-1,
            random_state=SEED+2, class_weight='balanced', n_jobs=-1,
        ),
    ]
    n_base    = len(base_models)
    oof_preds = np.zeros((X.shape[0], n_base))

    print(f"  OOF for {tgt_name}  (pos_weight={pos_weight:.1f})")
    for i, model in enumerate(base_models):
        oof_fold = np.zeros(X.shape[0])
        for tr, val in skf.split(X, y):
            model.fit(X.iloc[tr], y.iloc[tr])
            oof_fold[val] = model.predict_proba(X.iloc[val])[:,1]
        oof_preds[:,i] = oof_fold
        print(f"    model {i}: ll={log_loss(y,oof_fold):.4f}  auc={roc_auc_score(y,oof_fold):.4f}")

    meta = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000)
    meta.fit(oof_preds, y)
    meta_probs = meta.predict_proba(oof_preds)[:,1]
    print(f"  META ll={log_loss(y,meta_probs):.4f}  auc={roc_auc_score(y,meta_probs):.4f}")

    calibrator = CalibratedClassifierCV(meta, method='isotonic', cv='prefit')
    calibrator.fit(oof_preds, y)
    cal_probs = calibrator.predict_proba(oof_preds)[:,1]
    print(f"  CALIB ll={log_loss(y,cal_probs):.4f}")

    print("  Refitting on full data...")
    for model in base_models:
        model.fit(X, y)
    return base_models, meta, calibrator

def predict_stack(base_models, meta, calibrator, X_test):
    base_preds = np.column_stack([m.predict_proba(X_test)[:,1] for m in base_models])
    ranks      = np.column_stack([rankdata(base_preds[:,i]) for i in range(base_preds.shape[1])])
    rank_avg   = ranks.mean(axis=1)
    auc_pred     = np.clip(rank_avg / rank_avg.max(), 1e-6, 1-1e-6)
    logloss_pred = np.clip(calibrator.predict_proba(base_preds)[:,1], 1e-6, 1-1e-6)
    return auc_pred, logloss_pred

def main():
    print("Loading data...")
    train, test, prior, sample = load_data()

    print("Building features...")
    train, test = add_prior_features(train, test, prior)

    print("Preprocessing...")
    train_proc, test_proc = preprocess(train, test)

    submission_auc     = {}
    submission_logloss = {}

    for tgt in TARGETS:
        print(f"\n{'='*60}\nTarget: {tgt}\n{'='*60}")
        y = train[tgt].reset_index(drop=True)
        base_models, meta, calibrator = train_stack_model(train_proc, y, tgt)
        auc_pred, ll_pred = predict_stack(base_models, meta, calibrator, test_proc)
        submission_auc[tgt]     = auc_pred
        submission_logloss[tgt] = ll_pred

    sub = sample[['ID']].copy()
    sub['Target_07_AUC']      = submission_auc['adopted_within_07_days']
    sub['Target_90_AUC']      = submission_auc['adopted_within_90_days']
    sub['Target_120_AUC']     = submission_auc['adopted_within_120_days']
    sub['Target_07_LogLoss']  = submission_logloss['adopted_within_07_days']
    sub['Target_90_LogLoss']  = submission_logloss['adopted_within_90_days']
    sub['Target_120_LogLoss'] = submission_logloss['adopted_within_120_days']

    expected = list(sample.columns)
    actual   = list(sub.columns)
    if expected != actual:
        raise ValueError(f"Column mismatch!\n  Expected: {expected}\n  Got: {actual}")

    out_path = 'submission_final.csv'
    sub.to_csv(out_path, index=False)
    print(f"\n✓ Saved {out_path}  ({len(sub)} rows)")
    print(sub.head())

if __name__ == '__main__':
    main()
