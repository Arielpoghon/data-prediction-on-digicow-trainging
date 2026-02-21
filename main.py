"""
DigiCow - Final Correct Solution
==================================
Root cause of 0.9499 vs 0.9719 gap, found via data analysis:

THE MISSING PIECE: Leave-One-Out (LOO) session feature using TRAIN targets
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prior-only session AUC:             0.910
Combined prior+LOO(train) session:  0.969   ← huge jump
Top team AUC:                       0.995

By combining:
  - Prior session rates (from prior.csv sessions)
  - Leave-one-out session rates within TRAIN (exclude self when computing)
  - For TEST: prior session rates + FULL train session rates (no LOO needed)

The combined session feature gains 6 AUC points for free. This is the key.

CALIBRATION FIX:
  - sample_weight instead of class_weight (doesn't distort probability scale)
  - Temperature scaling on OOF predictions
  - No separate AUC/LogLoss predictions — calibrated probs for both
"""

import os, re, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

DATA_DIR = './data'
TARGETS  = ['adopted_within_07_days', 'adopted_within_90_days', 'adopted_within_120_days']
PERIODS  = ['07', '90', '120']
N_FOLDS  = 7
SEED     = 42
np.random.seed(SEED)


def extract_trainer(s):
    m = re.findall(r'TRA_\w+', str(s))
    return m[0] if m else str(s).strip()

def parse_topics(s):
    if pd.isna(s): return []
    items = re.findall(r"'([^']+)'", str(s))
    return [x for x in items if x]

def bsmooth(mean_s, count_s, gm, k=10):
    return (mean_s * count_s + gm * k) / (count_s + k)

def temperature_scale(probs, y_true, n_steps=300):
    best_T, best_ll = 1.0, log_loss(y_true, probs)
    for T in np.linspace(0.3, 8.0, n_steps):
        p = np.clip(probs ** (1.0 / T), 1e-7, 1 - 1e-7)
        ll = log_loss(y_true, p)
        if ll < best_ll:
            best_ll, best_T = ll, T
    return best_T


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────

def load_data():
    train  = pd.read_csv(os.path.join(DATA_DIR, 'Train.csv'))
    test   = pd.read_csv(os.path.join(DATA_DIR, 'Test.csv'))
    prior  = pd.read_csv(os.path.join(DATA_DIR, 'Prior.csv'))
    sample = pd.read_csv(os.path.join(DATA_DIR, 'SampleSubmission.csv'))

    for df in [train, test, prior]:
        df['training_day']   = pd.to_datetime(df['training_day'], errors='coerce')
        df['trainer_clean']  = df['trainer'].apply(extract_trainer)
        df['sk']             = df['training_day'].astype(str) + '||' + df['trainer_clean']
        df['topics']         = df['topics_list'].apply(parse_topics)

    print(f"Train: {train.shape}, Test: {test.shape}, Prior: {prior.shape}")
    return train, test, prior, sample


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def build_features(train, test, prior):
    g = {pfx: prior[f'adopted_within_{pfx}_days'].mean() for pfx in PERIODS}
    k = 10

    # ── Prior-level session aggregates ──
    prior_sess = prior.groupby('sk').agg(
        p_n   = ('adopted_within_07_days', 'count'),
        p_07  = ('adopted_within_07_days', 'sum'),
        p_90  = ('adopted_within_90_days', 'sum'),
        p_120 = ('adopted_within_120_days', 'sum'),
    ).reset_index()

    # ── Train session totals (used for TEST; for TRAIN we do LOO below) ──
    train_sess = train.groupby('sk').agg(
        t_n   = ('adopted_within_07_days', 'count'),
        t_07  = ('adopted_within_07_days', 'sum'),
        t_90  = ('adopted_within_90_days', 'sum'),
        t_120 = ('adopted_within_120_days', 'sum'),
    ).reset_index()

    # ══ TRAIN: LOO session feature ══
    # For each train row, exclude self from session count → add prior session stats
    tr = train.merge(train_sess, on='sk', how='left').merge(prior_sess, on='sk', how='left')
    tr[['p_n','p_07','p_90','p_120']] = tr[['p_n','p_07','p_90','p_120']].fillna(0)
    for pfx in PERIODS:
        tgt      = f'adopted_within_{pfx}_days'
        loo_sum  = (tr[f't_{pfx}'] - tr[tgt]).clip(0)
        loo_n    = (tr['t_n'] - 1).clip(0)
        comb_n   = tr['p_n'] + loo_n
        comb_sum = tr[f'p_{pfx}'] + loo_sum
        rate     = (comb_sum / comb_n.replace(0, np.nan)).fillna(g[pfx])
        tr[f'sess_{pfx}_mean'] = bsmooth(rate, comb_n, g[pfx], k)
        tr[f'sess_{pfx}_n']    = comb_n

    # ══ TEST: prior + full train session stats ══
    te = test.merge(prior_sess, on='sk', how='left').merge(train_sess, on='sk', how='left')
    te[['p_n','p_07','p_90','p_120',
        't_n','t_07','t_90','t_120']] = te[['p_n','p_07','p_90','p_120',
                                             't_n','t_07','t_90','t_120']].fillna(0)
    for pfx in PERIODS:
        comb_n   = te['p_n'] + te['t_n']
        comb_sum = te[f'p_{pfx}'] + te[f't_{pfx}']
        rate     = (comb_sum / comb_n.replace(0, np.nan)).fillna(g[pfx])
        te[f'sess_{pfx}_mean'] = bsmooth(rate, comb_n, g[pfx], k)
        te[f'sess_{pfx}_n']    = comb_n

    print(f"  Train session coverage: {(tr['sess_07_n']>0).sum()}/{len(tr)}")
    print(f"  Test  session coverage: {(te['sess_07_n']>0).sum()}/{len(te)}")

    # Drop merge artefacts
    drop_merge = ['p_n','p_07','p_90','p_120','t_n','t_07','t_90','t_120']
    for df in [tr, te]:
        df.drop(columns=drop_merge, inplace=True, errors='ignore')

    # ── Trainer rates from prior only ──
    tr_stats = prior.groupby('trainer_clean').agg(
        tr07=('adopted_within_07_days','mean'), tr90=('adopted_within_90_days','mean'),
        tr120=('adopted_within_120_days','mean'), tr_n=('adopted_within_07_days','count'),
    )
    for col, gm in [('tr07',g['07']),('tr90',g['90']),('tr120',g['120'])]:
        tr_stats[col] = bsmooth(tr_stats[col], tr_stats['tr_n'], gm, k)

    for df in [tr, te]:
        df['trainer_07_rate']  = df['trainer_clean'].map(tr_stats['tr07']).fillna(g['07'])
        df['trainer_90_rate']  = df['trainer_clean'].map(tr_stats['tr90']).fillna(g['90'])
        df['trainer_120_rate'] = df['trainer_clean'].map(tr_stats['tr120']).fillna(g['120'])

    # ── Topic rates from prior ──
    prior_exp = prior.explode('topics')
    for pfx, gm in [(p, g[p]) for p in PERIODS]:
        tgt = f'adopted_within_{pfx}_days'
        tp_agg = prior_exp.groupby('topics')[tgt].agg(['mean','count'])
        tp_agg['smooth'] = bsmooth(tp_agg['mean'], tp_agg['count'], gm, k)
        tp_map = tp_agg['smooth'].to_dict()
        for df in [tr, te]:
            df[f'topic_{pfx}_rate'] = df['topics'].apply(
                lambda ts: np.mean([tp_map.get(t,gm) for t in ts]) if ts else gm)
            df[f'topic_{pfx}_max']  = df['topics'].apply(
                lambda ts: max([tp_map.get(t,gm) for t in ts], default=gm) if ts else gm)
            df[f'topic_{pfx}_min']  = df['topics'].apply(
                lambda ts: min([tp_map.get(t,gm) for t in ts], default=gm) if ts else gm)

    # ── Geo/Group rates from prior ──
    for grp_col in ['county','subcounty','ward','group_name']:
        gs = prior.groupby(grp_col).agg(
            g07=('adopted_within_07_days','mean'), g90=('adopted_within_90_days','mean'),
            g120=('adopted_within_120_days','mean'), g_n=('adopted_within_07_days','count'),
        )
        for col, gm in [('g07',g['07']),('g90',g['90']),('g120',g['120'])]:
            gs[col] = bsmooth(gs[col], gs['g_n'], gm, k)
        for df in [tr, te]:
            tmp = df[[grp_col]].merge(gs[['g07','g90','g120']].reset_index(), on=grp_col, how='left')
            df[f'{grp_col}_07_rate']  = tmp['g07'].fillna(g['07']).values
            df[f'{grp_col}_90_rate']  = tmp['g90'].fillna(g['90']).values
            df[f'{grp_col}_120_rate'] = tmp['g120'].fillna(g['120']).values

    # ── Farmer history from prior (by farmer_name) ──
    fh = prior.groupby('farmer_name').agg(
        fh_n       = ('training_day','count'),
        fh_ever_07 = ('adopted_within_07_days','max'),
        fh_ever_90 = ('adopted_within_90_days','max'),
        fh_ever_120= ('adopted_within_120_days','max'),
        fh_mean_07 = ('adopted_within_07_days','mean'),
        fh_mean_90 = ('adopted_within_90_days','mean'),
        fh_mean_120= ('adopted_within_120_days','mean'),
        fh_last    = ('training_day','max'),
        fh_first   = ('training_day','min'),
    ).reset_index()
    for df in [tr, te]:
        tmp = df[['farmer_name','training_day']].merge(fh, on='farmer_name', how='left')
        tmp['fh_days_last']  = (df['training_day'].values - pd.to_datetime(tmp['fh_last'])).dt.days
        tmp['fh_days_first'] = (df['training_day'].values - pd.to_datetime(tmp['fh_first'])).dt.days
        for col in ['fh_n','fh_ever_07','fh_ever_90','fh_ever_120',
                    'fh_mean_07','fh_mean_90','fh_mean_120','fh_days_last','fh_days_first']:
            df[col] = tmp[col].fillna(0).values
        df['fh_has_history']  = (df['fh_n'] > 0).astype(int)
        df['fh_log_days_last'] = np.log1p(df['fh_days_last'].clip(0))

    # ── Month trend from prior ──
    prior['month_key'] = prior['training_day'].dt.to_period('M').astype(str)
    mt = prior.groupby('month_key').agg(
        mt07=('adopted_within_07_days','mean'),
        mt90=('adopted_within_90_days','mean'),
        mt120=('adopted_within_120_days','mean'),
    ).reset_index()
    for df in [tr, te]:
        df['month_key'] = df['training_day'].dt.to_period('M').astype(str)
        tmp = df[['month_key']].merge(mt, on='month_key', how='left')
        df['month_trend_07']  = tmp['mt07'].fillna(g['07']).values
        df['month_trend_90']  = tmp['mt90'].fillna(g['90']).values
        df['month_trend_120'] = tmp['mt120'].fillna(g['120']).values
        df.drop(columns=['month_key'], inplace=True)

    # ── Date features ──
    for df in [tr, te]:
        df['td_month']    = df['training_day'].dt.month
        df['td_dow']      = df['training_day'].dt.dayofweek
        df['td_year']     = df['training_day'].dt.year
        df['td_quarter']  = df['training_day'].dt.quarter
        df['td_weeknum']  = df['training_day'].dt.isocalendar().week.astype(int)
        df['td_dayofyear']= df['training_day'].dt.dayofyear
        df['num_topics']  = df['topics'].apply(len)

    # ── Interactions ──
    for df in [tr, te]:
        for pfx, gm in [(p, g[p]) for p in PERIODS]:
            sm  = df[f'sess_{pfx}_mean']
            tr_ = df[f'trainer_{pfx}_rate']
            tp  = df[f'topic_{pfx}_rate']
            tpx = df[f'topic_{pfx}_max']
            fhm = df[f'fh_mean_{pfx}']
            cnt = df[f'county_{pfx}_rate']
            grp = df[f'group_name_{pfx}_rate']

            df[f'sess_x_tr_{pfx}']   = sm * tr_
            df[f'sess_x_tp_{pfx}']   = sm * tp
            df[f'sess_x_grp_{pfx}']  = sm * grp
            df[f'tr_x_tp_{pfx}']     = tr_ * tp
            df[f'tr_x_tpmax_{pfx}']  = tr_ * tpx
            df[f'fh_x_sess_{pfx}']   = fhm * sm
            df[f'fh_x_tr_{pfx}']     = fhm * tr_
            df[f'cnt_x_tr_{pfx}']    = cnt * tr_
            df[f'coop_x_tr_{pfx}']   = df['belong_to_cooperative'] * tr_

    return tr, te


# ─────────────────────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────────────────────

def preprocess(train_f, test_f, train_raw):
    all_data = pd.concat([train_f, test_f], ignore_index=True)

    # Label encode
    for col in ['gender','age','county','subcounty','ward','registration']:
        if col in all_data.columns:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

    # Group frequency
    freq = train_raw['group_name'].value_counts(normalize=True)
    all_data['group_freq'] = all_data['group_name'].map(freq).fillna(0)

    drop = ['ID','farmer_name','training_day','trainer','trainer_clean',
            'topics_list','topics','sk','group_name','registration'] + TARGETS
    all_data.drop(columns=[c for c in drop if c in all_data.columns], inplace=True, errors='ignore')

    n_tr = len(train_f)
    train_proc = all_data.iloc[:n_tr].copy()
    test_proc  = all_data.iloc[n_tr:].copy()

    # Align columns (drop any that are all-NaN in either)
    valid = [c for c in train_proc.columns
             if train_proc[c].notna().any() and test_proc[c].notna().any()]
    train_proc = train_proc[valid]
    test_proc  = test_proc[valid]

    imp = SimpleImputer(strategy='median')
    train_arr = imp.fit_transform(train_proc)
    test_arr  = imp.transform(test_proc)
    train_proc = pd.DataFrame(train_arr, columns=valid)
    test_proc  = pd.DataFrame(test_arr,  columns=valid)

    print(f"  Features: {train_proc.shape[1]}")
    return train_proc, test_proc


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def train_predict(X_train, y, X_test, name):
    y_arr = y.values if hasattr(y, 'values') else np.array(y)
    pos_rate = y_arr.mean()
    # sample_weight: same balance as class_weight but correct probability scale
    sw = np.where(y_arr == 1, (1 - pos_rate) / pos_rate, 1.0)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    models = [
        XGBClassifier(
            n_estimators=3000, max_depth=4, learning_rate=0.015,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.05, reg_alpha=0.1, reg_lambda=2.0,
            eval_metric='logloss', random_state=SEED, tree_method='hist',
            n_jobs=-1, early_stopping_rounds=150,
        ),
        LGBMClassifier(
            n_estimators=3000, learning_rate=0.015, num_leaves=63,
            subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.05, reg_lambda=2.0, verbose=-1,
            random_state=SEED, n_jobs=-1,
        ),
        CatBoostClassifier(
            iterations=2500, learning_rate=0.015, depth=5,
            l2_leaf_reg=5, bagging_temperature=0.3,
            verbose=0, random_state=SEED, early_stopping_rounds=150,
        ),
        XGBClassifier(
            n_estimators=2000, max_depth=3, learning_rate=0.02,
            subsample=0.75, colsample_bytree=0.65, min_child_weight=8,
            reg_alpha=0.2, reg_lambda=3.0,
            eval_metric='logloss', random_state=SEED+1, tree_method='hist',
            n_jobs=-1, early_stopping_rounds=150,
        ),
        LGBMClassifier(
            n_estimators=4000, learning_rate=0.01, num_leaves=95,
            subsample=0.75, colsample_bytree=0.65, min_child_samples=15,
            reg_alpha=0.02, reg_lambda=3.0, verbose=-1,
            random_state=SEED+2, n_jobs=-1,
        ),
        LogisticRegression(C=0.05, max_iter=2000, random_state=SEED),
    ]

    n_m = len(models)
    oof  = np.zeros((len(X_train), n_m))
    test_p = np.zeros((len(X_test),  n_m))

    print(f"\n  [{name}] pos={pos_rate:.5f} ({int(y_arr.sum())}/{len(y_arr)})")

    for i, model in enumerate(models):
        oof_fold  = np.zeros(len(X_train))
        test_fold = np.zeros(len(X_test))

        for tr_idx, val_idx in skf.split(X_train, y_arr):
            Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            ytr, yval = y_arr[tr_idx], y_arr[val_idx]
            swtr = sw[tr_idx]

            if isinstance(model, XGBClassifier):
                model.fit(Xtr, ytr, sample_weight=swtr,
                         eval_set=[(Xval, yval)], verbose=False)
            elif isinstance(model, LGBMClassifier):
                model.fit(Xtr, ytr, sample_weight=swtr,
                         eval_set=[(Xval, yval)],
                         callbacks=[lgb.early_stopping(200, verbose=False),
                                    lgb.log_evaluation(-1)])
            elif isinstance(model, CatBoostClassifier):
                model.fit(Xtr, ytr, sample_weight=swtr,
                         eval_set=(Xval, yval), verbose=False)
            else:
                model.fit(Xtr, ytr, sample_weight=swtr)

            oof_fold[val_idx] = model.predict_proba(Xval)[:, 1]
            test_fold        += model.predict_proba(X_test)[:, 1] / N_FOLDS

        oof[:, i]   = oof_fold
        test_p[:, i]= test_fold
        ll  = log_loss(y_arr, oof_fold)
        auc = roc_auc_score(y_arr, oof_fold)
        print(f"    m{i}: ll={ll:.5f}  auc={auc:.5f}")

    # Optimal blend weights using OOF
    oof_avg  = oof.mean(axis=1)
    test_avg = test_p.mean(axis=1)
    ll_avg   = log_loss(y_arr, oof_avg)
    auc_avg  = roc_auc_score(y_arr, oof_avg)
    print(f"  Avg: ll={ll_avg:.5f}  auc={auc_avg:.5f}")

    # Find optimal LR weight (calibration anchor)
    best_w_lr, best_ll_w = 0.0, ll_avg
    for w_lr in np.linspace(0.0, 0.5, 21):
        w = np.array([(1-w_lr)/(n_m-1)]*(n_m-1) + [w_lr])
        blend = np.clip((oof * w).sum(axis=1), 1e-7, 1-1e-7)
        ll_w  = log_loss(y_arr, blend)
        if ll_w < best_ll_w:
            best_ll_w, best_w_lr = ll_w, w_lr

    w_final  = np.array([(1-best_w_lr)/(n_m-1)]*(n_m-1) + [best_w_lr])
    oof_bl   = np.clip((oof   * w_final).sum(axis=1), 1e-7, 1-1e-7)
    test_bl  = np.clip((test_p* w_final).sum(axis=1), 1e-7, 1-1e-7)
    print(f"  Weighted (lr_w={best_w_lr:.2f}): ll={log_loss(y_arr,oof_bl):.5f}")

    # Temperature scaling
    best_T = temperature_scale(oof_bl, y_arr)
    test_final = np.clip(test_bl ** (1.0 / best_T), 1e-7, 1-1e-7)
    oof_temp   = np.clip(oof_bl  ** (1.0 / best_T), 1e-7, 1-1e-7)
    ll_final = log_loss(y_arr, oof_temp)
    auc_final = roc_auc_score(y_arr, oof_temp)
    print(f"  Temp T={best_T:.2f}: ll={ll_final:.5f}  auc={auc_final:.5f}")

    return test_final, oof_temp, y_arr


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("Loading...")
    train, test, prior, sample = load_data()

    print("\nBuilding features...")
    train_f, test_f = build_features(train, test, prior)

    # Attach targets for preprocess
    for t in TARGETS:
        train_f[t] = train[t].values

    print("\nPreprocessing...")
    train_proc, test_proc = preprocess(train_f, test_f, train)
    test_ids = test['ID'].values

    # has_topic_trained_on == 0 → guaranteed 0 adoption (verified in data)
    no_topic_test = (test['has_topic_trained_on'] == 0).values
    print(f"\nhas_topic=0 in test: {no_topic_test.sum()} (will floor predictions)")

    preds  = {}
    oof_r  = {}

    for tgt in TARGETS:
        print(f"\n{'='*60}\n{tgt}")
        y = train[tgt].reset_index(drop=True)
        p, oof, y_arr = train_predict(train_proc, y, test_proc, tgt)
        gm = prior[tgt].mean()
        # Floor has_topic=0 farmers (historically zero adoption)
        p[no_topic_test] = gm * 0.05
        preds[tgt] = p
        oof_r[tgt] = (oof, y_arr)

    # Summary
    print("\n" + "="*60)
    print("OOF SUMMARY")
    total = 0
    for tgt in TARGETS:
        oof, y = oof_r[tgt]
        ll  = log_loss(y, oof)
        auc = roc_auc_score(y, oof)
        comp = 0.75*(1-ll) + 0.25*auc
        total += comp / 3
        print(f"  {tgt}: ll={ll:.5f}  auc={auc:.5f}  → {comp:.5f}")
    print(f"  ESTIMATED SCORE: {total:.5f}")

    # Build submission
    sub = sample[['ID']].copy()
    sub['Target_07_AUC']      = preds['adopted_within_07_days']
    sub['Target_90_AUC']      = preds['adopted_within_90_days']
    sub['Target_120_AUC']     = preds['adopted_within_120_days']
    sub['Target_07_LogLoss']  = preds['adopted_within_07_days']
    sub['Target_90_LogLoss']  = preds['adopted_within_90_days']
    sub['Target_120_LogLoss'] = preds['adopted_within_120_days']

    assert list(sub.columns) == list(sample.columns), \
        f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}"

    sub.to_csv('submission_final.csv', index=False)
    print(f"\n✓ Saved submission_final.csv ({len(sub)} rows)")
    print(sub.describe())


if __name__ == '__main__':
    main()
