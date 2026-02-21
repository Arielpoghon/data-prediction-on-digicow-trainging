"""
DigiCow - PATCH for original 0.9499 model
==========================================
DO NOT CHANGE the original architecture.
ADD ONLY these two new features to build_features():

1. FARMER HISTORY (fh_*): from Prior by farmer_name
   - fh_ever_07/90/120: has this farmer ever adopted before?
   - fh_mean_07/90/120: their historical adoption rate
   - fh_n: how many prior sessions
   
2. TRAIN SESSION DATA for TEST only:
   - For test: sess_mean = (prior_sum + train_sum) / (prior_n + train_n)
   - For train: keep prior-only session (no CV leakage)
   - This bumps session coverage on test from 86% → same sessions

KEEP EVERYTHING ELSE IDENTICAL:
   - class_weight='balanced' ✓ (DO NOT use sample_weight)
   - isotonic calibration ✓ (DO NOT use sigmoid or temperature)
   - rankdata for AUC column ✓
   - calibrated probs for LogLoss column ✓
   - all existing session/trainer/topic/group features ✓

HOW TO USE:
Copy the functions below and paste into your original main.py,
replacing the corresponding sections.
"""

import re
import numpy as np
import pandas as pd


def parse_list(s):
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return []
    s = re.sub(r"[\[\]'\"]", '', s)
    return [item.strip() for item in s.split(',') if item.strip()]

def parse_first(s):
    parts = parse_list(s)
    return parts[0] if parts else ''

def bsmooth(mean_s, count_s, gm, k=10):
    return (mean_s * count_s + gm * k) / (count_s + k)


def build_features_patched(train, test, prior):
    """
    Drop-in replacement for your build_features() function.
    Adds farmer history and train-session-for-test on top of original features.
    """
    # ── Parse dates and keys (same as original) ──
    for df in [train, test, prior]:
        for dc in ['training_day', 'first_training_date']:
            if dc in df.columns:
                df[dc] = pd.to_datetime(df[dc], errors='coerce')
        if 'topics_list' in df.columns:
            df['topics_parsed'] = df['topics_list'].apply(parse_list)
        if 'trainer' in df.columns:
            df['trainer_parsed'] = df['trainer'].apply(parse_list)

    k = 10
    global07  = prior['adopted_within_07_days'].mean()
    global90  = prior['adopted_within_90_days'].mean()
    global120 = prior['adopted_within_120_days'].mean()

    prior['sk'] = (prior['training_day'].astype(str) + '||' +
                   prior['trainer'].apply(parse_first))
    for df in [train, test]:
        df['sk'] = (df['training_day'].astype(str) + '||' +
                    df['trainer'].apply(parse_first))

    # ══════════════════════════════════════════════════════════════════
    # 1. SESSION PEER ADOPTION — ORIGINAL (prior only for train)
    # ══════════════════════════════════════════════════════════════════
    sess = prior.groupby('sk').agg(
        sess_07_mean  = ('adopted_within_07_days',  'mean'),
        sess_90_mean  = ('adopted_within_90_days',  'mean'),
        sess_120_mean = ('adopted_within_120_days', 'mean'),
        sess_07_sum   = ('adopted_within_07_days',  'sum'),
        sess_90_sum   = ('adopted_within_90_days',  'sum'),
        sess_120_sum  = ('adopted_within_120_days', 'sum'),
        sess_n        = ('adopted_within_07_days',  'count'),
    ).reset_index()
    for col, gm in [('sess_07_mean', global07), ('sess_90_mean', global90),
                    ('sess_120_mean', global120)]:
        sess[col] = bsmooth(sess[col], sess['sess_n'], gm, k)

    # TRAIN: prior-only session (same as original — no leakage)
    train.drop(columns=[c for c in train.columns if c.startswith('sess_')],
               inplace=True, errors='ignore')
    merged_tr = train[['sk']].merge(sess, on='sk', how='left')
    for col in ['sess_07_mean','sess_90_mean','sess_120_mean',
                'sess_07_sum','sess_90_sum','sess_120_sum','sess_n']:
        fill = global07 if '07' in col else global90 if '90' in col else \
               global120 if '120' in col else 0
        train[col] = merged_tr[col].fillna(fill).values

    # ══════════════════════════════════════════════════════════════════
    # 1b. NEW: TEST SESSION — prior + full train data
    #     Test farmers are disjoint from train, so no leakage here
    # ══════════════════════════════════════════════════════════════════
    train_sess_agg = train.groupby('sk').agg(
        tr_07_sum  = ('adopted_within_07_days',  'sum'),
        tr_90_sum  = ('adopted_within_90_days',  'sum'),
        tr_120_sum = ('adopted_within_120_days', 'sum'),
        tr_n       = ('adopted_within_07_days',  'count'),
    ).reset_index()

    test.drop(columns=[c for c in test.columns if c.startswith('sess_')],
              inplace=True, errors='ignore')
    merged_te = test[['sk']].merge(sess, on='sk', how='left')
    merged_te = merged_te.merge(train_sess_agg, on='sk', how='left')

    for col in ['sess_07_sum','sess_90_sum','sess_120_sum','sess_n',
                'tr_07_sum','tr_90_sum','tr_120_sum','tr_n']:
        merged_te[col] = merged_te[col].fillna(0)

    for pfx, gm in [('07', global07), ('90', global90), ('120', global120)]:
        # Combine prior + train session counts
        comb_n   = merged_te['sess_n'] + merged_te['tr_n']
        comb_sum = merged_te[f'sess_{pfx}_sum'] + merged_te[f'tr_{pfx}_sum']
        rate = (comb_sum / comb_n.replace(0, np.nan)).fillna(gm)
        test[f'sess_{pfx}_mean'] = bsmooth(rate, comb_n, gm, k).values
        test[f'sess_{pfx}_sum']  = comb_sum.values
        test[f'sess_{pfx}_n_extra'] = merged_te['tr_n'].values  # new feature

    test['sess_n'] = (merged_te['sess_n'] + merged_te['tr_n']).values

    n_tr_sess = (train['sess_n'] > 0).sum()
    n_te_sess = (test['sess_n'] > 0).sum()
    print(f"  Session matches — train:{n_tr_sess}/{len(train)}  test:{n_te_sess}/{len(test)}")

    # ══════════════════════════════════════════════════════════════════
    # 2. NEW: FARMER HISTORY from Prior by farmer_name
    #    This farmer's OWN past adoption record
    # ══════════════════════════════════════════════════════════════════
    fh = prior.groupby('farmer_name').agg(
        fh_n        = ('training_day', 'count'),
        fh_ever_07  = ('adopted_within_07_days', 'max'),
        fh_ever_90  = ('adopted_within_90_days', 'max'),
        fh_ever_120 = ('adopted_within_120_days', 'max'),
        fh_mean_07  = ('adopted_within_07_days', 'mean'),
        fh_mean_90  = ('adopted_within_90_days', 'mean'),
        fh_mean_120 = ('adopted_within_120_days', 'mean'),
        fh_sum_07   = ('adopted_within_07_days', 'sum'),
        fh_last     = ('training_day', 'max'),
    ).reset_index()

    for df in [train, test]:
        tmp = df[['farmer_name', 'training_day']].merge(fh, on='farmer_name', how='left')
        tmp['fh_days_since'] = (df['training_day'].values -
                                 pd.to_datetime(tmp['fh_last'])).dt.days
        for col in ['fh_n','fh_ever_07','fh_ever_90','fh_ever_120',
                    'fh_mean_07','fh_mean_90','fh_mean_120','fh_sum_07',
                    'fh_days_since']:
            df[col] = tmp[col].fillna(0).values
        df['fh_has_history']    = (df['fh_n'] > 0).astype(int)
        df['fh_log_days_since'] = np.log1p(df['fh_days_since'].clip(0))

        # Interaction: farmer history × session
        for pfx in ['07','90','120']:
            df[f'fh_x_sess_{pfx}'] = df[f'fh_mean_{pfx}'] * df[f'sess_{pfx}_mean']
            df[f'fh_ever_x_sess_{pfx}'] = df[f'fh_ever_{pfx}'] * df[f'sess_{pfx}_mean']

    print(f"  Farmer history — train:{(train['fh_n']>0).sum()}/{len(train)}"
          f"  test:{(test['fh_n']>0).sum()}/{len(test)}")

    # ══════════════════════════════════════════════════════════════════
    # (Keep all original features below — copy from your original code)
    # ══════════════════════════════════════════════════════════════════
    # Paste here from your original build_features():
    # - Section 2: PER-FARMER HISTORY (time-based, keep if you have it)
    # - Section 3: TRAINER adoption rates
    # - Section 4: TOPIC adoption rates
    # - Section 5: GROUP adoption rates
    # Then return train, test as before

    return train, test


# ══════════════════════════════════════════════════════════════════════
# SUMMARY OF CHANGES TO MAKE IN main.py
# ══════════════════════════════════════════════════════════════════════
"""
IN build_features():
  - Replace session feature for TEST with combined prior+train version (above)
  - Add farmer history block (above)
  - Keep everything else identical

IN preprocess():
  - Add new columns to feature list:
    fh_n, fh_ever_07, fh_ever_90, fh_ever_120,
    fh_mean_07, fh_mean_90, fh_mean_120, fh_sum_07,
    fh_has_history, fh_log_days_since,
    fh_x_sess_07, fh_x_sess_90, fh_x_sess_120,
    fh_ever_x_sess_07, fh_ever_x_sess_90, fh_ever_x_sess_120,
    sess_07_n_extra, sess_90_n_extra, sess_120_n_extra

IN train_stack_model(): NO CHANGES
IN predict_stack():     NO CHANGES  
IN main():              NO CHANGES
"""
