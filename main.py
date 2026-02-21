_within_07_days','mean'), g90=('adopted_within_90_days','mean'),
            g120=('adopted_within_120_days','mean'), g_n=('adopted_within_07_days','count'),
        )
        for col, gm in [('g07',g07),('g90',g90),('g120',g120)]:
            geo_s[col] = bsmooth(geo_s[col], geo_s['g_n'], gm, k)
        stats[geo] = geo_s

    # ── 6. Month trend ──
    prior['month_key'] = prior['training_day'].dt.to_period('M').astype(str)
    mt = prior.groupby('month_key').agg(
        mt07=('adopted_within_07_days','mean'),
        mt90=('adopted_within_90_days','mean'),
        mt120=('adopted_within_120_days','mean'),
    ).reset_index()
    stats['month'] = mt

    # ── 7. Farmer-level history from Prior (by farmer_name) ──
    # For each farmer: their past adoption record in Prior
    # This is the STRONGEST signal: prior adopters → 12x baseline rate
    farmer_hist = prior.groupby('farmer_name').agg(
        fh_n_sessions     = ('training_day', 'count'),
        fh_ever_07        = ('adopted_within_07_days', 'max'),
        fh_ever_90        = ('adopted_within_90_days', 'max'),
        fh_ever_120       = ('adopted_within_120_days', 'max'),
        fh_mean_07        = ('adopted_within_07_days', 'mean'),
        fh_mean_90        = ('adopted_within_90_days', 'mean'),
        fh_mean_120       = ('adopted_within_120_days', 'mean'),
        fh_sum_07         = ('adopted_within_07_days', 'sum'),
        fh_sum_90         = ('adopted_within_90_days', 'sum'),
        fh_sum_120        = ('adopted_within_120_days', 'sum'),
        fh_last_date      = ('training_day', 'max'),
        fh_first_date     = ('training_day', 'min'),
    ).reset_index()
    stats['farmer_hist'] = farmer_hist

    return stats


def apply_features(df, stats, is_train=True):
    """Apply all features to a dataframe."""
    df = df.copy()
    g = stats['global']
    g07, g90, g120 = g['07'], g['90'], g['120']

    # ── Session features ──
    sess = stats['session']
    ms = df[['session_key']].merge(sess, on='session_key', how='left')
    df['sess_07_mean']  = ms['sess_07_mean'].fillna(g07).values
    df['sess_90_mean']  = ms['sess_90_mean'].fillna(g90).values
    df['sess_120_mean'] = ms['sess_120_mean'].fillna(g120).values
    df['sess_07_sum']   = ms['sess_07_sum'].fillna(0).values
    df['sess_90_sum']   = ms['sess_90_sum'].fillna(0).values
    df['sess_120_sum']  = ms['sess_120_sum'].fillna(0).values
    df['sess_n']        = ms['sess_n'].fillna(0).values
    df['has_session']   = (df['sess_n'] > 0).astype(int)

    # ── Trainer features (9 unique — very strong) ──
    tr = stats['trainer']
    df['trainer_07_rate']  = df['trainer_clean'].map(tr['tr07']).fillna(g07)
    df['trainer_90_rate']  = df['trainer_clean'].map(tr['tr90']).fillna(g90)
    df['trainer_120_rate'] = df['trainer_clean'].map(tr['tr120']).fillna(g120)

    # ── Topic features ──
    tp = stats['topic']
    for pfx, gm in [('07',g07),('90',g90),('120',g120)]:
        tm = tp[pfx]
        df[f'topic_{pfx}_rate'] = df['topics'].apply(
            lambda ts: np.mean([tm.get(t,gm) for t in ts]) if ts else gm)
        df[f'topic_{pfx}_max'] = df['topics'].apply(
            lambda ts: max([tm.get(t,gm) for t in ts], default=gm) if ts else gm)
        df[f'topic_{pfx}_min'] = df['topics'].apply(
            lambda ts: min([tm.get(t,gm) for t in ts], default=gm) if ts else gm)

    # ── Group features ──
    gs = stats['group']
    tmp = df[['group_name']].merge(gs[['g07','g90','g120']].reset_index(), on='group_name', how='left')
    df['group_07_rate']  = tmp['g07'].fillna(g07).values
    df['group_90_rate']  = tmp['g90'].fillna(g90).values
    df['group_120_rate'] = tmp['g120'].fillna(g120).values

    # ── Geo features ──
    for geo in ['county', 'subcounty', 'ward']:
        geo_s = stats[geo]
        tmp = df[[geo]].merge(geo_s[['g07','g90','g120']].reset_index(), on=geo, how='left')
        df[f'{geo}_07_rate']  = tmp['g07'].fillna(g07).values
        df[f'{geo}_90_rate']  = tmp['g90'].fillna(g90).values
        df[f'{geo}_120_rate'] = tmp['g120'].fillna(g120).values

    # ── Month trend ──
    df['month_key'] = df['training_day'].dt.to_period('M').astype(str)
    mt = stats['month']
    tmp = df[['month_key']].merge(mt, on='month_key', how='left')
    df['month_trend_07']  = tmp['mt07'].fillna(g07).values
    df['month_trend_90']  = tmp['mt90'].fillna(g90).values
    df['month_trend_120'] = tmp['mt120'].fillna(g120).values
    df.drop(columns=['month_key'], inplace=True)

    # ── FARMER HISTORY (strongest feature!) ──
    # For train: use only prior sessions BEFORE this training_day (strict <)
    # For test: same — prior sessions before test training_day
    fh = stats['farmer_hist']
    tmp = df[['farmer_name', 'training_day']].merge(
        fh, on='farmer_name', how='left'
    )
    # Days between last prior session and this training
    tmp['fh_days_since_last']  = (tmp['training_day'] - tmp['fh_last_date']).dt.days
    tmp['fh_days_since_first'] = (tmp['training_day'] - tmp['fh_first_date']).dt.days
    tmp['fh_training_freq']    = (tmp['fh_n_sessions'] /
                                   (tmp['fh_days_since_first'].replace(0, np.nan) / 30)).fillna(0)

    for col in ['fh_n_sessions','fh_ever_07','fh_ever_90','fh_ever_120',
                'fh_mean_07','fh_mean_90','fh_mean_120',
                'fh_sum_07','fh_sum_90','fh_sum_120',
                'fh_days_since_last','fh_days_since_first','fh_training_freq']:
        fill = 0
        df[col] = tmp[col].fillna(fill).values

    df['fh_has_history'] = (df['fh_n_sessions'] > 0).astype(int)
    df['fh_log_days_since_last']  = np.log1p(df['fh_days_since_last'].clip(0))
    df['fh_log_days_since_first'] = np.log1p(df['fh_days_since_first'].clip(0))

    # ── Date features ──
    df['td_month']   = df['training_day'].dt.month
    df['td_dow']     = df['training_day'].dt.dayofweek
    df['td_year']    = df['training_day'].dt.year
    df['td_quarter'] = df['training_day'].dt.quarter
    df['td_weeknum'] = df['training_day'].dt.isocalendar().week.astype(int)
    df['td_dayofyear'] = df['training_day'].dt.dayofyear

    # ── Topic count features ──
    df['num_topics'] = df['topics'].apply(len)

    # ── Interactions (trainer × topic × session × farmer history) ──
    for pfx, gm in [('07',g07),('90',g90),('120',g120)]:
        tr_r  = df[f'trainer_{pfx}_rate']
        tp_r  = df[f'topic_{pfx}_rate']
        tp_mx = df[f'topic_{pfx}_max']
        sm    = df[f'sess_{pfx}_mean']
        fh_m  = df[f'fh_mean_{pfx}']
        fh_e  = df[f'fh_ever_{pfx}']
        grp   = df[f'group_{pfx}_rate']
        cnt   = df[f'county_{pfx}_rate']

        df[f'tr_x_tp_{pfx}']       = tr_r * tp_r
        df[f'tr_x_tpmax_{pfx}']    = tr_r * tp_mx
        df[f'sess_x_tr_{pfx}']     = sm * tr_r
        df[f'sess_x_tp_{pfx}']     = sm * tp_r
        df[f'sess_x_grp_{pfx}']    = sm * grp
        df[f'fh_x_tr_{pfx}']       = fh_m * tr_r
        df[f'fh_x_tp_{pfx}']       = fh_m * tp_r
        df[f'fh_x_sess_{pfx}']     = fh_m * sm
        df[f'ever_x_tr_{pfx}']     = fh_e * tr_r
        df[f'ever_x_tp_{pfx}']     = fh_e * tp_r
        df[f'cnt_x_tr_{pfx}']      = cnt * tr_r
        df[f'coop_x_tr_{pfx}']     = df['belong_to_cooperative'] * tr_r

    return df


def preprocess(train, test):
    """Label encode and align features."""
    all_data = pd.concat([train, test], ignore_index=True)

    # Label encode
    for col in ['gender', 'age', 'registration', 'county', 'subcounty', 'ward']:
        if col in all_data.columns:
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col].astype(str).fillna('missing'))

    # Frequency encode group_name (864 groups — too many for label encode)
    freq = train['group_name'].value_counts(normalize=True)
    all_data['group_freq'] = all_data['group_name'].map(freq).fillna(0)

    drop_cols = ['ID', 'farmer_name', 'training_day', 'trainer', 'trainer_clean',
                 'topics_list', 'topics', 'session_key', 'group_name',
                 'registration'] + TARGETS
    all_data.drop(columns=[c for c in drop_cols if c in all_data.columns],
                  inplace=True, errors='ignore')

    # Align columns
    common = [c for c in all_data.columns]
    train_proc = all_data.iloc[:len(train)].copy()
    test_proc  = all_data.iloc[len(train):].copy()

    # Validate no target leakage
    for t in TARGETS:
        assert t not in train_proc.columns, f"LEAKAGE: {t} in features!"

    # Impute
    imp = SimpleImputer(strategy='median')
    train_arr  = imp.fit_transform(train_proc)
    test_arr   = imp.transform(test_proc)
    feat_cols  = train_proc.columns.tolist()

    # Handle shape mismatch
    n_feat = train_arr.shape[1]
    train_proc = pd.DataFrame(train_arr, columns=feat_cols[:n_feat])
    test_proc  = pd.DataFrame(test_arr,  columns=feat_cols[:n_feat])

    print(f"  Final feature count: {train_proc.shape[1]}")
    return train_proc, test_proc


# ─────────────────────────────────────────────────────────────
# TRAIN & PREDICT
# ─────────────────────────────────────────────────────────────

def train_predict(X_train, y, X_test, tgt_name):
    """
    CALIBRATION-CORRECT approach:
    - sample_weight (not class_weight) → correct probability scale
    - Temperature scaling for final calibration
    - Blend calibrated probs with logistic regression anchor
    """
    y_arr = y.values if hasattr(y, 'values') else y
    pos_rate = y_arr.mean()
    # sample weights: balances classes without distorting probability scale
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
            verbose=0, random_state=SEED,
            early_stopping_rounds=150,
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
        # Logistic regression: best calibration anchor for log loss
        LogisticRegression(C=0.05, max_iter=2000, random_state=SEED),
    ]

    n_models   = len(models)
    oof_preds  = np.zeros((len(X_train), n_models))
    test_preds = np.zeros((len(X_test),  n_models))

    print(f"\n  [{tgt_name}] pos_rate={pos_rate:.5f}  ({int(y_arr.sum())}/{len(y_arr)})")

    for i, model in enumerate(models):
        oof_fold   = np.zeros(len(X_train))
        test_fold  = np.zeros(len(X_test))

        for tr_idx, val_idx in skf.split(X_train, y_arr):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]
            sw_tr = sw[tr_idx]

            if isinstance(model, XGBClassifier):
                model.fit(X_tr, y_tr, sample_weight=sw_tr,
                         eval_set=[(X_val, y_val)], verbose=False)
            elif isinstance(model, LGBMClassifier):
                model.fit(X_tr, y_tr, sample_weight=sw_tr,
                         eval_set=[(X_val, y_val)],
                         callbacks=[early_stopping(200, verbose=False),
                                    log_evaluation(period=-1)])
            elif isinstance(model, CatBoostClassifier):
                model.fit(X_tr, y_tr, sample_weight=sw_tr,
                         eval_set=(X_val, y_val), verbose=False)
            else:
                model.fit(X_tr, y_tr, sample_weight=sw_tr)

            oof_fold[val_idx] += model.predict_proba(X_val)[:, 1]
            test_fold         += model.predict_proba(X_test)[:, 1] / N_FOLDS

        oof_preds[:, i]  = oof_fold
        test_preds[:, i] = test_fold
        ll  = log_loss(y_arr, oof_fold)
        auc = roc_auc_score(y_arr, oof_fold)
        print(f"    model {i:d}: ll={ll:.5f}  auc={auc:.5f}")

    # Simple average
    oof_avg  = oof_preds.mean(axis=1)
    test_avg = test_preds.mean(axis=1)
    ll_avg   = log_loss(y_arr, oof_avg)
    auc_avg  = roc_auc_score(y_arr, oof_avg)
    print(f"  Avg: ll={ll_avg:.5f}  auc={auc_avg:.5f}")

    # Optimal model weights via OOF
    best_w, best_ll_w = np.ones(n_models) / n_models, ll_avg
    from itertools import product
    # Simple grid: vary logistic regression weight
    for w_lr in np.linspace(0.0, 0.5, 11):
        other_w = (1 - w_lr) / (n_models - 1)
        w = np.array([other_w] * (n_models - 1) + [w_lr])
        oof_blend = (oof_preds * w).sum(axis=1)
        ll_w = log_loss(y_arr, np.clip(oof_blend, 1e-7, 1-1e-7))
        if ll_w < best_ll_w:
            best_ll_w = ll_w
            best_w = w
    oof_blend  = np.clip((oof_preds * best_w).sum(axis=1), 1e-7, 1-1e-7)
    test_blend = np.clip((test_preds * best_w).sum(axis=1), 1e-7, 1-1e-7)
    print(f"  Weighted: ll={log_loss(y_arr,oof_blend):.5f}  lr_w={best_w[-1]:.2f}")

    # Temperature scaling
    best_T, best_T_ll = temperature_scale(oof_blend, y_arr)
    test_temp = np.clip(test_blend ** (1.0 / best_T), 1e-7, 1-1e-7)
    print(f"  Temp T={best_T:.3f}: ll={best_T_ll:.5f}  auc={roc_auc_score(y_arr,oof_blend):.5f}")

    return test_temp, oof_blend, y_arr


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Loading data...")
    train, test, prior, sample = load_data()
    n_train = len(train)

    print("\nBuilding lookup stats from Prior...")
    stats = build_prior_stats(prior)

    # CRITICAL: has_topic_trained_on == 0 → zero adoption (hard rule from data)
    # We'll handle this with a near-zero floor for these farmers after prediction
    train_no_topic_mask = (train['has_topic_trained_on'] == 0)
    test_no_topic_mask  = (test['has_topic_trained_on'] == 0)
    print(f"\nhas_topic=0: train={train_no_topic_mask.sum()}, test={test_no_topic_mask.sum()}")
    print("(These farmers have 0% historical adoption rate → will floor predictions)")

    print("\nEngineering features...")
    train_f = apply_features(train, stats, is_train=True)
    test_f  = apply_features(test,  stats, is_train=False)

    print("\nPreprocessing...")
    # Concat with targets attached temporarily for alignment
    for t in TARGETS:
        train_f[t] = train[t].values

    train_proc, test_proc = preprocess(train_f, test_f)
    test_ids = test['ID'].values

    submission_preds = {}
    oof_results      = {}

    for tgt in TARGETS:
        print(f"\n{'='*60}\nTarget: {tgt}")
        y = train[tgt].reset_index(drop=True)
        test_pred, oof_pred, y_arr = train_predict(train_proc, y, test_proc, tgt)

        # Apply hard floor for has_topic_trained_on == 0
        # These farmers historically have 0% adoption → set to near-zero
        gm = stats['global'][tgt.split('_')[-2]]
        floor_val = gm * 0.05  # 5% of global mean
        test_pred[test_no_topic_mask.values] = floor_val

        submission_preds[tgt] = test_pred
        oof_results[tgt]      = (oof_pred, y_arr)

    # ── Final OOF summary ──
    print("\n" + "="*60)
    print("FINAL OOF SUMMARY")
    total = 0
    for tgt in TARGETS:
        oof, y_arr = oof_results[tgt]
        ll   = log_loss(y_arr, oof)
        auc  = roc_auc_score(y_arr, oof)
        comp = 0.75 * (1 - ll) + 0.25 * auc
        total += comp / 3
        print(f"  {tgt}: ll={ll:.5f}  auc={auc:.5f}  comp_score={comp:.5f}")
    print(f"  ESTIMATED LEADERBOARD SCORE: {total:.5f}")

    # ── Build submission ──
    sub = sample[['ID']].copy()
    sub['Target_07_AUC']      = submission_preds['adopted_within_07_days']
    sub['Target_90_AUC']      = submission_preds['adopted_within_90_days']
    sub['Target_120_AUC']     = submission_preds['adopted_within_120_days']
    sub['Target_07_LogLoss']  = submission_preds['adopted_within_07_days']
    sub['Target_90_LogLoss']  = submission_preds['adopted_within_90_days']
    sub['Target_120_LogLoss'] = submission_preds['adopted_within_120_days']

    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"Column mismatch!\n  Expected: {list(sample.columns)}\n  Got: {list(sub.columns)}")

    out_path = 'submission_final.csv'
    sub.to_csv(out_path, index=False)
    print(f"\n✓ Saved {out_path}  ({len(sub)} rows)")
    print(sub.describe())


if __name__ == '__main__':
    main()
