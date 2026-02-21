[['g07','g90','g120']].reset_index(), on=grp_col, how='left')
            df[f'{grp_col}_07_rate']  = tmp['g07'].fillna(g['07']).values
            df[f'{grp_col}_90_rate']  = tmp['g90'].fillna(g['90']).values
 
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
