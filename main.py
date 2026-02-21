
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
