
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
