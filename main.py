
    
    # Fit calibrator on OOF predictions (isotonic for imbalance)
    calibrator = CalibratedClassifierCV(meta, method='isotonic', cv='prefit')
    calibrator.fit(oof_preds, y)
    
    return base_models, meta, calibrator
def predict_stack(base_models, meta, calibrator, test_proc):
    base_preds = np.column_stack([m.predict_proba(test_proc)[:,1] for m in base_models])
    raw_pred = meta.predict_proba(base_preds)[:,1]  # For AUC
    calibrated_pred = calibrator.predict_proba(base_preds)[:,1]  # For LogLoss
    return np.clip(raw_pred, 0.001, 0.999), np.clip(calibrated_pred, 0.001, 0.999)
# --------------------- MAIN ---------------------
def main():
    train, test, prior, sample = load_data()
    print("Building prior features...")
    train, test = add_prior_features(train, test, prior)
    print("Preprocessing...")
    train_proc, test_proc = preprocess(train, test)
    print(f"Final features: {train_proc.shape[1]}")
    submission_preds_auc = {}
    submission_preds_logloss = {}
    for tgt in TARGETS:
        print(f"Training {tgt} with stacking...")
        y = train[tgt]
        base_models, meta_model, calibrator = train_stack_model(train_proc, y)
        auc_pred, logloss_pred = predict_stack(base_models, meta_model, calibrator, test_proc)
        submission_preds_auc[tgt] = auc_pred
        submission_preds_logloss[tgt] = logloss_pred
    sub = sample[['ID']].copy()
    sub['Target_07_AUC'] = submission_preds_auc['adopted_within_07_days']
    sub['Target_07_LogLoss'] = submission_preds_logloss['adopted_within_07_days']
    sub['Target_90_AUC'] = submission_preds_auc['adopted_within_90_days']
    sub['Target_90_LogLoss'] = submission_preds_logloss['adopted_within_90_days']
    sub['Target_120_AUC'] = submission_preds_auc['adopted_within_120_days']
    sub['Target_120_LogLoss'] = submission_preds_logloss['adopted_within_120_days']
    sub.to_csv('submission_final.csv', index=False)
    print("\nsubmission_final.csv saved → submit this now!")
if __name__ == "__main__":
    main()
