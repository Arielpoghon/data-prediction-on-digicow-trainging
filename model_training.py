# =========================================
# DigiCow Challenge - Advanced Model Training
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, log_loss, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

print("=" * 60)
print("DIGICOW - ADVANCED MODEL TRAINING & EVALUATION")
print("=" * 60)

# ------------------------
# Load processed data
# ------------------------
print("\n📂 LOADING PROCESSED DATA:")

train_processed = pd.read_csv("train_processed.csv")
val_processed = pd.read_csv("val_processed.csv")
test_processed = pd.read_csv("test_processed.csv")

# Separate features and target
target_col = 'adopted_within_07_days'
feature_cols = [col for col in train_processed.columns if col != target_col]

X_train = train_processed[feature_cols]
y_train = train_processed[target_col]
X_val = val_processed[feature_cols]
y_val = val_processed[target_col]
X_test = test_processed[feature_cols]

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")
print(f"Number of features: {len(feature_cols)}")

# ------------------------
# 1️⃣ LightGBM Model with Class Weights
# ------------------------
print("\n🌳 1️⃣ TRAINING LIGHTGBM MODEL:")

# Calculate class weights for imbalance
class_weights = {0: 1, 1: (len(y_train) - sum(y_train)) / sum(y_train)}
print(f"Class weights: {class_weights}")

# Initialize LightGBM with optimized parameters
lgb_model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=2000,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    class_weight='balanced'  # Handle class imbalance
)

# Train with early stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

print("✅ LightGBM training completed")

# ------------------------
# 2️⃣ Model Evaluation
# ------------------------
print("\n📊 2️⃣ MODEL EVALUATION:")

# Predict probabilities
y_train_pred = lgb_model.predict_proba(X_train)[:, 1]
y_val_pred = lgb_model.predict_proba(X_val)[:, 1]

# Calculate metrics
train_auc = roc_auc_score(y_train, y_train_pred)
val_auc = roc_auc_score(y_val, y_val_pred)
train_logloss = log_loss(y_train, y_train_pred)
val_logloss = log_loss(y_val, y_val_pred)

print(f"\n📈 PERFORMANCE METRICS:")
print(f"Training ROC-AUC: {train_auc:.4f}")
print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Training Log Loss: {train_logloss:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")

# Classification report with threshold 0.5
y_val_pred_class = (y_val_pred >= 0.5).astype(int)
print(f"\n📋 CLASSIFICATION REPORT (threshold=0.5):")
print(classification_report(y_val, y_val_pred_class))

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred_class)
print(f"\n🔢 CONFUSION MATRIX:")
print(cm)

# ------------------------
# 3️⃣ Feature Importance Analysis
# ------------------------
print("\n🎯 3️⃣ FEATURE IMPORTANCE ANALYSIS:")

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🏆 TOP 15 MOST IMPORTANT FEATURES:")
print(feature_importance.head(15).to_string(index=False))

# Plot feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, x='importance', y='feature')
plt.title('Top 15 Feature Importance - LightGBM')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 4️⃣ Cross-Validation Analysis
# ------------------------
print("\n🔄 4️⃣ CROSS-VALIDATION ANALYSIS:")

# Combine train and validation for CV
X_full = pd.concat([X_train, X_val], axis=0)
y_full = pd.concat([y_train, y_val], axis=0)

# Stratified K-Fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
cv_logloss = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
    print(f"Fold {fold + 1}/5...")
    
    X_fold_train, X_fold_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_fold_train, y_fold_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
    
    # Train model for this fold
    fold_model = LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=10,
        n_estimators=1000,  # Reduced for CV speed
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        class_weight='balanced'
    )
    
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Predict and evaluate
    fold_pred = fold_model.predict_proba(X_fold_val)[:, 1]
    fold_auc = roc_auc_score(y_fold_val, fold_pred)
    fold_logloss = log_loss(y_fold_val, fold_pred)
    
    cv_scores.append(fold_auc)
    cv_logloss.append(fold_logloss)
    
    print(f"  Fold AUC: {fold_auc:.4f}, Log Loss: {fold_logloss:.4f}")

print(f"\n📊 CROSS-VALIDATION RESULTS:")
print(f"Mean ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
print(f"Mean Log Loss: {np.mean(cv_logloss):.4f} ± {np.std(cv_logloss):.4f}")

# ------------------------
# 5️⃣ Threshold Optimization
# ------------------------
print("\n⚖️ 5️⃣ THRESHOLD OPTIMIZATION:")

# Find optimal threshold for F1 score
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []

from sklearn.metrics import f1_score

for threshold in thresholds:
    y_pred_thresh = (y_val_pred >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred_thresh)
    f1_scores.append(f1)

best_threshold_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_threshold_idx]
best_f1 = f1_scores[best_threshold_idx]

print(f"Best threshold: {best_threshold:.2f} (F1: {best_f1:.4f})")

# Plot threshold vs F1
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, 'bo-', linewidth=2, markersize=8)
plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold: {best_threshold:.2f}')
plt.xlabel('Classification Threshold')
plt.ylabel('F1 Score')
plt.title('Threshold Optimization for F1 Score')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 6️⃣ Test Set Predictions
# ------------------------
print("\n🔮 6️⃣ TEST SET PREDICTIONS:")

# Load original test data for ID extraction
test_original = pd.read_csv("data/Test.csv")

# Predict on test set
test_pred_proba = lgb_model.predict_proba(X_test)[:, 1]

# Create submission
submission = pd.DataFrame({
    'ID': test_original['ID'],
    'Target_AUC': test_pred_proba,
    'Target_LogLoss': test_pred_proba
})

# Save submission
submission.to_csv("submission_advanced_model.csv", index=False)

print(f"✅ Test predictions completed")
print(f"Probability range: [{test_pred_proba.min():.4f}, {test_pred_proba.max():.4f}]")
print(f"Mean probability: {test_pred_proba.mean():.4f}")

# ------------------------
# 7️⃣ Model Diagnostics
# ------------------------
print("\n🔍 7️⃣ MODEL DIAGNOSTICS:")

# Calibration plot
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_val, y_val_pred, n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='LightGBM')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('calibration_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_val, y_val_pred)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'LightGBM (AUC = {val_auc:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 📊 Final Summary
# ------------------------
print("\n" + "=" * 60)
print("🎯 MODEL TRAINING SUMMARY")
print("=" * 60)

print(f"\n📈 FINAL PERFORMANCE:")
print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation Log Loss: {val_logloss:.4f}")
print(f"Cross-Validation ROC-AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

print(f"\n🏆 TOP 5 FEATURES:")
for i, row in feature_importance.head(5).iterrows():
    print(f"{i+1}. {row['feature']}: {row['importance']:.0f}")

print(f"\n💾 OUTPUT FILES:")
print(f"- submission_advanced_model.csv")
print(f"- feature_importance.png")
print(f"- threshold_optimization.png")
print(f"- calibration_curve.png")
print(f"- roc_curve.png")

print(f"\n✅ Model training and evaluation complete!")
print("=" * 60)
