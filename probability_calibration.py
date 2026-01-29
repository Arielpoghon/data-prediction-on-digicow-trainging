# =========================================
# DigiCow Challenge - Probability Calibration
# =========================================

import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

print("=" * 60)
print("DIGICOW - PROBABILITY CALIBRATION FOR LOG LOSS OPTIMIZATION")
print("=" * 60)

# ------------------------
# Load data and model
# ------------------------
print("\n📂 LOADING DATA AND MODEL:")

# Load processed data
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

# Load trained model (retrain for calibration)
from lightgbm import LGBMClassifier
import lightgbm as lgb

print("\n🔄 RETRAINING BASE MODEL FOR CALIBRATION:")

base_model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=165,  # Use optimal iterations from previous training
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    class_weight='balanced'
)

base_model.fit(X_train, y_train)

# Get original predictions
y_val_pred_orig = base_model.predict_proba(X_val)[:, 1]
y_test_pred_orig = base_model.predict_proba(X_test)[:, 1]

print("✅ Base model retrained")

# ------------------------
# 1️⃣ Original Model Performance
# ------------------------
print("\n📊 1️⃣ ORIGINAL MODEL PERFORMANCE:")

orig_auc = roc_auc_score(y_val, y_val_pred_orig)
orig_logloss = log_loss(y_val, y_val_pred_orig)
orig_brier = brier_score_loss(y_val, y_val_pred_orig)

print(f"Original ROC-AUC: {orig_auc:.4f}")
print(f"Original Log Loss: {orig_logloss:.4f}")
print(f"Original Brier Score: {orig_brier:.4f}")

# ------------------------
# 2️⃣ Platt Scaling (Logistic Calibration)
# ------------------------
print("\n🔧 2️⃣ PLATT SCALING CALIBRATION:")

# Platt scaling with cross-validation
platt_calibrated = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
platt_calibrated.fit(X_train, y_train)

y_val_pred_platt = platt_calibrated.predict_proba(X_val)[:, 1]
y_test_pred_platt = platt_calibrated.predict_proba(X_test)[:, 1]

platt_auc = roc_auc_score(y_val, y_val_pred_platt)
platt_logloss = log_loss(y_val, y_val_pred_platt)
platt_brier = brier_score_loss(y_val, y_val_pred_platt)

print(f"Platt ROC-AUC: {platt_auc:.4f}")
print(f"Platt Log Loss: {platt_logloss:.4f}")
print(f"Platt Brier Score: {platt_brier:.4f}")

print(f"Log Loss Improvement: {orig_logloss - platt_logloss:.4f}")
print(f"Brier Score Improvement: {orig_brier - platt_brier:.4f}")

# ------------------------
# 3️⃣ Isotonic Regression Calibration
# ------------------------
print("\n🔧 3️⃣ ISOTONIC REGRESSION CALIBRATION:")

# Isotonic regression with cross-validation
isotonic_calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
isotonic_calibrated.fit(X_train, y_train)

y_val_pred_isotonic = isotonic_calibrated.predict_proba(X_val)[:, 1]
y_test_pred_isotonic = isotonic_calibrated.predict_proba(X_test)[:, 1]

isotonic_auc = roc_auc_score(y_val, y_val_pred_isotonic)
isotonic_logloss = log_loss(y_val, y_val_pred_isotonic)
isotonic_brier = brier_score_loss(y_val, y_val_pred_isotonic)

print(f"Isotonic ROC-AUC: {isotonic_auc:.4f}")
print(f"Isotonic Log Loss: {isotonic_logloss:.4f}")
print(f"Isotonic Brier Score: {isotonic_brier:.4f}")

print(f"Log Loss Improvement: {orig_logloss - isotonic_logloss:.4f}")
print(f"Brier Score Improvement: {orig_brier - isotonic_brier:.4f}")

# ------------------------
# 4️⃣ Temperature Scaling
# ------------------------
print("\n🔧 4️⃣ TEMPERATURE SCALING:")

def temperature_scaling(predictions, temperature):
    """Apply temperature scaling to predictions"""
    scaled = np.log(predictions + 1e-15) / temperature
    scaled = np.exp(scaled)
    return scaled / (1 + scaled)

# Find optimal temperature on validation set
from scipy.optimize import minimize_scalar

def temperature_objective(temp, y_true, y_pred):
    y_scaled = temperature_scaling(y_pred, temp)
    return log_loss(y_true, y_scaled)

result = minimize_scalar(temperature_objective, 
                         bounds=(0.1, 10.0), 
                         args=(y_val, y_val_pred_orig),
                         method='bounded')

optimal_temp = result.x
print(f"Optimal Temperature: {optimal_temp:.4f}")

y_val_pred_temp = temperature_scaling(y_val_pred_orig, optimal_temp)
y_test_pred_temp = temperature_scaling(y_test_pred_orig, optimal_temp)

temp_auc = roc_auc_score(y_val, y_val_pred_temp)
temp_logloss = log_loss(y_val, y_val_pred_temp)
temp_brier = brier_score_loss(y_val, y_val_pred_temp)

print(f"Temperature ROC-AUC: {temp_auc:.4f}")
print(f"Temperature Log Loss: {temp_logloss:.4f}")
print(f"Temperature Brier Score: {temp_brier:.4f}")

print(f"Log Loss Improvement: {orig_logloss - temp_logloss:.4f}")
print(f"Brier Score Improvement: {orig_brier - temp_brier:.4f}")

# ------------------------
# 5️⃣ Calibration Comparison
# ------------------------
print("\n📊 5️⃣ CALIBRATION COMPARISON:")

comparison_data = {
    'Method': ['Original', 'Platt Scaling', 'Isotonic Regression', 'Temperature Scaling'],
    'ROC-AUC': [orig_auc, platt_auc, isotonic_auc, temp_auc],
    'Log Loss': [orig_logloss, platt_logloss, isotonic_logloss, temp_logloss],
    'Brier Score': [orig_brier, platt_brier, isotonic_brier, temp_brier]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.round(4))

# Find best method for Log Loss
best_method = comparison_df.loc[comparison_df['Log Loss'].idxmin(), 'Method']
best_logloss = comparison_df['Log Loss'].min()

print(f"\n🏆 Best method for Log Loss: {best_method}")
print(f"Best Log Loss: {best_logloss:.4f}")

# ------------------------
# 6️⃣ Calibration Curves Visualization
# ------------------------
print("\n📈 6️⃣ CALIBRATION CURVES:")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Calibration Curves Comparison', fontsize=16, fontweight='bold')

# Original model
prob_true_orig, prob_pred_orig = calibration_curve(y_val, y_val_pred_orig, n_bins=10)
axes[0, 0].plot(prob_pred_orig, prob_true_orig, marker='o', linewidth=2, label='Original Model')
axes[0, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
axes[0, 0].set_title(f'Original Model (Log Loss: {orig_logloss:.4f})')
axes[0, 0].set_xlabel('Mean Predicted Probability')
axes[0, 0].set_ylabel('Fraction of Positives')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Platt scaling
prob_true_platt, prob_pred_platt = calibration_curve(y_val, y_val_pred_platt, n_bins=10)
axes[0, 1].plot(prob_pred_platt, prob_true_platt, marker='o', linewidth=2, label='Platt Scaling')
axes[0, 1].plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
axes[0, 1].set_title(f'Platt Scaling (Log Loss: {platt_logloss:.4f})')
axes[0, 1].set_xlabel('Mean Predicted Probability')
axes[0, 1].set_ylabel('Fraction of Positives')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Isotonic regression
prob_true_iso, prob_pred_iso = calibration_curve(y_val, y_val_pred_isotonic, n_bins=10)
axes[1, 0].plot(prob_pred_iso, prob_true_iso, marker='o', linewidth=2, label='Isotonic Regression')
axes[1, 0].plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
axes[1, 0].set_title(f'Isotonic Regression (Log Loss: {isotonic_logloss:.4f})')
axes[1, 0].set_xlabel('Mean Predicted Probability')
axes[1, 0].set_ylabel('Fraction of Positives')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Temperature scaling
prob_true_temp, prob_pred_temp = calibration_curve(y_val, y_val_pred_temp, n_bins=10)
axes[1, 1].plot(prob_pred_temp, prob_true_temp, marker='o', linewidth=2, label='Temperature Scaling')
axes[1, 1].plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
axes[1, 1].set_title(f'Temperature Scaling (Log Loss: {temp_logloss:.4f})')
axes[1, 1].set_xlabel('Mean Predicted Probability')
axes[1, 1].set_ylabel('Fraction of Positives')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 7️⃣ Probability Distribution Comparison
# ------------------------
print("\n📊 7️⃣ PROBABILITY DISTRIBUTIONS:")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Probability Distributions Comparison', fontsize=16, fontweight='bold')

# Original predictions
axes[0, 0].hist(y_val_pred_orig, bins=50, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title(f'Original Predictions (Mean: {np.mean(y_val_pred_orig):.3f})')
axes[0, 0].set_xlabel('Predicted Probability')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(True, alpha=0.3)

# Platt scaling
axes[0, 1].hist(y_val_pred_platt, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0, 1].set_title(f'Platt Scaling (Mean: {np.mean(y_val_pred_platt):.3f})')
axes[0, 1].set_xlabel('Predicted Probability')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(True, alpha=0.3)

# Isotonic regression
axes[1, 0].hist(y_val_pred_isotonic, bins=50, alpha=0.7, color='red', edgecolor='black')
axes[1, 0].set_title(f'Isotonic Regression (Mean: {np.mean(y_val_pred_isotonic):.3f})')
axes[1, 0].set_xlabel('Predicted Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Temperature scaling
axes[1, 1].hist(y_val_pred_temp, bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 1].set_title(f'Temperature Scaling (Mean: {np.mean(y_val_pred_temp):.3f})')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('probability_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 8️⃣ Final Submission with Best Calibration
# ------------------------
print("\n🔮 8️⃣ CREATING CALIBRATED SUBMISSION:")

# Select best calibration method
if best_method == 'Platt Scaling':
    final_test_pred = y_test_pred_platt
    final_val_pred = y_val_pred_platt
elif best_method == 'Isotonic Regression':
    final_test_pred = y_test_pred_isotonic
    final_val_pred = y_val_pred_isotonic
elif best_method == 'Temperature Scaling':
    final_test_pred = y_test_pred_temp
    final_val_pred = y_val_pred_temp
else:
    final_test_pred = y_test_pred_orig
    final_val_pred = y_val_pred_orig

# Load original test data for ID extraction
test_original = pd.read_csv("data/Test.csv")

# Create calibrated submission
submission_calibrated = pd.DataFrame({
    'ID': test_original['ID'],
    'Target_AUC': final_test_pred,
    'Target_LogLoss': final_test_pred
})

submission_calibrated.to_csv("submission_calibrated.csv", index=False)

print(f"✅ Calibrated submission created with {best_method}")
print(f"Final validation Log Loss: {log_loss(y_val, final_val_pred):.4f}")
print(f"Final validation ROC-AUC: {roc_auc_score(y_val, final_val_pred):.4f}")
print(f"Probability range: [{final_test_pred.min():.4f}, {final_test_pred.max():.4f}]")

# ------------------------
# 📊 Final Summary
# ------------------------
print("\n" + "=" * 60)
print("🎯 PROBABILITY CALIBRATION SUMMARY")
print("=" * 60)

print(f"\n📈 CALIBRATION RESULTS:")
print(f"Original Log Loss: {orig_logloss:.4f}")
print(f"Best Calibrated Log Loss: {best_logloss:.4f}")
print(f"Log Loss Improvement: {orig_logloss - best_logloss:.4f}")
print(f"Best Method: {best_method}")

print(f"\n📊 PERFORMANCE COMPARISON:")
for _, row in comparison_df.iterrows():
    print(f"{row['Method']}: AUC={row['ROC-AUC']:.4f}, LogLoss={row['Log Loss']:.4f}, Brier={row['Brier Score']:.4f}")

print(f"\n💾 OUTPUT FILES:")
print(f"- submission_calibrated.csv (Best calibration method)")
print(f"- calibration_comparison.png")
print(f"- probability_distributions.png")

print(f"\n✅ Probability calibration complete!")
print("=" * 60)
