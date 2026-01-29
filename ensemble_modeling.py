# =========================================
# DigiCow Challenge - Ensemble Modeling
# =========================================

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMClassifier
try:
    import catboost as cb
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, will skip CatBoost model")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend to non-interactive
import matplotlib
matplotlib.use('Agg')

print("=" * 60)
print("DIGICOW - ENSEMBLE MODELING FOR OPTIMAL PERFORMANCE")
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
# 1️⃣ Train Individual Models
# ------------------------
print("\n🌳 1️⃣ TRAINING INDIVIDUAL MODELS:")

models = {}
predictions = {}

# 1. LightGBM
print("Training LightGBM...")
lgb_model = LGBMClassifier(
    objective='binary',
    boosting_type='gbdt',
    learning_rate=0.03,
    num_leaves=64,
    max_depth=-1,
    min_child_samples=10,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
    class_weight='balanced'
)

lgb_model.fit(X_train, y_train)
models['LightGBM'] = lgb_model
predictions['LightGBM'] = {
    'val': lgb_model.predict_proba(X_val)[:, 1],
    'test': lgb_model.predict_proba(X_test)[:, 1]
}
print("✅ LightGBM trained")

# 2. CatBoost (skip if not available)
if CATBOOST_AVAILABLE:
    try:
        print("Training CatBoost...")
        cat_model = CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            border_count=32,
            random_state=42,
            verbose=False,
            class_weights=[1, (len(y_train) - sum(y_train)) / sum(y_train)]
        )

        cat_model.fit(X_train, y_train)
        models['CatBoost'] = cat_model
        predictions['CatBoost'] = {
            'val': cat_model.predict_proba(X_val)[:, 1],
            'test': cat_model.predict_proba(X_test)[:, 1]
        }
        print("✅ CatBoost trained")
    except Exception as e:
        print(f"⚠️ CatBoost training failed: {e}")
        print("Skipping CatBoost and continuing with other models...")
else:
    print("⚠️ CatBoost not available, skipping...")

# 3. Logistic Regression
print("Training Logistic Regression...")
# Scale features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    random_state=42,
    class_weight='balanced',
    max_iter=1000
)

lr_model.fit(X_train_scaled, y_train)
models['LogisticRegression'] = lr_model
predictions['LogisticRegression'] = {
    'val': lr_model.predict_proba(X_val_scaled)[:, 1],
    'test': lr_model.predict_proba(X_test_scaled)[:, 1]
}
print("✅ Logistic Regression trained")

# 4. Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
models['RandomForest'] = rf_model
predictions['RandomForest'] = {
    'val': rf_model.predict_proba(X_val)[:, 1],
    'test': rf_model.predict_proba(X_test)[:, 1]
}
print("✅ Random Forest trained")

# 5. Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

gb_model.fit(X_train, y_train)
models['GradientBoosting'] = gb_model
predictions['GradientBoosting'] = {
    'val': gb_model.predict_proba(X_val)[:, 1],
    'test': gb_model.predict_proba(X_test)[:, 1]
}
print("✅ Gradient Boosting trained")

# ------------------------
# 2️⃣ Individual Model Performance
# ------------------------
print("\n📊 2️⃣ INDIVIDUAL MODEL PERFORMANCE:")

individual_results = []

for model_name, pred_dict in predictions.items():
    val_pred = pred_dict['val']
    auc = roc_auc_score(y_val, val_pred)
    logloss = log_loss(y_val, val_pred)
    brier = brier_score_loss(y_val, val_pred)
    
    individual_results.append({
        'Model': model_name,
        'ROC-AUC': auc,
        'Log Loss': logloss,
        'Brier Score': brier
    })
    
    print(f"{model_name}: AUC={auc:.4f}, LogLoss={logloss:.4f}, Brier={brier:.4f}")

individual_df = pd.DataFrame(individual_results)

# ------------------------
# 3️⃣ Simple Ensemble Methods
# ------------------------
print("\n🔀 3️⃣ SIMPLE ENSEMBLE METHODS:")

ensemble_results = []

# Simple Average Ensemble
val_avg = np.mean([pred['val'] for pred in predictions.values()], axis=0)
test_avg = np.mean([pred['test'] for pred in predictions.values()], axis=0)

avg_auc = roc_auc_score(y_val, val_avg)
avg_logloss = log_loss(y_val, val_avg)
avg_brier = brier_score_loss(y_val, val_avg)

ensemble_results.append({
    'Method': 'Simple Average',
    'ROC-AUC': avg_auc,
    'Log Loss': avg_logloss,
    'Brier Score': avg_brier
})

print(f"Simple Average: AUC={avg_auc:.4f}, LogLoss={avg_logloss:.4f}, Brier={avg_brier:.4f}")

# Weighted Average (based on validation AUC)
weights = []
for model_name in predictions.keys():
    model_auc = individual_df[individual_df['Model'] == model_name]['ROC-AUC'].values[0]
    weights.append(model_auc)

weights = np.array(weights)
weights = weights / weights.sum()  # Normalize

val_weighted = np.average([pred['val'] for pred in predictions.values()], 
                         axis=0, weights=weights)
test_weighted = np.average([pred['test'] for pred in predictions.values()], 
                          axis=0, weights=weights)

weighted_auc = roc_auc_score(y_val, val_weighted)
weighted_logloss = log_loss(y_val, val_weighted)
weighted_brier = brier_score_loss(y_val, val_weighted)

ensemble_results.append({
    'Method': 'Weighted Average (AUC)',
    'ROC-AUC': weighted_auc,
    'Log Loss': weighted_logloss,
    'Brier Score': weighted_brier
})

print(f"Weighted Average (AUC): AUC={weighted_auc:.4f}, LogLoss={weighted_logloss:.4f}, Brier={weighted_brier:.4f}")

# Weighted Average (based on inverse Log Loss)
weights_logloss = []
for model_name in predictions.keys():
    model_logloss = individual_df[individual_df['Model'] == model_name]['Log Loss'].values[0]
    weights_logloss.append(1 / model_logloss)

weights_logloss = np.array(weights_logloss)
weights_logloss = weights_logloss / weights_logloss.sum()

val_weighted_logloss = np.average([pred['val'] for pred in predictions.values()], 
                                  axis=0, weights=weights_logloss)
test_weighted_logloss = np.average([pred['test'] for pred in predictions.values()], 
                                   axis=0, weights=weights_logloss)

weighted_logloss_auc = roc_auc_score(y_val, val_weighted_logloss)
weighted_logloss_logloss = log_loss(y_val, val_weighted_logloss)
weighted_logloss_brier = brier_score_loss(y_val, val_weighted_logloss)

ensemble_results.append({
    'Method': 'Weighted Average (Log Loss)',
    'ROC-AUC': weighted_logloss_auc,
    'Log Loss': weighted_logloss_logloss,
    'Brier Score': weighted_logloss_brier
})

print(f"Weighted Average (Log Loss): AUC={weighted_logloss_auc:.4f}, LogLoss={weighted_logloss_logloss:.4f}, Brier={weighted_logloss_brier:.4f}")

# Top 3 Models Ensemble
top_3_models = individual_df.nlargest(3, 'ROC-AUC')['Model'].tolist()
val_top3 = np.mean([predictions[model]['val'] for model in top_3_models], axis=0)
test_top3 = np.mean([predictions[model]['test'] for model in top_3_models], axis=0)

top3_auc = roc_auc_score(y_val, val_top3)
top3_logloss = log_loss(y_val, val_top3)
top3_brier = brier_score_loss(y_val, val_top3)

ensemble_results.append({
    'Method': f'Top 3 Models ({", ".join(top_3_models)})',
    'ROC-AUC': top3_auc,
    'Log Loss': top3_logloss,
    'Brier Score': top3_brier
})

print(f"Top 3 Models: AUC={top3_auc:.4f}, LogLoss={top3_logloss:.4f}, Brier={top3_brier:.4f}")

# ------------------------
# 4️⃣ Stacking Ensemble
# ------------------------
print("\n🏗️ 4️⃣ STACKING ENSEMBLE:")

# Create out-of-fold predictions for stacking
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = np.zeros((len(X_train), len(predictions)))
test_predictions = np.zeros((len(X_test), len(predictions)))

for i, (model_name, model) in enumerate(models.items()):
    print(f"Generating OOF predictions for {model_name}...")
    
    oof_pred = np.zeros(len(X_train))
    test_pred = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        if model_name == 'LogisticRegression':
            # Scale for Logistic Regression
            scaler_fold = StandardScaler()
            X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler_fold.transform(X_fold_val)
            
            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_fold_train_scaled, y_fold_train)
            oof_pred[val_idx] = fold_model.predict_proba(X_fold_val_scaled)[:, 1]
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            test_pred += fold_model.predict_proba(X_test_scaled)[:, 1] / skf.n_splits
        else:
            fold_model = type(model)(**model.get_params())
            fold_model.fit(X_fold_train, y_fold_train)
            oof_pred[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
            test_pred += fold_model.predict_proba(X_test)[:, 1] / skf.n_splits
    
    oof_predictions[:, i] = oof_pred
    test_predictions[:, i] = test_pred

# Train meta-model
print("Training meta-model...")
meta_model = LogisticRegression(
    C=1.0,
    penalty='l2',
    random_state=42,
    max_iter=1000
)

meta_model.fit(oof_predictions, y_train)

# Stacking predictions
val_stacking = meta_model.predict_proba(np.column_stack([pred['val'] for pred in predictions.values()]))[:, 1]
test_stacking = meta_model.predict_proba(test_predictions)[:, 1]

stacking_auc = roc_auc_score(y_val, val_stacking)
stacking_logloss = log_loss(y_val, val_stacking)
stacking_brier = brier_score_loss(y_val, val_stacking)

ensemble_results.append({
    'Method': 'Stacking (Logistic Regression)',
    'ROC-AUC': stacking_auc,
    'Log Loss': stacking_logloss,
    'Brier Score': stacking_brier
})

print(f"Stacking: AUC={stacking_auc:.4f}, LogLoss={stacking_logloss:.4f}, Brier={stacking_brier:.4f}")

# ------------------------
# 5️⃣ Results Comparison
# ------------------------
print("\n📊 5️⃣ ENSEMBLE RESULTS COMPARISON:")

ensemble_df = pd.DataFrame(ensemble_results)
print("\nIndividual Models:")
print(individual_df.round(4).to_string(index=False))

print("\nEnsemble Methods:")
print(ensemble_df.round(4).to_string(index=False))

# Find best method for each metric
best_auc_method = ensemble_df.loc[ensemble_df['ROC-AUC'].idxmax(), 'Method']
best_auc_score = ensemble_df['ROC-AUC'].max()

best_logloss_method = ensemble_df.loc[ensemble_df['Log Loss'].idxmin(), 'Method']
best_logloss_score = ensemble_df['Log Loss'].min()

best_brier_method = ensemble_df.loc[ensemble_df['Brier Score'].idxmin(), 'Method']
best_brier_score = ensemble_df['Brier Score'].min()

print(f"\n🏆 Best Performance:")
print(f"ROC-AUC: {best_auc_method} ({best_auc_score:.4f})")
print(f"Log Loss: {best_logloss_method} ({best_logloss_score:.4f})")
print(f"Brier Score: {best_brier_method} ({best_brier_score:.4f})")

# ------------------------
# 6️⃣ Visualization
# ------------------------
print("\n📈 6️⃣ CREATING VISUALIZATIONS:")

# Performance comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

# ROC-AUC comparison
all_results = pd.concat([individual_df, ensemble_df])
all_results_sorted = all_results.sort_values('ROC-AUC')

sns.barplot(data=all_results_sorted, x='ROC-AUC', y='Method', ax=axes[0, 0])
axes[0, 0].set_title('ROC-AUC Comparison')
axes[0, 0].set_xlabel('ROC-AUC Score')

# Log Loss comparison
all_results_sorted_logloss = all_results.sort_values('Log Loss', ascending=True)
sns.barplot(data=all_results_sorted_logloss, x='Log Loss', y='Method', ax=axes[0, 1])
axes[0, 1].set_title('Log Loss Comparison (Lower is Better)')
axes[0, 1].set_xlabel('Log Loss')

# Correlation of predictions
pred_correlation = pd.DataFrame()
for model_name, pred_dict in predictions.items():
    pred_correlation[model_name] = pred_dict['val']

sns.heatmap(pred_correlation.corr(), annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', ax=axes[1, 0])
axes[1, 0].set_title('Prediction Correlation Matrix')

# Performance improvement
improvement_df = ensemble_df.copy()
improvement_df['AUC_Improvement'] = improvement_df['ROC-AUC'] - individual_df['ROC-AUC'].max()
improvement_df['LogLoss_Improvement'] = individual_df['Log Loss'].min() - improvement_df['Log Loss']

improvement_melted = pd.melt(improvement_df, 
                           id_vars=['Method'], 
                           value_vars=['AUC_Improvement', 'LogLoss_Improvement'],
                           var_name='Metric', value_name='Improvement')

sns.barplot(data=improvement_melted, x='Improvement', y='Method', hue='Metric', ax=axes[1, 1])
axes[1, 1].set_title('Improvement over Best Individual Model')
axes[1, 1].set_xlabel('Improvement')
axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('ensemble_performance.png', dpi=300, bbox_inches='tight')
plt.close()

# ------------------------
# 7️⃣ Final Ensemble Submission
# ------------------------
print("\n🔮 7️⃣ CREATING ENSEMBLE SUBMISSIONS:")

# Create submissions for best performing methods
test_original = pd.read_csv("data/Test.csv")

# Best AUC submission
if best_auc_method == 'Simple Average':
    best_auc_pred = test_avg
elif best_auc_method == 'Weighted Average (AUC)':
    best_auc_pred = test_weighted
elif best_auc_method == 'Weighted Average (Log Loss)':
    best_auc_pred = test_weighted_logloss
elif best_auc_method.startswith('Top 3'):
    best_auc_pred = test_top3
else:
    best_auc_pred = test_stacking

# Best Log Loss submission
if best_logloss_method == 'Simple Average':
    best_logloss_pred = test_avg
elif best_logloss_method == 'Weighted Average (AUC)':
    best_logloss_pred = test_weighted
elif best_logloss_method == 'Weighted Average (Log Loss)':
    best_logloss_pred = test_weighted_logloss
elif best_logloss_method.startswith('Top 3'):
    best_logloss_pred = test_top3
else:
    best_logloss_pred = test_stacking

# Create submissions
submission_best_auc = pd.DataFrame({
    'ID': test_original['ID'],
    'Target_AUC': best_auc_pred,
    'Target_LogLoss': best_auc_pred
})

submission_best_logloss = pd.DataFrame({
    'ID': test_original['ID'],
    'Target_AUC': best_logloss_pred,
    'Target_LogLoss': best_logloss_pred
})

submission_best_auc.to_csv("submission_ensemble_best_auc.csv", index=False)
submission_best_logloss.to_csv("submission_ensemble_best_logloss.csv", index=False)

print(f"✅ Created submission_ensemble_best_auc.csv ({best_auc_method})")
print(f"✅ Created submission_ensemble_best_logloss.csv ({best_logloss_method})")

# ------------------------
# 📊 Final Summary
# ------------------------
print("\n" + "=" * 60)
print("🎯 ENSEMBLE MODELING SUMMARY")
print("=" * 60)

print(f"\n📈 BEST INDIVIDUAL MODEL:")
best_individual = individual_df.loc[individual_df['ROC-AUC'].idxmax()]
print(f"{best_individual['Model']}: AUC={best_individual['ROC-AUC']:.4f}, LogLoss={best_individual['Log Loss']:.4f}")

print(f"\n🏆 BEST ENSEMBLE:")
print(f"ROC-AUC: {best_auc_method} ({best_auc_score:.4f})")
print(f"Log Loss: {best_logloss_method} ({best_logloss_score:.4f})")

print(f"\n📊 IMPROVEMENT:")
auc_improvement = best_auc_score - best_individual['ROC-AUC']
logloss_improvement = best_individual['Log Loss'] - best_logloss_score
print(f"AUC Improvement: +{auc_improvement:.4f}")
print(f"Log Loss Improvement: -{logloss_improvement:.4f}")

print(f"\n💾 OUTPUT FILES:")
print(f"- submission_ensemble_best_auc.csv")
print(f"- submission_ensemble_best_logloss.csv")
print(f"- ensemble_performance.png")

print(f"\n✅ Ensemble modeling complete!")
print("=" * 60)
