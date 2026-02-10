Indent mode
Indent size
Line wrap mode
Editing main.py file contents
Selection deleted
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
ate_90, on='trainer')

param_dist = {
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'num_leaves': [20, 50, 100, 200],
    'max_depth': [3, 6, 10, 15],
    'min_child_samples': [5, 10, 20, 30],
    'n_estimators': [500, 1000, 1500, 2000],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

def train_model(X, y):
    lgb_model = LGBMClassifier(objective='binary', random_state=42, verbose=-1)
    random_search = RandomizedSearchCV(
        lgb_model, param_distributions=param_dist, n_iter=30, cv=skf, scoring='roc_auc', random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    best_params = random_search.best_params_
    print("Best params:", best_params)
    best_model = LGBMClassifier(**best_params, objective='binary', random_state=42, verbose=-1)
    best_model.fit(X, y)
    return best_model

# Train models for each time window
model_07 = train_model(X, y_07)
model_90 = train_model(X, y_90)
model_120 = train_model(X, y_120)

# Generate Test Predictions
probs_07 = model_07.predict_proba(X_test)[:, 1]
probs_90 = model_90.predict_proba(X_test)[:, 1]
probs_120 = model_120.predict_proba(X_test)[:, 1]

# Calibration
calibrated_07 = CalibratedClassifierCV(model_07, method='sigmoid', cv=skf)
calibrated_07.fit(X, y_07)
calibrated_probs_07 = calibrated_07.predict_proba(X_test)[:, 1]

calibrated_90 = CalibratedClassifierCV(model_90, method='sigmoid', cv=skf)
calibrated_90.fit(X, y_90)
calibrated_probs_90 = calibrated_90.predict_proba(X_test)[:, 1]

calibrated_120 = CalibratedClassifierCV(model_120, method='sigmoid', cv=skf)
calibrated_120.fit(X, y_120)
calibrated_probs_120 = calibrated_120.predict_proba(X_test)[:, 1]

# Clip probabilities to ensure they are within [0, 1]
eps = 1e-6
calibrated_probs_07 = np.clip(calibrated_probs_07, eps, 1 - eps)
calibrated_probs_90 = np.clip(calibrated_probs_90, eps, 1 - eps)
calibrated_probs_120 = np.clip(calibrated_probs_120, eps, 1 - eps)

# Create Submission File
submission = pd.DataFrame({
    'ID': test['ID'],
    'Target_07_AUC': calibrated_probs_07,
    'Target_07_LogLoss': calibrated_probs_07,
    'Target_90_AUC': calibrated_probs_90,
    'Target_90_LogLoss': calibrated_probs_90,
    'Target_120_AUC': calibrated_probs_120,
    'Target_120_LogLoss': calibrated_probs_120
})

# Save to CSV
submission.to_csv('submission_final_v3.csv', index=False)
print("✅ Submission file 'submission_final_v3.csv' created successfully!")
print(f"Submission shape: {submission.shape}")

