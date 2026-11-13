import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, StratifiedKFold
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

DATA_PATH = Path("../data/telco_churn.csv")
if not DATA_PATH.exists():
    DATA_PATH = Path("data/telco_churn.csv")

OUTPUT_DIR = Path("week7/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

y = df['Churn']
X = df.drop(['customerID', 'Churn'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train churn: {y_train.mean():.3f}, Test churn: {y_test.mean():.3f}")

num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

spw = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {spw:.3f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'clf__n_estimators': randint(100, 600),
    'clf__max_depth': randint(3, 12),
    'clf__learning_rate': uniform(0.01, 0.2),
    'clf__subsample': uniform(0.6, 0.4),
    'clf__colsample_bytree': uniform(0.6, 0.4),
    'clf__min_child_weight': randint(1, 10),
    'clf__gamma': uniform(0, 0.5),
}

print("\n" + "=" * 60)
print("STRATEGY 1: XGBoost + scale_pos_weight")
print("=" * 60)

pipe_spw = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(
        scale_pos_weight=spw,
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric='logloss'
    ))
])

search_spw = RandomizedSearchCV(
    pipe_spw, param_dist,
    n_iter=20, cv=cv, scoring='f1',
    random_state=42, n_jobs=-1, verbose=1
)

search_spw.fit(X_train, y_train)
print(f"Best F1 (CV): {search_spw.best_score_:.4f}")
print(f"Best params: {search_spw.best_params_}")

y_proba_spw = search_spw.best_estimator_.predict_proba(X_test)[:, 1]
y_pred_spw = (y_proba_spw >= 0.5).astype(int)

print("\n--- Test set (threshold=0.5) ---")
print(f"Precision: {precision_score(y_test, y_pred_spw):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_spw):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_spw):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_spw):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_spw):.4f}")

print("\n" + "=" * 60)
print("STRATEGY 2: XGBoost + SMOTE")
print("=" * 60)

pipe_smote = ImbPipeline([
    ('prep', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(
        random_state=42, n_jobs=-1, verbosity=0,
        eval_metric='logloss'
    ))
])

search_smote = RandomizedSearchCV(
    pipe_smote, param_dist,
    n_iter=20, cv=cv, scoring='f1',
    random_state=42, n_jobs=-1, verbose=1
)

search_smote.fit(X_train, y_train)
print(f"Best F1 (CV): {search_smote.best_score_:.4f}")
print(f"Best params: {search_smote.best_params_}")

y_proba_smote = search_smote.best_estimator_.predict_proba(X_test)[:, 1]
y_pred_smote = (y_proba_smote >= 0.5).astype(int)

print("\n--- Test set (threshold=0.5) ---")
print(f"Precision: {precision_score(y_test, y_pred_smote):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_smote):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_smote):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba_smote):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_proba_smote):.4f}")

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)

comparison = pd.DataFrame([
    {
        'Strategy': 'scale_pos_weight',
        'Precision': precision_score(y_test, y_pred_spw),
        'Recall': recall_score(y_test, y_pred_spw),
        'F1': f1_score(y_test, y_pred_spw),
        'ROC-AUC': roc_auc_score(y_test, y_proba_spw),
        'PR-AUC': average_precision_score(y_test, y_proba_spw),
    },
    {
        'Strategy': 'SMOTE',
        'Precision': precision_score(y_test, y_pred_smote),
        'Recall': recall_score(y_test, y_pred_smote),
        'F1': f1_score(y_test, y_pred_smote),
        'ROC-AUC': roc_auc_score(y_test, y_proba_smote),
        'PR-AUC': average_precision_score(y_test, y_proba_smote),
    }
])
print(comparison.round(4).to_string(index=False))
comparison.to_csv(OUTPUT_DIR / "strategy_comparison.csv", index=False)

print("\n" + "=" * 60)
print("PRECISION-RECALL CURVE + THRESHOLD SELECTION")
print("=" * 60)

best_proba = y_proba_spw if f1_score(y_test, y_pred_spw) > f1_score(y_test, y_pred_smote) else y_proba_smote
best_name = "scale_pos_weight" if f1_score(y_test, y_pred_spw) > f1_score(y_test, y_pred_smote) else "SMOTE"
print(f"Best strategy: {best_name}")

precisions, recalls, thresholds = precision_recall_curve(y_test, best_proba)

# Choose threshold that maximizes recall subject to precision >= 0.45
target_precision = 0.45
valid = precisions[:-1] >= target_precision
chosen_threshold = thresholds[valid].min() if valid.any() else 0.5
chosen_idx = np.argmin(np.abs(thresholds - chosen_threshold))

print(f"Chosen threshold: {chosen_threshold:.3f} (target precision >= {target_precision})")
print(f"At this threshold: precision={precisions[chosen_idx]:.4f}, recall={recalls[chosen_idx]:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, color='#3659a7', lw=2, label='PR curve')
plt.axvline(recalls[chosen_idx], color='#dc2626', linestyle='--', label=f'Chosen threshold = {chosen_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve ({best_name})')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=100)
plt.close()
print(f"Saved: {OUTPUT_DIR / 'precision_recall_curve.png'}")

y_pred_threshold = (best_proba >= chosen_threshold).astype(int)

print("\n--- Test set with chosen threshold ---")
print(f"Precision: {precision_score(y_test, y_pred_threshold):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_threshold):.4f}")
print(f"F1:        {f1_score(y_test, y_pred_threshold):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_threshold))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_threshold, target_names=['No Churn', 'Churn']))

print("\n" + "=" * 60)
print("ERROR ANALYSIS: 10 Misclassified Customers")
print("=" * 60)

misclassified = X_test.copy()
misclassified['y_true'] = y_test.values
misclassified['y_pred'] = y_pred_threshold
misclassified['y_proba'] = best_proba

errors = misclassified[misclassified['y_true'] != misclassified['y_pred']].copy()
errors['error_type'] = errors.apply(
    lambda r: 'FN' if r['y_true'] == 1 and r['y_pred'] == 0 else 'FP', axis=1
)

print(f"Total misclassified: {len(errors)} (FN={sum(errors['error_type']=='FN')}, FP={sum(errors['error_type']=='FP')})")

sample_fn = errors[errors['error_type'] == 'FN'].head(5)
sample_fp = errors[errors['error_type'] == 'FP'].head(5)

cols_to_show = ['tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'InternetService', 'PaymentMethod', 'error_type', 'y_proba']
print("\n--- 5 False Negatives (missed churners) ---")
print(sample_fn[cols_to_show].to_string())

print("\n--- 5 False Positives (false alarms) ---")
print(sample_fp[cols_to_show].to_string())

print("\n--- Pattern summary ---")
print(f"FN mean tenure: {sample_fn['tenure'].mean():.1f}, FN mean charges: ${sample_fn['MonthlyCharges'].mean():.2f}")
print(f"FP mean tenure: {sample_fp['tenure'].mean():.1f}, FP mean charges: ${sample_fp['MonthlyCharges'].mean():.2f}")
print(f"FN contract distribution: {sample_fn['Contract'].value_counts().to_dict()}")
print(f"FP contract distribution: {sample_fp['Contract'].value_counts().to_dict()}")

errors[cols_to_show].to_csv(OUTPUT_DIR / "error_analysis_sample.csv", index=False)
print(f"\nSaved: {OUTPUT_DIR / 'error_analysis_sample.csv'}")

print("\nAll outputs saved to:", OUTPUT_DIR)