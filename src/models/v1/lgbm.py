import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# =====================================
# 0. 데이터 로드
# =====================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE_train = BASE_DIR / "data" / "processed" / "train_v1.csv"
DATA_FILE_test = BASE_DIR / "data" / "processed" / "test_v1.csv"

train_df = pd.read_csv(DATA_FILE_train)
test_df = pd.read_csv(DATA_FILE_test)

train_df.info()
test_df.info()

# =====================================
# 1. 데이터 준비
# =====================================
feature_cols = [
    "count", "facility_operation_cycleTime", "production_cycletime",
    "low_section_speed", "high_section_speed", "cast_pressure",
    "biscuit_thickness", "upper_mold_temp1", "upper_mold_temp2",
    "lower_mold_temp1", "lower_mold_temp2", "sleeve_temperature",
    "physical_strength", "Coolant_temperature", "EMS_operation_time",
    "mold_code", "molten_temp"
]

X = train_df[feature_cols]
y = train_df["passorfail"].astype(int)

# Train / Validation 분리
X_train_full, X_valid, y_train_full, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE 적용 (훈련 데이터에만)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)

# 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_valid_scaled = scaler.transform(X_valid)

# 새 테스트셋
X_test_scaled = scaler.transform(test_df[feature_cols])

# =====================================
# 2. Optuna 목적 함수 정의
# =====================================
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds, trues = [], []

    for train_idx, valid_idx in skf.split(X_train_scaled, y_train_res):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[valid_idx]
        y_train_fold, y_val_fold = y_train_res.iloc[train_idx], y_train_res.iloc[valid_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric="binary_logloss",
            verbose=-1
        )

        y_pred_prob = model.predict_proba(X_val_fold)[:, 1]
        preds.extend(y_pred_prob)
        trues.extend(y_val_fold)

    # threshold 최적화 (F1 기준)
    best_thresh, best_f1 = 0.5, -1
    for thresh in np.arange(0.1, 0.9, 0.01):
        f1 = f1_score(trues, np.array(preds) > thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_f1

# =====================================
# 3. Optuna 실행
# =====================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best trial parameters:", study.best_trial.params)

# =====================================
# 4. 최종 모델 학습
# =====================================
best_params = study.best_trial.params
final_model = LGBMClassifier(**best_params)
final_model.fit(X_train_scaled, y_train_res)

# =====================================
# 5. 최적 threshold 계산 (Train 기준)
# =====================================
y_train_prob = final_model.predict_proba(X_train_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
scores = [f1_score(y_train_res, y_train_prob > t) for t in thresholds]
best_thresh = thresholds[np.argmax(scores)]
print("Optimal threshold (Train):", best_thresh)

# =====================================
# 6. 테스트셋 성능 평가 (최종 테스트)
# =====================================
y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_prob > best_thresh).astype(int)

if "passorfail" in test_df.columns:
    y_test_true = test_df["passorfail"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
    accuracy = accuracy_score(y_test_true, y_test_pred)
    f1 = f1_score(y_test_true, y_test_pred)
    precision = precision_score(y_test_true, y_test_pred)
    recall = recall_score(y_test_true, y_test_pred)
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_test_true, y_test_prob)

    print("\n=== Test Set Performance ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Confusion Matrix: \nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
else:
    print("Test dataset에 실제 레이블(passorfail)이 없어 성능 평가는 생략됩니다.")

# =====================================
# 7. 테스트 혼동 행렬 시각화
# =====================================
if "passorfail" in test_df.columns:
    cm = confusion_matrix(y_test_true, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted: Normal(0)", "Predicted: Defect(1)"],
                yticklabels=["Actual: Normal(0)", "Actual: Defect(1)"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Test Confusion Matrix")
    plt.show()

# =====================================
# 8. Feature Importance
# =====================================
importance = final_model.feature_importances_
feat_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": importance
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10,6))
plt.barh(feat_imp["feature"][:20][::-1], feat_imp["importance"][:20][::-1])
plt.title("Top 20 Feature Importances")
plt.show()

# =====================================
# 9. Permutation Importance (Test 데이터)
# =====================================
if "passorfail" in test_df.columns:
    perm_importance = permutation_importance(
        final_model, X_test_scaled, y_test_true,
        n_repeats=10, random_state=42, scoring='f1'
    )

    perm_feat_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_importance.importances_mean,
        "importance_std": perm_importance.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(
        perm_feat_imp["feature"][:20][::-1],
        perm_feat_imp["importance_mean"][:20][::-1],
        xerr=perm_feat_imp["importance_std"][:20][::-1]
    )
    plt.title("Top 20 Permutation Feature Importances (Test Data)")
    plt.xlabel("Mean Decrease in F1-score")
    plt.show()
else:
    print("Test dataset에 실제 레이블(passorfail)이 없어 Permutation Importance를 계산할 수 없습니다.")
