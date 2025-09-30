import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier
import lightgbm as lgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance

# =====================================
# 0. 데이터 로드
# =====================================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_FILE_train = BASE_DIR / "data" / "processed" / "train_v1.csv"
DATA_FILE_test = BASE_DIR / "data" / "processed" / "test_v1.csv"

train_df = pd.read_csv(DATA_FILE_train)
test_df = pd.read_csv(DATA_FILE_test)

# 불필요 컬럼 제거
drop_cols = ["Unnamed: 0", "line", "name", "mold_name", "working", "date", "time"]
train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

# =====================================
# 1. object / numeric 분리 및 전처리
# =====================================
object_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = train_df.select_dtypes(exclude=["object"]).columns.tolist()

# 후처리용 컬럼은 모델 학습에서 제외
exclude_cols = ["tryshot_signal", "emergency_stop"]
for col in exclude_cols:
    if col in object_cols:
        object_cols.remove(col)

# 타겟 변수 제외
if "passorfail" in numeric_cols:
    numeric_cols.remove("passorfail")

# 라벨 인코딩 (object 컬럼만)
label_encoders = {}
for col in object_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    # test_df에 없는 라벨이 있을 수 있으므로 안전하게 처리
    test_df[col] = test_df[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    label_encoders[col] = le

# =====================================
# 2. 데이터 준비
# =====================================
feature_cols = numeric_cols + object_cols
X = train_df[feature_cols]
y = train_df["passorfail"].astype(int)

# Train / Validation 분리
X_train_full, X_valid, y_train_full, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE 적용 (훈련 데이터에만)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_full, y_train_full)

# 표준화 (numeric만)
scaler = StandardScaler()
X_train_res[numeric_cols] = scaler.fit_transform(X_train_res[numeric_cols])
X_valid[numeric_cols] = scaler.transform(X_valid[numeric_cols])
test_df_scaled = test_df[feature_cols].copy()
test_df_scaled[numeric_cols] = scaler.transform(test_df_scaled[numeric_cols])

# numpy array 변환
X_train_scaled = X_train_res.values
X_valid_scaled = X_valid.values
X_test_scaled = test_df_scaled.values

# =====================================
# 3. Optuna 목적 함수 정의
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
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
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
# 4. Optuna 실행
# =====================================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print("Best trial parameters:", study.best_trial.params)

# =====================================
# 5. 최종 모델 학습
# =====================================
best_params = study.best_trial.params
final_model = LGBMClassifier(**best_params)
final_model.fit(X_train_scaled, y_train_res)

# =====================================
# 6. 최적 threshold 계산 (Train 기준)
# =====================================
y_train_prob = final_model.predict_proba(X_train_scaled)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.01)
scores = [f1_score(y_train_res, y_train_prob > t) for t in thresholds]
best_thresh = thresholds[np.argmax(scores)]
print("Optimal threshold (Train):", best_thresh)

# =====================================
# 7. 테스트셋 예측 및 후처리
# =====================================
y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_prob > best_thresh).astype(int)

# test_df에 합치기
test_df["pred_passorfail"] = y_test_pred

# 1) tryshot_signal이 "D"이면 불량 1로
test_df.loc[test_df["tryshot_signal"] == "D", "pred_passorfail"] = 1

# 2) emergency_stop이 결측치면 불량 1로
test_df.loc[test_df["emergency_stop"].isna(), "pred_passorfail"] = 1

# =====================================
# 8. 성능 평가 (실제 레이블이 있을 경우)
# =====================================
if "passorfail" in test_df.columns:
    y_test_true = test_df["passorfail"].astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test_true, test_df["pred_passorfail"]).ravel()
    accuracy = accuracy_score(y_test_true, test_df["pred_passorfail"])
    f1 = f1_score(y_test_true, test_df["pred_passorfail"])
    precision = precision_score(y_test_true, test_df["pred_passorfail"])
    recall = recall_score(y_test_true, test_df["pred_passorfail"])
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
# 9. Feature Importance (LightGBM)
# =====================================
importances = final_model.feature_importances_
feat_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feat_importance.head(20), palette="viridis")
plt.title("Top 20 Feature Importances (LightGBM)")
plt.tight_layout()
plt.show()

# =====================================
# 10. Permutation Importance (테스트셋)
# =====================================

if "passorfail" in test_df.columns:
    perm_importance_test = permutation_importance(
        final_model,
        X_test_scaled,   # 테스트셋 특징
        y_test_true,     # 테스트셋 정답 라벨
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    perm_df_test = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": perm_importance_test.importances_mean,
        "importance_std": perm_importance_test.importances_std
    }).sort_values(by="importance_mean", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="importance_mean", y="feature", data=perm_df_test.head(20), palette="viridis")
    plt.title("Top 20 Permutation Importances (Test set)")
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Test dataset에 실제 레이블(passorfail)이 없어 permutation importance 계산 불가합니다.")
