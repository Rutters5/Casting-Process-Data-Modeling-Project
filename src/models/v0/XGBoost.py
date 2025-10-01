# xgb_smotenc_f1_opt.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTENC, SMOTE
from collections import Counter
import joblib
import multiprocessing
import matplotlib.pyplot as plt
from statistics import median
from xgboost import XGBClassifier

# ---------------------------
# 설정
# ---------------------------
RANDOM_STATE = 2025
N_TRIALS = 40
N_FOLDS = 5

CPU_COUNT = multiprocessing.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1)

# ---------------------------
# 파일 경로
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v0.csv"
DATA_FILE_TEST = BASE_DIR / "data" / "processed" / "test_v0.csv"
MODEL_OUTPUT = BASE_DIR / "data" / "models" / "v0" / "XGBoost_v0.pkl"
SCORE_OUTPUT = BASE_DIR / "data" / "models" / "v0" / "XGBoost_v0_scores.csv"


# ---------------------------
# 데이터 로드
# ---------------------------
train_df = pd.read_csv(DATA_FILE_TRAIN)
test_df = pd.read_csv(DATA_FILE_TEST)

train_df.drop(columns=["date", "time", "Unnamed: 0"], inplace=True, errors='ignore')
test_df.drop(columns=["date", "time", "Unnamed: 0"], inplace=True, errors='ignore')

target_col = "passorfail"

y_train = train_df[target_col].astype(int).copy()
X_train = train_df.drop(columns=[target_col]).copy()

y_test = test_df[target_col].astype(int).copy()
X_test = test_df.drop(columns=[target_col]).copy()

# 수치형/범주형 컬럼 분리
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()


# ---------------------------
# 유틸 함수
# pr_auc_score = PR-AUC
# find_best_threshold_for_f1 = f1기준 threshold 찾기
# ---------------------------
def pr_auc_score(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    return auc(recall, precision)

def find_best_threshold_for_f1(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    best_f1 = -1.0
    best_thr = 0.5
    best_prec = 0.0
    best_rec = 0.0

    # 모든 threshold에 대해 F1 계산하고 최대 F1인 threshold 선택
    for i, thr in enumerate(thresholds):
        prec_i = precision[i + 1]
        rec_i = recall[i + 1]
        if prec_i + rec_i <= 0:
            continue
        f1_i = 2 * prec_i * rec_i / (prec_i + rec_i)
        if f1_i > best_f1:
            best_f1 = f1_i
            best_thr = thr
            best_prec = prec_i
            best_rec = rec_i

    # thresholds가 없거나 best_f1을 찾지 못한 경우 0.5 기준으로 계산
    if best_f1 < 0:
        pred_bin = (y_scores >= 0.5).astype(int)
        return 0.5, f1_score(y_true, pred_bin, zero_division=0), precision_score(y_true, pred_bin, zero_division=0), recall_score(y_true, pred_bin, zero_division=0)

    return best_thr, best_f1, best_prec, best_rec

# ---------------------------
# Optuna objective
# ---------------------------
def objective(trial):
    # XGBoost 하이퍼파라미터
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "subsample": trial.suggest_uniform("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_loguniform("gamma", 1e-8, 10.0),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 10.0),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "use_label_encoder": False,
        "objective": "binary:logistic",
        "n_jobs": N_JOBS,
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr = X_train.iloc[tr_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train.iloc[tr_idx].copy()
        y_val = y_train.iloc[val_idx].copy()

        # Ordinal encoding (SMOTENC 입력용)
        ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_tr_cat_ord = ord_enc.fit_transform(X_tr[cat_cols]).astype(int)
        X_val_cat_ord = ord_enc.transform(X_val[cat_cols]).astype(int)
        
        # Standard scaling (tree 계열이지만 동일 조건 유지)
        scaler = StandardScaler()
        X_tr_num_scaled = scaler.fit_transform(X_tr[numeric_cols])
        X_val_num_scaled = scaler.transform(X_val[numeric_cols])
        
        # SMOTENC
        X_tr_for_smote = np.hstack([X_tr_num_scaled, X_tr_cat_ord]) if (X_tr_num_scaled.size or X_tr_cat_ord.size) else np.empty((len(X_tr), 0))

        cat_feature_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
        sm = SMOTENC(categorical_features=cat_feature_indices, random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_tr_for_smote, y_tr.values)
        
        X_res_num = X_res[:, :len(numeric_cols)]
        
        X_res_cat_ord = X_res[:, len(numeric_cols):].astype(int)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(X_tr_cat_ord)  # fit on ordinal-coded original train
        X_res_cat_ohe = ohe.transform(X_res_cat_ord)
        X_val_cat_ohe = ohe.transform(X_val_cat_ord)
        
        X_res_final = np.hstack([X_res_num, X_res_cat_ohe]) if (X_res_num.size or X_res_cat_ohe.size) else np.zeros((len(y_res), 0))
        X_val_final = np.hstack([X_val_num_scaled, X_val_cat_ohe]) if (X_val_num_scaled.size or X_val_cat_ohe.size) else np.zeros((len(X_val), 0))

        # XGBoost 학습
        clf = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            gamma=float(params["gamma"]),
            reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]),
            min_child_weight=int(params["min_child_weight"]),
            use_label_encoder=False,
            objective="binary:logistic",
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        clf.fit(X_res_final, y_res)

        # 적합
        if hasattr(clf, "predict_proba"):
            val_probs = clf.predict_proba(X_val_final)[:, 1]
        else:
            val_probs = clf.predict(X_val_final).astype(float)

        # F1을 기준으로 최적 임계값 탐색
        thr, f1_at_thr, prec_at_thr, rec_at_thr = find_best_threshold_for_f1(y_val, val_probs)
        # 폴드별 F1 수집
        fold_f1s.append(f1_at_thr)

    # 폴드별 평균 F1 반환
    return float(np.mean(fold_f1s))

# ---------------------------
# Optuna study
# ---------------------------
sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
study = optuna.create_study(direction="maximize", sampler=sampler, study_name="xgb_f1")
study.optimize(objective, n_trials=N_TRIALS, n_jobs=1, show_progress_bar=True)

# ---------------------------
# CV 기반: 폴드별(원본 validation) threshold 수집 -> median을 운영 임계값으로 사용
# ---------------------------
best_params = study.best_trial.params

final_xgb_params = {
    "n_estimators": int(best_params["n_estimators"]),
    "max_depth": int(best_params["max_depth"]),
    "learning_rate": float(best_params["learning_rate"]),
    "subsample": float(best_params["subsample"]),
    "colsample_bytree": float(best_params["colsample_bytree"]),
    "gamma": float(best_params["gamma"]),
    "reg_alpha": float(best_params["reg_alpha"]),
    "reg_lambda": float(best_params["reg_lambda"]),
    "min_child_weight": int(best_params["min_child_weight"]),
    "use_label_encoder": False,
    "objective": "binary:logistic",
    "n_jobs": N_JOBS,
    "random_state": RANDOM_STATE,
    "verbosity": 0,
}

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
fold_thresholds = []
fold_precisions = []
fold_recalls = []
fold_f1s = []

for tr_idx, val_idx in skf.split(X_train, y_train):
    X_tr = X_train.iloc[tr_idx].copy()
    X_val = X_train.iloc[val_idx].copy()
    y_tr = y_train.iloc[tr_idx].copy()
    y_val = y_train.iloc[val_idx].copy()

    # ordinal + scaling

    ord_enc_cv = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_tr_cat_ord = ord_enc_cv.fit_transform(X_tr[cat_cols]).astype(int)
    X_val_cat_ord = ord_enc_cv.transform(X_val[cat_cols]).astype(int)
    
    scaler_cv = StandardScaler()
    X_tr_num_scaled = scaler_cv.fit_transform(X_tr[numeric_cols])
    X_val_num_scaled = scaler_cv.transform(X_val[numeric_cols])
    
    # SMOTENC
    X_tr_for_smote = np.hstack([X_tr_num_scaled, X_tr_cat_ord]) if (X_tr_num_scaled.size or X_tr_cat_ord.size) else np.empty((len(X_tr), 0))
    cat_feature_indices = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
    sm_cv = SMOTENC(categorical_features=cat_feature_indices, random_state=RANDOM_STATE)
    X_res_cv, y_res_cv = sm_cv.fit_resample(X_tr_for_smote, y_tr.values)
    
    # split + onehot
    X_res_num_cv = X_res_cv[:, :len(numeric_cols)]
    
    X_res_cat_ord_cv = X_res_cv[:, len(numeric_cols):].astype(int)
    ohe_cv = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ohe_cv.fit(X_tr_cat_ord)
    X_res_cat_ohe_cv = ohe_cv.transform(X_res_cat_ord_cv)
    X_val_cat_ohe_cv = ohe_cv.transform(X_val_cat_ord)
    
    X_res_final_cv = np.hstack([X_res_num_cv, X_res_cat_ohe_cv]) if (X_res_num_cv.size or X_res_cat_ohe_cv.size) else np.zeros((len(y_res_cv), 0))
    X_val_final_cv = np.hstack([X_val_num_scaled, X_val_cat_ohe_cv]) if (X_val_num_scaled.size or X_val_cat_ohe_cv.size) else np.zeros((len(X_val), 0))

    # XGBoost 학습
    clf_cv = XGBClassifier(**final_xgb_params)
    clf_cv.fit(X_res_final_cv, y_res_cv)

    # 적합
    if hasattr(clf_cv, "predict_proba"):
        val_probs_cv = clf_cv.predict_proba(X_val_final_cv)[:, 1]
    else:
        val_probs_cv = clf_cv.predict(X_val_final_cv).astype(float)

    # F1을 기준으로 최적 임계값 탐색
    thr_cv, f1_cv, prec_cv, rec_cv = find_best_threshold_for_f1(y_val, val_probs_cv)

    fold_thresholds.append(thr_cv)
    fold_precisions.append(prec_cv)
    fold_recalls.append(rec_cv)
    fold_f1s.append(f1_cv)

print("Per-fold thresholds (F1-opt):", fold_thresholds)
print("Per-fold precision@selected_thr:", fold_precisions)
print("Per-fold recall@selected_thr:", fold_recalls)
print("Per-fold f1@selected_thr:", fold_f1s)

# f1 기준 최적 임계값의 중앙값을 임계값으로 사용
thr_oper = float(median(fold_thresholds))
print(f"Selected operating threshold (median of folds): {thr_oper:.6f}")

# ---------------------------
# 전처리
# ---------------------------
ord_enc_full = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_cat_ord_full = ord_enc_full.fit_transform(X_train[cat_cols]).astype(int)
X_test_cat_ord_full = ord_enc_full.transform(X_test[cat_cols]).astype(int)

scaler_full = StandardScaler()
X_train_num_scaled_full = scaler_full.fit_transform(X_train[numeric_cols])
X_test_num_scaled_full = scaler_full.transform(X_test[numeric_cols])

X_train_for_smote_full = np.hstack([X_train_num_scaled_full, X_train_cat_ord_full]) if (X_train_num_scaled_full.size or X_train_cat_ord_full.size) else np.empty((len(X_train), 0))

cat_feature_indices_full = list(range(len(numeric_cols), len(numeric_cols) + len(cat_cols)))
sm_full = SMOTENC(categorical_features=cat_feature_indices_full, random_state=RANDOM_STATE)
X_res_full, y_res_full = sm_full.fit_resample(X_train_for_smote_full, y_train.values)

print("Resampled full train distribution:", Counter(y_res_full))

# 분리: numeric / ordinal-cat
X_res_num_full = X_res_full[:, :len(numeric_cols)]

X_res_cat_ord_full = X_res_full[:, len(numeric_cols):].astype(int)
ohe_full = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ohe_full.fit(X_train_cat_ord_full)
X_res_cat_ohe_full = ohe_full.transform(X_res_cat_ord_full)
X_test_cat_ohe_full = ohe_full.transform(X_test_cat_ord_full)

X_res_final_full = np.hstack([X_res_num_full, X_res_cat_ohe_full]) if (X_res_num_full.size or X_res_cat_ohe_full.size) else np.zeros((len(y_res_full), 0))
X_test_final = np.hstack([X_test_num_scaled_full, X_test_cat_ohe_full]) if (X_test_num_scaled_full.size or X_test_cat_ohe_full.size) else np.zeros((len(X_test), 0))

# ---------------------------
# 최종 XGBoost 학습
# ---------------------------
clf_final = XGBClassifier(**final_xgb_params)
clf_final.fit(X_res_final_full, y_res_full)

# ---------------------------
# 테스트 예측
# ---------------------------
if hasattr(clf_final, "predict_proba"):
    y_test_proba = clf_final.predict_proba(X_test_final)[:, 1]
else:
    y_test_proba = clf_final.predict(X_test_final).astype(float)

y_test_pred_oper = (y_test_proba >= thr_oper).astype(int)

# tryshot_signal이 D인 경우 무조건 1로 예측
X_test_with_pred = X_test.copy()
X_test_with_pred["pred"] = y_test_pred_oper
if "tryshot_signal" in X_test_with_pred.columns:
    X_test_with_pred.loc[X_test_with_pred["tryshot_signal"] == "D", "pred"] = 1
y_test_pred_oper = X_test_with_pred["pred"].values


# ---------------------------
# 테스트 평가
# ---------------------------
test_pr_auc = pr_auc_score(y_test, y_test_proba)
test_roc_auc = roc_auc_score(y_test, y_test_proba) if len(np.unique(y_test)) == 2 else np.nan
test_acc = accuracy_score(y_test, y_test_pred_oper)
test_prec = precision_score(y_test, y_test_pred_oper, zero_division=0)
test_rec = recall_score(y_test, y_test_pred_oper, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred_oper, zero_division=0)
cm = confusion_matrix(y_test, y_test_pred_oper)

print("\n== Test set performance at operating threshold (XGBoost) ==")
print(f"Operating threshold: {thr_oper:.6f}")
print(f"PR-AUC: {test_pr_auc:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}, Recall: {test_rec:.4f}, F1: {test_f1:.4f}")
print("Confusion matrix:\n", cm)

# ---------------------------
# 모델 저장
# ---------------------------
artifact_obj = {
    "model": clf_final,
    "scaler": (scaler_full if numeric_cols else None),
    "ordinal_encoder": (ord_enc_full if cat_cols else None),
    "onehot_encoder": (ohe_full if cat_cols else None),
    "best_params": study.best_trial.params,
    "operating_threshold": thr_oper
}
joblib.dump(artifact_obj, MODEL_OUTPUT, compress=3)

# ---------------------------
# 점수 저장 (CSV)
# ---------------------------
pd.DataFrame([{
    "pr_auc": test_pr_auc,
    "roc_auc": test_roc_auc,
    "accuracy": test_acc,
    "precision": test_prec,
    "recall": test_rec,
    "f1_score": test_f1,
    "operating_threshold": thr_oper
}]).to_csv(SCORE_OUTPUT, index=False)
