from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
import optuna
import warnings
warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTENC, SMOTE

# ---------------------------
# 설정
# ---------------------------
RANDOM_STATE = 2025
N_TRIALS = 40
N_FOLDS = 5 
STUDY_NAME = "logreg_optuna_smotenc_nopipe"
OUT_DIR_NAME = "outputs"

# ---------------------------
# 경로 및 데이터 로드
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v1.csv"
DATA_FILE_TEST = BASE_DIR / "data" / "processed" / "test_v1.csv"

train_df = pd.read_csv(DATA_FILE_TRAIN)
test_df = pd.read_csv(DATA_FILE_TEST)

train_df.drop(columns=["date", "time"], inplace=True)
train_df.drop(columns=["Unnamed: 0"], inplace=True)
test_df.drop(columns=["date", "time"], inplace=True)
test_df.drop(columns=["Unnamed: 0"], inplace=True)

# ---------------------------
# 타겟 분리 및 컬럼 구분
# ---------------------------
target_col = "passorfail"

y_train = train_df[target_col].astype(int).copy()
X_train = train_df.drop(columns=[target_col]).copy()

y_test = test_df[target_col].astype(int).copy()
X_test = test_df.drop(columns=[target_col]).copy()

# 숫자/범주형 분리
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# 카테고리 인덱스(SMOTENC에 사용될 인덱스: numeric 먼저, 그 다음 카테고리)
n_num = len(numeric_cols)
n_cat = len(categorical_cols)
cat_indices_for_smote = np.arange(n_num, n_num + n_cat, dtype=int)

# ---------------------------
# 전처리 객체(수동 사용)
# ---------------------------
# pre-smote: 결측치 보간 + ordinal encoding (SMOTENC 입력은 정수 카테고리여야 함)
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="__missing__")
# OrdinalEncoder: unknown_value 파라미터가 있는 sklearn 버전 사용 (없으면 오류 날 수 있음)
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

# post-smote: numeric scaler + one-hot for categorical
scaler = StandardScaler()
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

# ---------------------------
# CV 준비
# ---------------------------
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ---------------------------
# Optuna objective (각 fold에서 수동 전처리 -> SMOTENC -> 학습 -> 평가)
# ---------------------------
def objective(trial):
    # 하이퍼파라미터 탐색 공간
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    C = trial.suggest_loguniform("C", 1e-4, 1e2)
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])
    max_iter = trial.suggest_categorical("max_iter", [500, 1000, 2000])
    tol = trial.suggest_loguniform("tol", 1e-6, 1e-2)
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    aucs = []
    # fold loop (수동)
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr_df = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx].to_numpy()
        X_val_df = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx].to_numpy()

        # ---------- pre-smote on train fold ----------
        # numeric
        if n_num > 0:
            X_tr_num = num_imputer.fit_transform(X_tr_df[numeric_cols])
        else:
            X_tr_num = np.empty((len(X_tr_df), 0))
        # categorical -> impute then ordinal encode
        if n_cat > 0:
            X_tr_cat_raw = cat_imputer.fit_transform(X_tr_df[categorical_cols])
            X_tr_cat_int = ord_enc.fit_transform(X_tr_cat_raw).astype(int)
        else:
            X_tr_cat_int = np.empty((len(X_tr_df), 0))

        # concat numeric + cat_int for SMOTENC input
        X_tr_pre = np.hstack([X_tr_num, X_tr_cat_int]) if (n_num + n_cat) > 0 else np.empty((len(X_tr_df), 0))

        # ---------- oversample (SMOTENC or SMOTE) ----------
        try:
            if n_cat > 0:
                sm = SMOTENC(categorical_features=cat_indices_for_smote, random_state=RANDOM_STATE)
            else:
                sm = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = sm.fit_resample(X_tr_pre, y_tr)
        except Exception as e:
            # SMOTE 실패(예: 소수 클래스 부족) 시 해당 fold를 실패로 처리하여 낮은 점수 반환
            print(f"[SMOTE error in fold] {e}")
            return 0.0

        # ---------- post-smote fit (fit scaler & onehot on resampled train) ----------
        if n_num > 0:
            X_res_num = X_res[:, :n_num]
            X_train_scaled = scaler.fit_transform(X_res_num)
        else:
            X_train_scaled = np.empty((X_res.shape[0], 0))

        if n_cat > 0:
            X_res_cat_int = X_res[:, n_num:].astype(int)
            X_train_cat_ohe = onehot.fit_transform(X_res_cat_int)
        else:
            X_train_cat_ohe = np.empty((X_res.shape[0], 0))

        X_train_final = np.hstack([X_train_scaled, X_train_cat_ohe]) if (X_train_scaled.size + X_train_cat_ohe.size) > 0 else np.empty((X_res.shape[0], 0))

        # ---------- prepare validation set using transformers fitted on train-resampled ----------
        # pre-transform validation with pre-smote imputers/ordinal (note: ordinal/num imputers were fit on train fold, not resampled)
        if n_num > 0:
            X_val_num = num_imputer.transform(X_val_df[numeric_cols])
            X_val_num_scaled = scaler.transform(X_val_num)  # scaler fitted on resampled numeric
        else:
            X_val_num_scaled = np.empty((len(X_val_df), 0))

        if n_cat > 0:
            X_val_cat_raw = cat_imputer.transform(X_val_df[categorical_cols])
            # ordinal encoder was fit on X_tr_cat_raw (train fold), use transform (unknown -> -1)
            X_val_cat_int = ord_enc.transform(X_val_cat_raw).astype(int)
            # onehot was fit on resampled categories; transform will handle unseen categories (ignored)
            X_val_cat_ohe = onehot.transform(X_val_cat_int)
        else:
            X_val_cat_ohe = np.empty((len(X_val_df), 0))

        X_val_final = np.hstack([X_val_num_scaled, X_val_cat_ohe]) if (X_val_num_scaled.size + X_val_cat_ohe.size) > 0 else np.empty((len(X_val_df), 0))

        # ---------- classifier training on resampled train ----------
        clf_kwargs = {
            "solver": "saga",
            "penalty": penalty,
            "C": C,
            "class_weight": class_weight,
            "max_iter": max_iter,
            "tol": tol,
            "random_state": RANDOM_STATE,
        }
        if penalty == "elasticnet":
            clf_kwargs["l1_ratio"] = l1_ratio

        clf = LogisticRegression(**clf_kwargs)
        try:
            clf.fit(X_train_final, y_res)
        except Exception as e:
            print(f"[Classifier fit error] {e}")
            return 0.0

        # predict prob on validation
        try:
            y_val_proba = clf.predict_proba(X_val_final)[:, 1]
        except Exception as e:
            # predict_proba not available (shouldn't happen for LogisticRegression), fall back
            print(f"[predict_proba error] {e}")
            y_val_proba = clf.decision_function(X_val_final)

        # compute ROC-AUC (if only one class present in val, skip fold)
        try:
            auc = roc_auc_score(y_val, y_val_proba)
        except Exception as e:
            print(f"[ROC-AUC error in fold] {e}")
            return 0.0

        aucs.append(auc)

    # 평균 AUC 반환
    mean_auc = float(np.mean(aucs)) if len(aucs) > 0 else 0.0
    return mean_auc

# ---------------------------
# Optuna study 실행
# ---------------------------
OUT_DIR = BASE_DIR / OUT_DIR_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
storage_url = f"sqlite:///{OUT_DIR / (STUDY_NAME + '.db')}"

study = optuna.create_study(
    study_name=STUDY_NAME,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    storage=storage_url,
    load_if_exists=True
)

print(f"Start Optuna: n_trials={N_TRIALS}")
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
t_elapsed = time.time() - t0
print(f"Optuna done in {t_elapsed:.1f}s")
print("Best params:", study.best_trial.params)
print(f"Best CV ROC-AUC: {study.best_value:.5f}")

# ---------------------------
# 최적 파라미터로 전체 학습 및 테스트 평가 (파이프라인 없이 수동 실행)
# ---------------------------
best = study.best_trial.params.copy()
penalty = best.get("penalty", "l2")
C = best.get("C", 1.0)
class_weight = best.get("class_weight", None)
max_iter = best.get("max_iter", 1000)
tol = best.get("tol", 1e-4)
l1_ratio = best.get("l1_ratio", None)

# 1) fit pre-smote imputers & ord on full train
if n_num > 0:
    X_train_num = num_imputer.fit_transform(X_train[numeric_cols])
else:
    X_train_num = np.empty((len(X_train), 0))

if n_cat > 0:
    X_train_cat_raw = cat_imputer.fit_transform(X_train[categorical_cols])
    X_train_cat_int = ord_enc.fit_transform(X_train_cat_raw).astype(int)
else:
    X_train_cat_int = np.empty((len(X_train), 0))

X_train_pre = np.hstack([X_train_num, X_train_cat_int]) if (n_num + n_cat) > 0 else np.empty((len(X_train), 0))

# 2) SMOTENC/SMOTE on full train
if n_cat > 0:
    sm = SMOTENC(categorical_features=cat_indices_for_smote, random_state=RANDOM_STATE)
else:
    sm = SMOTE(random_state=RANDOM_STATE)

X_res_all, y_res_all = sm.fit_resample(X_train_pre, y_train.to_numpy())

# 3) post-smote fit scaler & onehot on resampled data
if n_num > 0:
    X_res_num = X_res_all[:, :n_num]
    scaler.fit(X_res_num)
    X_train_scaled = scaler.transform(X_res_num)
else:
    X_train_scaled = np.empty((X_res_all.shape[0], 0))

if n_cat > 0:
    X_res_cat_int = X_res_all[:, n_num:].astype(int)
    onehot.fit(X_res_cat_int)
    X_train_cat_ohe = onehot.transform(X_res_cat_int)
else:
    X_train_cat_ohe = np.empty((X_res_all.shape[0], 0))

X_train_final = np.hstack([X_train_scaled, X_train_cat_ohe]) if (X_train_scaled.size + X_train_cat_ohe.size) > 0 else np.empty((X_res_all.shape[0], 0))

# 4) final classifier fit
clf_kwargs = {
    "solver": "saga",
    "penalty": penalty,
    "C": C,
    "class_weight": class_weight,
    "max_iter": max_iter,
    "tol": tol,
    "random_state": RANDOM_STATE,
}
if penalty == "elasticnet":
    clf_kwargs["l1_ratio"] = l1_ratio

final_clf = LogisticRegression(**clf_kwargs)
final_clf.fit(X_train_final, y_res_all)

# 5) transform test set (use pre-fit num_imputer/cat_imputer/ord_enc + scaler + onehot)
if n_num > 0:
    X_test_num = num_imputer.transform(X_test[numeric_cols])
    X_test_num_scaled = scaler.transform(X_test_num)
else:
    X_test_num_scaled = np.empty((len(X_test), 0))

if n_cat > 0:
    X_test_cat_raw = cat_imputer.transform(X_test[categorical_cols])
    X_test_cat_int = ord_enc.transform(X_test_cat_raw).astype(int)
    X_test_cat_ohe = onehot.transform(X_test_cat_int)
else:
    X_test_cat_ohe = np.empty((len(X_test), 0))

X_test_final = np.hstack([X_test_num_scaled, X_test_cat_ohe]) if (X_test_num_scaled.size + X_test_cat_ohe.size) > 0 else np.empty((len(X_test), 0))

# evaluate
y_test_proba = final_clf.predict_proba(X_test_final)[:, 1]
test_roc_auc = roc_auc_score(y_test.to_numpy(), y_test_proba)
print("\n--- Final Test ROC-AUC ---")
print(f"ROC-AUC: {test_roc_auc:.4f}")

# ---------------------------
# 모델 및 전처리 객체 저장
# ---------------------------
save_obj = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "num_imputer": num_imputer,
    "cat_imputer": cat_imputer,
    "ordinal_encoder": ord_enc,
    "scaler": scaler,
    "onehot": onehot,
    "classifier": final_clf,
    "smote_info": {"type": "SMOTENC" if n_cat > 0 else "SMOTE", "cat_indices": cat_indices_for_smote.tolist()},
}

OUT_MODEL = OUT_DIR / "logreg_optuna_smotenc_nopipeline_joblib.pkl"
joblib.dump(save_obj, OUT_MODEL)
print(f"Saved model+preprocessors to: {OUT_MODEL}")

# save study info
with open(OUT_DIR / "optuna_best_params.json", "w") as f:
    json.dump(study.best_trial.params, f, indent=2)
study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)
print(f"Saved Optuna study artifacts to: {OUT_DIR}")
