from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
# 병렬/스레드 과다 사용 제어(BLAS/OpenMP)
from joblib import parallel_backend
from threadpoolctl import threadpool_limits


# ---------------------------
# 파일 경로 설정
# ---------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"

# ---------------------------
# 데이터 로드
# ---------------------------
df = pd.read_csv(DATA_FILE)
# date/time 컬럼은 모델 학습에 사용하지 않는다고 판단되어 제거
df.drop(columns=["date", "time"], inplace=True)

# ---------------------------
# X, y 선택
# ---------------------------
target_col = "passorfail"
X = df.drop(columns=[target_col])
y = df[target_col].copy().astype(int)

# ---------------------------
# 수치형, 범주형 컬럼 분리
# ---------------------------
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# ---------------------------
# train/test 분할
# ---------------------------
# stratify=y : 불량 양품 비율 유지
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------
# 전처리 파이프라인
# ---------------------------
num_pipe = Pipeline([("scaler", StandardScaler())])
cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

transformers = []
transformers.append(("num", num_pipe, numeric_cols))
transformers.append(("cat", cat_pipe, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

# ---------------------------
# Optuna objective
# ---------------------------
def objective(trial):
    solver = trial.suggest_categorical("solver", ["saga", "liblinear", "lbfgs"])
    
    # penalty는 규제 유형: l1(lasso), l2(ridge), elasticnet(혼합)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    
    # C는 규제 강도 역수(작을수록 강한 규제)
    C = trial.suggest_loguniform("C", 1e-4, 1e2)
    
    # elasticnet에서 사용하는 l1_ratio (l1과 l2의 혼합 비율)
    l1_ratio = trial.suggest_float("l1_ratio", 0.05, 0.95)
    
    # 클래스 불균형을 고려한 가중치 사용 여부
    use_balanced = trial.suggest_categorical("class_weight_balanced", [True, False])
    class_weight = "balanced" if use_balanced else None
    max_iter = trial.suggest_categorical("max_iter", [200, 500, 1000])

    # -----------------------
    # Solver와 penalty의 호환성 보정
    # - 일부 solver는 특정 penalty를 지원하지 않음
    # - 예: lbfgs는 elasticnet 불가 -> penalty를 l2로 강제
    # - liblinear는 elasticnet 불가 -> l1으로 대체
    # -----------------------
    if solver == "lbfgs" and penalty != "l2":
        penalty_effective = "l2"
    elif solver == "liblinear" and penalty == "elasticnet":
        penalty_effective = "l1"  # liblinear는 elasticnet 미지원이므로 안전한 대체로 l1 선택
    else:
        penalty_effective = penalty

    # -----------------------
    # LogisticRegression에 전달할 인자 딕셔너리 구성
    # -----------------------
    lr_kwargs = {
        "C": C,
        "solver": solver,
        "penalty": penalty_effective,
        "class_weight": class_weight,
        "max_iter": max_iter,
        "random_state": 42,
    }
    # elasticnet인 경우에만 l1_ratio 전달 (다른 penalty에서는 무시)
    if penalty_effective == "elasticnet":
        lr_kwargs["l1_ratio"] = l1_ratio

    # -----------------------
    # 모델 파이프라인 구성
    # Pipeline 순서:
    #  1. preproc: ColumnTransformer -> 숫자/범주형 전처리 수행
    #  2. clf: LogisticRegression -> 전처리된 데이터를 학습
    # -----------------------
    model = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(**lr_kwargs))])

    # -----------------------
    # 교차검증(CV) 설정
    # - StratifiedKFold: 클래스 비율을 유지하는 K-fold CV
    # - n_splits=5: 5-겹 교차검증 (데이터를 5등분하여 5번 평가)
    # -----------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # scoring은 dict로 주면 cross_validate가 여러 지표를 계산합니다.
    # 여기서는 average_precision(PR AUC)을 'avg_prec'라는 이름으로 계산하게 합니다.
    scoring = {"avg_prec": "average_precision"}

    # ===== 병렬 안전 실행(핵심) =====
    # 이유:
    # - scikit-learn의 cross_validate에 n_jobs=-1(병렬)을 주면 내부적으로 여러 프로세스를 띄웁니다.
    # - 동시에 NumPy/BLAS(LAPACK)/OpenMP 같은 라이브러리가 자체적으로 스레드를 사용하면
    #   시스템 스레드가 폭주(oversubscription)하여 성능 저하 또는 오류가 발생할 수 있습니다.
    # 해결 방법 (여기서 적용한 A 방법):
    # 1) threadpool_limits(limits=1): BLAS/OpenMP 계열 라이브러리가 사용하는 내부 스레드 수를 1로 제한
    # 2) parallel_backend("loky", n_jobs=-1): joblib이 loky(프로세스 기반) 백엔드를 사용해 cross_validate 병렬 실행
    # 3) cross_validate(..., n_jobs=-1): 실제로 fold들을 병렬로 실행
    #
    # 이 조합은 "내부 스레드 + 외부 프로세스"의 충돌을 막아 안정적인 병렬 실행을 가능하게 합니다.
    # (환경에 따라서는 n_jobs를 1로 줄여서 단순화하는 편이 더 안정적일 수 있음)
    with threadpool_limits(limits=1):
        with parallel_backend("loky", n_jobs=-1):
            cv_results = cross_validate(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                error_score="raise"  # 오류 발생시 예외를 일으켜 원인을 바로 알게 함
            )

    # cross_validate의 반환값은 dict이고, "test_avg_prec" 키에 각 fold의 점수가 들어있습니다.
    avg_prec_mean = np.mean(cv_results["test_avg_prec"])

    # Optuna는 objective가 반환한 수치를 기준으로 탐색합니다.
    # study = optuna.create_study(direction='maximize') 로 만들었기 때문에
    # 이 값이 클수록 좋은 하이퍼파라미터로 간주됩니다.
    return avg_prec_mean

# ---------------------------
# Optuna 스터디 생성 및 실행
# ---------------------------
# create_study: 최적화 목표(direction)를 'maximize'로 설정(여기서는 PR AUC 최대화)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

# 몇 번의 trial(시도)을 수행할지 설정
N_TRIALS = 60
print(f"Start optimization with {N_TRIALS} trials...")

# study.optimize로 objective를 반복 실행. show_progress_bar=True는 진행바 표시.
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ---------------------------
# 최적 결과 출력
# ---------------------------
print("Best trial:")
trial = study.best_trial
print("  Value (best avg precision): ", trial.value)
print("  Params: ")
for k, v in trial.params.items():
    print(f"    {k}: {v}")

# ---------------------------
# 베스트 파라미터로 모델 재학습 (동일한 호환성 보정 적용)
# ---------------------------
# study.best_trial.params로 얻은 하이퍼파라미터를 가져와 동일한 보정 규칙을 다시 적용합니다.
best_params = trial.params
solver = best_params.get("solver")
penalty = best_params.get("penalty")
C = best_params.get("C")
class_weight = "balanced" if best_params.get("class_weight_balanced", False) else None
max_iter = best_params.get("max_iter", 1000)
l1_ratio = best_params.get("l1_ratio", None)

if solver == "lbfgs" and penalty != "l2":
    penalty_effective = "l2"
elif solver == "liblinear" and penalty == "elasticnet":
    penalty_effective = "l1"
else:
    penalty_effective = penalty

lr_kwargs = {
    "C": C,
    "solver": solver,
    "penalty": penalty_effective,
    "class_weight": class_weight,
    "max_iter": max_iter,
    "random_state": 42
}
if penalty_effective == "elasticnet" and l1_ratio is not None:
    lr_kwargs["l1_ratio"] = l1_ratio

# 최종 파이프라인 구성 (전처리 + 로지스틱 회귀)
best_model = Pipeline([("preproc", preprocessor), ("clf", LogisticRegression(**lr_kwargs))])

# ---------------------------
# 안전한 학습: BLAS 스레드 제한 적용
# ---------------------------
# 위에서와 동일한 이유로, 학습(특히 내부 BLAS를 사용하는 경우)에 앞서 스레드 수를 제한합니다.
with threadpool_limits(limits=1):
    best_model.fit(X_train, y_train)

# ---------------------------
# 예측 및 점수 계산
# ---------------------------
# predict: 최종 클래스(0/1)를 반환
y_pred = best_model.predict(X_test)

# predict_proba: 클래스일 확률을 반환 (양성 클래스에 대한 확률을 y_score로 사용)
# 모든 분류기가 predict_proba를 제공하는 것은 아니기 때문에 hasattr로 확인합니다.
if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)[:, 1]
else:
    # 일부 선형 분류기는 decision_function을 제공함 -> 점수로 사용 가능
    try:
        y_score = best_model.decision_function(X_test)
    except Exception:
        # 그 외의 경우는 y_pred를 점수로 사용(비추천, 최후의 수단)
        y_score = y_pred

# ---------------------------
# 규칙 기반 오버라이드 (특별 처리)
# ---------------------------
# 스크립트 원문에 있던 규칙: emergency_stop 결측이면 무조건 불량으로 판정
# (전처리 단계에서 제거했을 수 있으니 존재 여부를 먼저 확인)
if "emergency_stop" in X_test.columns:
    mask = X_test["emergency_stop"].isna()
    if mask.any():
        # 예측값을 복사하여 해당 인덱스를 1(불량)로 강제 설정
        y_pred = y_pred.copy()
        y_pred[mask] = 1
        # 확률/점수도 1로 설정하여 메트릭 계산에 반영
        if isinstance(y_score, np.ndarray):
            y_score[mask] = 1.0

# ---------------------------
# 성능 지표 계산 및 출력
# ---------------------------
# ROC AUC: 수신자 조작 특성 곡선 아래 면적(범위 0~1, 클수록 좋음)
roc_auc = roc_auc_score(y_test, y_score)
# PR AUC(average precision): 정밀도-재현율 곡선 아래 면적, 불균형 데이터에서 유용
pr_auc = average_precision_score(y_test, y_score)
# F1: 정밀도와 재현율의 조화평균 (균형 지표)
f1 = f1_score(y_test, y_pred)
# Recall: 실제 양성 중 예측 양성 비율(불량을 놓치지 않는 능력)
recall = recall_score(y_test, y_pred)

print("\n===== Final metrics on test set (best model) =====")
print(f"ROC AUC : {roc_auc:.4f}")
print(f"PR AUC  : {pr_auc:.4f} (average precision)")
print(f"F1      : {f1:.4f}")
print(f"Recall  : {recall:.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# ROC & PR 곡선 시각화
# ---------------------------
# ROC curve: FPR(False Positive Rate) vs TPR(True Positive Rate)
fpr, tpr, _ = roc_curve(y_test, y_score)
# Precision-Recall curve
precision, recall_vals, _ = precision_recall_curve(y_test, y_score)

plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.7)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision, label=f"PR AUC = {pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend()

plt.tight_layout()
plt.show()
