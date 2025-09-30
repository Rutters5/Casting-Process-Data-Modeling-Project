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

# 1️⃣ 한글 폰트 설정 (Mac, Windows, Linux에 따라 다를 수 있음)
# Windows 예시 (맑은 고딕 사용)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

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


# =====================================
# 11. SHAP (개별 예측 설명)
# =====================================

import shap
import numpy as np

# Boolean Series 생성
mask = y_valid == 1

# X_valid에서 불량 샘플 위치(순서 기준)
positions = np.flatnonzero(mask.values)  # numpy 배열 형태
print(positions)

# 샘플 선택
sample_idx = 110
sample = X_valid.iloc[[sample_idx]]

# TreeExplainer 생성
explainer = shap.TreeExplainer(final_model)

# SHAP 값 (log-odds 기준)
shap_values = np.array(explainer.shap_values(X_valid))

# logit -> probability 변환 함수
def logit_to_prob(logit):
    return 1 / (1 + np.exp(-logit))

# 확률 기준 SHAP 값
shap_values_prob = logit_to_prob(shap_values) - logit_to_prob(explainer.expected_value)
base_prob = logit_to_prob(explainer.expected_value)

# =====================================
# 양/음 기여도 큰 순서대로 정렬
# =====================================
sample_shap = shap_values_prob[sample_idx]
feature_names = sample.columns.tolist()

# 절댓값 기준으로 내림차순 정렬
sorted_idx = np.argsort(-np.abs(sample_shap))
sorted_shap = sample_shap[sorted_idx]
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_values = sample.iloc[0].values[sorted_idx]

# ============================
# 1️⃣ Force Plot
# ============================
shap.initjs()
shap.force_plot(
    base_prob,
    sorted_shap,
    sorted_values,
    feature_names=sorted_features
)

# ============================
# 2️⃣ Waterfall Plot
# ============================
shap_exp = shap.Explanation(
    values=sorted_shap,
    base_values=base_prob,
    data=sorted_values,
    feature_names=sorted_features
)

shap.plots.waterfall(shap_exp, max_display=40)  # 전체 feature 다 보이도록



# 대시보드 연동

# 슬라이더 → 값 변경 → 예측 확률/SHAP 재계산 → 시각화 업데이트

# 예측 확률 게이지(0~1)

# Force Plot/Waterfall Plot 갱신

# 이렇게 하면 특정 값을 변경하면 불량/양품인지 알 수 있고, 값이 변경되었을 때의

# waterfall plot을 보면서 내가 뭘 바꿔야 정상으로 갈 수 있을지 알 수 있지 않을까

# 대신 사용자가 값을 조정할 때는 너무 터무늬없는 값은 못넣도록 구간을 사전에 설정해둬야 할듯



# =====================================
# 12. Validation 기준: Low Section Speed vs 예측 확률 (실제 불량 색상, 원본 값 사용)
# =====================================

# 1️⃣ 스케일링 전 원본 값 가져오기
# X_valid는 스케일링된 상태이므로, 원본 값은 train_test_split 시 나온 X_valid_full에서 가져와야 합니다.
# 만약 X_valid_full이 없으면 X.iloc[val_idx]에서 가져올 수 있음
X_valid_orig = X.loc[X_valid.index, ['low_section_speed']]

# 2️⃣ 모델 예측 확률
y_valid_prob = final_model.predict_proba(X_valid_scaled)[:, 1]

# threshold: 이전에 최적화한 best_thresh 사용
threshold = best_thresh  # 이미 계산된 최적 threshold

plt.figure(figsize=(10,6))
sns.scatterplot(
    x=X_valid_orig['low_section_speed'],  # 원본 값 사용
    y=y_valid_prob,
    hue=y_valid,                          # 실제 불량 기준
    palette={0: "blue", 1: "red"},
    alpha=0.6
)

# threshold 수평선 추가
plt.axhline(y=threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.2f}')

plt.xlabel('Low Section Speed (원본 값)')
plt.ylabel('예측 불량 확률')
plt.title('Validation: Low Section Speed vs 예측 불량 확률 (Actual Pass/Fail + Threshold)')
plt.legend(title="Actual Pass/Fail / Threshold", loc='upper right')
plt.show()

# 저속속도가 낮을수록 불량 확률이 높아짐
# 실제로 waterfall에 사용된 행을 보면 low_section_speed가 10임
X_valid_orig.iloc[57]

# 이거처럼 양으로 영향미친거 3개 음으로 미친거 3개 자동으로 그려지게 대시보드를 만들면 어떨까요



# =====================================
# 13-0. 모델 기반 단일 피처 민감도 탐색 함수 정의
# =====================================
def find_normal_range(model, x_row, feature, feature_min, feature_max, scaler=None, step=0.5, threshold=0.1):
    """
    모델 기반 단일 피처 민감도 탐색 (각 컬럼 실제 범위 반영)
    
    Parameters
    ----------
    model : 학습된 classifier
    x_row : pd.Series
        스케일링 안 된 원본 행
    feature : str
        탐색할 컬럼명
    feature_min, feature_max : float
        컬럼 실제 최소/최대값
    scaler : StandardScaler or None
        numeric 컬럼 스케일링
    step : float
        탐색 스텝
    threshold : float
        불량 확률 기준 (이 값보다 작으면 정상 구간)
    """
    if feature_min >= feature_max:
        return None

    test_vals = np.linspace(feature_min, feature_max, int((feature_max - feature_min) / step) + 1)
    normal_vals = []

    for val in test_vals:
        x_mod = x_row.copy()
        x_mod[feature] = val
        x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
        prob = model.predict_proba(x_input)[0, 1]  # 불량 확률
        if prob < threshold:
            normal_vals.append(val)

    return (min(normal_vals), max(normal_vals)) if normal_vals else None


# =====================================
# 13-1. SHAP 기반 Top 3 feature 선택 및 정상 구간 계산
# =====================================
# sample_idx 기준 SHAP 값 (prob 기준)
sample_shap_abs = shap_values_prob[sample_idx]
feature_names = X.columns.tolist()

# 상위 3개 feature 인덱스 추출
top3_idx = np.argsort(-sample_shap_abs)[:3]
top_features = [feature_names[i] for i in top3_idx]
print("SHAP 기반 Top 3 features:", top_features)

# 각 컬럼별 실제 min/max
feature_ranges = {feat: (X[feat].min(), X[feat].max()) for feat in top_features}

# 정상 구간 탐색
x_row_orig = X.loc[X_valid.index].iloc[sample_idx].copy()  # 스케일링 전 원본
normal_ranges = {}

print("\n=== SHAP 기반 정상 구간 탐색 결과 ===")
for feat in top_features:
    f_min, f_max = feature_ranges[feat]
    normal_ranges[feat] = find_normal_range(
        final_model,
        x_row_orig,
        feat,
        f_min,
        f_max,
        scaler=scaler,
        step=0.5,
        threshold=best_thresh
    )
    print(f"{feat}: 현재={x_row_orig[feat]}, 정상 구간={normal_ranges[feat]}")


# =====================================
# 13-2. Top features 불량 확률 변화 시각화 (독립 figure)
# =====================================
import matplotlib.pyplot as plt

def simulate_top_features_independent(model, x_row, top_features, feature_ranges, normal_ranges, scaler=None, threshold=best_thresh):
    """
    Top features 각각에 대해 불량 확률 변화 및 정상 구간 시각화 (독립된 figure)
    """
    for feat in top_features:
        val_range = feature_ranges[feat]
        test_vals = np.linspace(val_range[0], val_range[1], 50)
        probs = []

        for val in test_vals:
            x_mod = x_row.copy()
            x_mod[feat] = val
            x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
            probs.append(model.predict_proba(x_input)[0, 1])

        # 독립 figure 생성
        plt.figure(figsize=(8,4))
        plt.plot(test_vals, probs, color='red', label='불량 확률')
        plt.axhline(y=threshold, color='green', linestyle='--', label='Threshold')

        # 정상 구간 표시
        normal = normal_ranges[feat]
        if normal is not None:
            plt.axvspan(normal[0], normal[1], color='blue', alpha=0.2, label='정상 구간')

        plt.xlabel(f'{feat} 값')
        plt.ylabel('불량 확률')
        plt.title(f'{feat} 변화에 따른 불량 확률')
        plt.legend()
        plt.show()



simulate_top_features_independent(final_model, x_row_orig, top_features, feature_ranges, normal_ranges, scaler=scaler)

# =====================================
# 13-4. 전역 탐색 기반 최적화 (Differential Evolution)
# =====================================
# 구간이 아닌 특정 값 추찬 방식
from scipy.optimize import differential_evolution

def objective_de(vals, model, x_row, features, scaler):
    x_mod = x_row.copy()
    for f, v in zip(features, vals):
        x_mod[f] = v
    x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
    prob = model.predict_proba(x_input)[0, 1]
    return prob  # minimize (불량 확률)

def optimize_features_de(model, x_row, features, feature_ranges, scaler=None):
    bounds = [feature_ranges[f] for f in features]

    result = differential_evolution(
        func=objective_de,
        bounds=bounds,
        args=(model, x_row, features, scaler),
        maxiter=200,       # 세대 수 (늘리면 더 정확)
        popsize=20,        # population 크기 (늘리면 더 전역적 탐색)
        tol=1e-6,
        polish=True,       # 지역 탐색으로 마무리
        seed=42
    )
    return result

res_de = optimize_features_de(final_model, x_row_orig, top_features, feature_ranges, scaler=scaler)
print("\n=== 전역 탐색 (Differential Evolution) 결과 ===")
print("최적화 변수 값:", res_de.x)
print("최소 불량 확률:", res_de.fun)

# 시각화
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_pdp_contour_pretty(model, x_row, features, feature_ranges, scaler=None, fixed_values=None, resolution=100, threshold=0.1):
    """
    PDP-style 등고선 + 색상 구분 + threshold 기준 + 최적해 표시
    """
    f1, f2 = features
    f1_min, f1_max = feature_ranges[f1]
    f2_min, f2_max = feature_ranges[f2]

    f1_vals = np.linspace(f1_min, f1_max, resolution)
    f2_vals = np.linspace(f2_min, f2_max, resolution)
    Z = np.zeros((resolution, resolution))

    for i, v1 in enumerate(f1_vals):
        for j, v2 in enumerate(f2_vals):
            x_mod = x_row.copy()
            x_mod[f1] = v1
            x_mod[f2] = v2
            if fixed_values:
                for f, val in fixed_values.items():
                    x_mod[f] = val
            x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
            Z[j, i] = model.predict_proba(x_input)[0, 1]

    # 색상: 낮은 확률 파랑 → 높은 확률 빨강
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(0, 1)

    plt.figure(figsize=(8,6))
    # 등고선 채우기
    contourf = plt.contourf(f1_vals, f2_vals, Z, levels=50, cmap=cmap, alpha=0.9)
    cbar = plt.colorbar(contourf)
    cbar.set_label("불량 확률", fontsize=12)
    
    # Threshold 경계 강조
    cs = plt.contour(f1_vals, f2_vals, Z, levels=[threshold], colors="black", linewidths=2, linestyles="--")
    plt.clabel(cs, fmt={threshold: f"Threshold = {threshold:.2f}"}, inline=True, fontsize=10)

    # 정상 영역 반투명 표시
    plt.contourf(f1_vals, f2_vals, Z, levels=[0, threshold], colors=["#ADD8E6"], alpha=0.3)

    # 최적값 표시
    plt.scatter(res_de.x[0], res_de.x[1], color="green", marker="X", s=150, label="전역 최적해")

    plt.xlabel(f1, fontsize=12)
    plt.ylabel(f2, fontsize=12)
    plt.title(f"PDP-style 등고선: {f1} vs {f2}", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


plot_pdp_contour_pretty(
    model=final_model,
    x_row=x_row_orig,
    features=top_features[:2],
    feature_ranges=feature_ranges,
    scaler=scaler,
    fixed_values={top_features[2]: res_de.x[2]},
    threshold=0.1
)




# 구간 추천 

# 1. 랜덤 샘플링
n_samples = 1000
valid_points = []

for _ in range(n_samples):
    vals = [np.random.uniform(*feature_ranges[f]) for f in top_features]
    x_mod = x_row_orig.copy()
    for f, v in zip(top_features, vals):
        x_mod[f] = v
    x_input = scaler.transform([x_mod.values])
    prob = final_model.predict_proba(x_input)[0,1]
    if prob < best_thresh:
        valid_points.append(vals)

valid_points = np.array(valid_points)

# 2. 각 feature별 min/max 구간
estimated_ranges = {f: (valid_points[:,i].min(), valid_points[:,i].max()) for i,f in enumerate(top_features)}
print("전역 탐색 기반 추정 정상 구간:", estimated_ranges)


# 바이너리 
def find_normal_range_binary_fixed(model, x_row, feature, f_min, f_max, scaler=None, threshold=0.1, tol=0.01, max_iter=20, n_check=5):
    """
    Interval Halving 방식으로 정상 구간 탐색
    n_check: 각 구간에서 여러 값 샘플링하여 정상 여부 확인
    """
    low, high = f_min, f_max
    iter_count = 0

    def is_normal(val):
        x_mod = x_row.copy()
        x_mod[feature] = val
        x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
        prob = model.predict_proba(x_input)[0,1]
        return prob < threshold

    while (high - low) > tol and iter_count < max_iter:
        # 구간 내 n_check 값 확인
        samples = np.linspace(low, high, n_check)
        normal_samples = [v for v in samples if is_normal(v)]
        if normal_samples:
            low, high = min(normal_samples), max(normal_samples)
        else:
            return None  # 현재 구간 내 정상 샘플 없음
        iter_count += 1

    return (low, high)


# =====================================
# 모든 feature에 대해 바이너리 탐색 실행
# =====================================
import numpy as np

def binary_search_normal_range(model, x_row, features, feature_ranges, scaler=None, threshold=0.1, max_iter=10):
    """
    상위 feature들을 대상으로 다차원 바이너리 탐색을 통해 정상 구간 추정
    """
    # 초기 min/max 범위
    current_ranges = {f: list(feature_ranges[f]) for f in features}

    for _ in range(max_iter):
        mid_vals = [(r[0]+r[1])/2 for r in current_ranges.values()]
        x_mod = x_row.copy()
        for f, mid in zip(features, mid_vals):
            x_mod[f] = mid

        x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
        prob = model.predict_proba(x_input)[0, 1]

        for i, f in enumerate(features):
            low, high = current_ranges[f]
            if prob < threshold:
                # 현재 mid 값 포함 가능, 범위를 절반으로 줄임
                current_ranges[f] = [low, mid] if (mid - low) > (high - mid) else [mid, high]
            else:
                # mid 값이 정상 기준 실패, 범위를 반대쪽으로
                current_ranges[f] = [mid, high] if (mid - low) > (high - mid) else [low, mid]

    return {f: tuple(r) for f, r in current_ranges.items()}

# 예시 사용
top5_features = [feature_names[i] for i in np.argsort(-sample_shap_abs)[:5]]
feature_ranges_top5 = {f: (X[f].min(), X[f].max()) for f in top5_features}

estimated_ranges_binary = binary_search_normal_range(
    model=final_model,
    x_row=x_row_orig,
    features=top5_features,
    feature_ranges=feature_ranges_top5,
    scaler=scaler,
    threshold=best_thresh,
    max_iter=15
)

print("다차원 바이너리 탐색 기반 정상 구간:")
for f, r in estimated_ranges_binary.items():
    print(f"{f}: {r}")


# 시각화
import matplotlib.pyplot as plt

def binary_search_normal_range_visual_status(model, x_row, features, feature_ranges, scaler=None, threshold=0.1, max_iter=10):
    """
    각 feature별로 iteration별 정상/불량 구간 시각화
    """
    current_ranges = {f: list(feature_ranges[f]) for f in features}
    history = {f: [] for f in features}
    
    for iter_idx in range(max_iter):
        mid_vals = [(r[0]+r[1])/2 for r in current_ranges.values()]
        x_mod = x_row.copy()
        for f, mid in zip(features, mid_vals):
            x_mod[f] = mid
        
        x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
        prob = model.predict_proba(x_input)[0, 1]
        status = 0 if prob < threshold else 1  # 0: 정상, 1: 불량
        
        for i, f in enumerate(features):
            low, high = current_ranges[f]
            history[f].append((low, high, status))
            if prob < threshold:
                current_ranges[f] = [low, mid] if (mid - low) > (high - mid) else [mid, high]
            else:
                current_ranges[f] = [mid, high] if (mid - low) > (high - mid) else [low, mid]
    
    # 시각화
    for f in features:
        lows = [h[0] for h in history[f]]
        highs = [h[1] for h in history[f]]
        statuses = [h[2] for h in history[f]]
        
        plt.figure(figsize=(6,4))
        for i in range(max_iter):
            color = 'skyblue' if statuses[i] == 0 else 'lightcoral'
            plt.fill_between([i, i+1], lows[i], highs[i], color=color, alpha=0.5)
            plt.plot([i, i+1], [lows[i], lows[i]], '--', color='blue')
            plt.plot([i, i+1], [highs[i], highs[i]], '--', color='red')
        
        plt.xlabel("Iteration")
        plt.ylabel(f"{f} 값")
        plt.title(f"{f} 정상/불량 구간 추적")
        plt.show()
    
    final_ranges = {f: tuple(r) for f, r in current_ranges.items()}
    return final_ranges

# 실행
estimated_ranges_binary_individual = binary_search_normal_range_visual_status(
    model=final_model,
    x_row=x_row_orig,
    features=top5_features,
    feature_ranges=feature_ranges_top5,
    scaler=scaler,
    threshold=best_thresh,
    max_iter=5
)

print("최종 추정 정상 구간 (독립 시각화):")
for f, r in estimated_ranges_binary_individual.items():
    print(f"{f}: {r}")








import numpy as np
import matplotlib.pyplot as plt

def plot_multifeature_normal_region(model, x_row, top_features, feature_ranges, scaler=None, threshold=0.1, resolution=50):
    """
    상위 feature 2개를 X, Y축으로 잡고, 나머지는 중간값(mid)으로 고정한 후 정상/불량 영역 시각화
    """
    # X축, Y축 feature 선택
    f1, f2 = top_features[:2]
    other_features = top_features[2:]

    f1_vals = np.linspace(*feature_ranges[f1], resolution)
    f2_vals = np.linspace(*feature_ranges[f2], resolution)
    
    Z = np.zeros((resolution, resolution))
    
    for i, v1 in enumerate(f1_vals):
        for j, v2 in enumerate(f2_vals):
            x_mod = x_row.copy()
            x_mod[f1] = v1
            x_mod[f2] = v2
            # 나머지 feature는 mid 값
            for f in other_features:
                mid = (feature_ranges[f][0] + feature_ranges[f][1]) / 2
                x_mod[f] = mid

            x_input = scaler.transform([x_mod.values]) if scaler else [x_mod.values]
            prob = model.predict_proba(x_input)[0,1]
            Z[j,i] = prob

    plt.figure(figsize=(7,6))
    # 정상 영역: prob < threshold
    plt.contourf(f1_vals, f2_vals, Z, levels=[0, threshold, 1], colors=['skyblue','salmon'], alpha=0.6)
    plt.colorbar(label="불량 확률")
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.title(f"{f1} vs {f2} 정상(파랑)/불량(빨강) 영역")
    plt.show()



plot_multifeature_normal_region(
    model=final_model,
    x_row=x_row_orig,
    top_features=top5_features,
    feature_ranges=feature_ranges_top5,
    scaler=scaler,
    threshold=best_thresh,
    resolution=50
)
