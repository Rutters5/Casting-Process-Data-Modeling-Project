import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, make_scorer
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import optuna
from optuna.samplers import TPESampler
from matplotlib import rc
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False
rc('font', family='Malgun Gothic')

### ==============================
# 0. 데이터 로드
### ==============================
BASE_DIR = Path(__file__).resolve().parents[2]
TRAIN_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"
TEST_FILE = BASE_DIR / "data" / "processed" / "test_v1.csv"

# 데이터 로드 (한 번만!)
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

# 불필요한 컬럼 제거
columns_to_drop = []
if "Unnamed: 0" in train_df.columns:
    columns_to_drop.append("Unnamed: 0")
if "line" in train_df.columns:
    columns_to_drop.extend(["line", "name", "mold_name"])

train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns])
test_df = test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns])

# passorfail 타입 확인 및 변환
if "passorfail" in train_df.columns:
    if train_df["passorfail"].dtype == "object":
        train_df["passorfail"] = pd.to_numeric(train_df["passorfail"], errors="coerce")
    
    if train_df["passorfail"].isna().sum() > 0:
        train_df = train_df.dropna(subset=['passorfail'])
    
    train_df["passorfail"] = train_df["passorfail"].astype(int)

if "passorfail" in test_df.columns:
    if test_df["passorfail"].dtype == "object":
        test_df["passorfail"] = pd.to_numeric(test_df["passorfail"], errors="coerce")
    
    if test_df["passorfail"].isna().sum() > 0:
        test_df = test_df.dropna(subset=['passorfail'])
    
    test_df["passorfail"] = test_df["passorfail"].astype(int)

# =====================================
# 1. 전처리 함수 정의
# =====================================
def preprocess_data(df, is_train=True):
    """데이터 전처리 함수"""
    df = df.copy()
    
    if "passorfail" in df.columns:
        X = df.drop(columns=['passorfail'])
        y = df['passorfail'].astype(int)
    else:
        X = df.copy()
        y = None
    
    date_time_cols = []
    
    if 'date' in X.columns:
        if X['date'].dtype == 'object':
            X['date'] = pd.to_datetime(X['date'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(X['date']):
            X['year'] = X['date'].dt.year
            X['month'] = X['date'].dt.month
            X['day'] = X['date'].dt.day
            X['dayofweek'] = X['date'].dt.dayofweek
            date_time_cols.append('date')
    
    if 'time' in X.columns:
        if X['time'].dtype == 'object':
            X['time'] = pd.to_datetime(X['time'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(X['time']):
            X['time_hour'] = X['time'].dt.hour
            X['time_minute'] = X['time'].dt.minute
            date_time_cols.append('time')
    
    if 'hour' in X.columns and 'weekday' in X.columns:
        pass
    
    if date_time_cols:
        X = X.drop(columns=date_time_cols, errors='ignore')
    
    string_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    categorical_cols = ['working', 'tryshot_signal', 'emergency_stop']
    for col in categorical_cols:
        if col in string_cols:
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
            string_cols.remove(col)
    
    if 'mold_code' in X.columns:
        X['mold_code'] = X['mold_code'].astype('category').cat.codes
    
    for col in string_cols:
        if col in X.columns:
            X[col] = X[col].astype('category').cat.codes
    
    if 'molten_temp_filled' in X.columns and 'molten_temp' not in X.columns:
        X = X.rename(columns={'molten_temp_filled': 'molten_temp'})
    
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)
    
    X = X.fillna(X.median())
    
    return X, y

# =====================================
# 2. 데이터 분할
# =====================================
X_train_full, y_train_full = preprocess_data(train_df, is_train=True)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, 
    test_size=0.2,
    random_state=42, 
    stratify=y_train_full
)

X_test, y_test = preprocess_data(test_df, is_train=True)

common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_val = X_val[common_cols]
X_test = X_test[common_cols]

# =====================================
# 3. Optuna를 사용한 하이퍼파라미터 최적화
# =====================================
def objective(trial):
    # ① 하이퍼파라미터는 trial에서 제안
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None]),
        'random_state': 42,
        'n_jobs': -1,
    }
    use_smote = trial.suggest_categorical('use_smote', [True, False])

    # ② 학습 데이터(옵션: SMOTE로 증강)
    X_train_opt = X_train.copy()
    y_train_opt = y_train.copy()

    if use_smote:
        pos = int((y_train == 1).sum())
        k = max(1, min(5, pos - 1))      # 안전한 k_neighbors
        smote = SMOTE(sampling_strategy=0.25, random_state=42, k_neighbors=k)
        X_train_opt, y_train_opt = smote.fit_resample(X_train_opt, y_train_opt)

    # ③ 모델 학습 → 검증 예측
    model = RandomForestClassifier(**params)
    model.fit(X_train_opt, y_train_opt)  # 반드시 fit 먼저!
    y_pred = model.predict(X_val)

    # ④ 점수 계산 (불합격=1 기준)
    rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
    f1  = f1_score(y_val, y_pred,    pos_label=1, zero_division=0)
    return 0.7 * rec + 0.3 * f1

# Optuna Study 생성 및 최적화 실행
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# 최적 파라미터와 점수 저장
best_params_info = {
    'params': study.best_params,
    'score': study.best_value
}

# =====================================
# 4. 최적 파라미터로 모델 재학습
# =====================================
best_params = study.best_params
use_smote = best_params.pop('use_smote', False)

# 최적 파라미터로 모델 생성
best_rf_model = RandomForestClassifier(
    **{k: v for k, v in best_params.items() if k != 'use_smote'},
    random_state=42,
    n_jobs=-1
)

# SMOTE 적용 여부에 따라 학습 데이터 준비
X_train_final = X_train.copy()
y_train_final = y_train.copy()

if use_smote:
    smote = SMOTE(random_state=42, k_neighbors=min(5, (y_train==1).sum()-1))
    X_train_final, y_train_final = smote.fit_resample(X_train_final, y_train_final)
    smote_distribution = pd.Series(y_train_final).value_counts()

# 모델 학습
best_rf_model.fit(X_train_final, y_train_final)

# =====================================
# 5. 성능 평가
# =====================================
def evaluate_model(model, X, y, dataset_name):
    """모델 성능 평가 함수"""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    cm = confusion_matrix(y, y_pred)
    
    return {
        'Dataset': dataset_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm
    }

# Validation과 Test 성능 평가
val_results = evaluate_model(best_rf_model, X_val, y_val, 'Validation')
test_results = evaluate_model(best_rf_model, X_test, y_test, 'Test')

# 결과 데이터프레임 생성
performance_df = pd.DataFrame([val_results, test_results])
performance_df = performance_df[['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]

# 혼동 행렬 데이터프레임
cm_df = pd.DataFrame(
    test_results['Confusion Matrix'], 
    index=['Actual_Pass(0)', 'Actual_Fail(1)'], 
    columns=['Pred_Pass(0)', 'Pred_Fail(1)']
)

# =====================================
# 6. Recall 향상을 위한 추가 전략
# =====================================
# 확률 예측을 사용한 임계값 조정
y_proba = best_rf_model.predict_proba(X_test)[:, 1]

# 다양한 임계값에 대한 성능 평가
thresholds = np.arange(0.2, 0.8, 0.05)
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_proba >= threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold, zero_division=0)
    recall = recall_score(y_test, y_pred_threshold, zero_division=0)
    f1 = f1_score(y_test, y_pred_threshold, zero_division=0)
    
    threshold_results.append({
        'Threshold': threshold,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

threshold_df = pd.DataFrame(threshold_results)

# Recall이 가장 높은 임계값 찾기 (단, F1 score도 고려)
best_threshold_idx = threshold_df['Recall'].idxmax()
best_threshold = threshold_df.loc[best_threshold_idx, 'Threshold']
best_threshold_performance = threshold_df.loc[best_threshold_idx]

# 임계값별 성능 시각화
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Recall vs Precision Trade-off
axes[0].plot(threshold_df['Threshold'], threshold_df['Recall'], label='Recall', marker='o')
axes[0].plot(threshold_df['Threshold'], threshold_df['Precision'], label='Precision', marker='s')
axes[0].plot(threshold_df['Threshold'], threshold_df['F1-Score'], label='F1-Score', marker='^')
axes[0].axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Score')
axes[0].set_title('Performance Metrics vs Threshold')
axes[0].legend()
axes[0].grid(True)

# Accuracy vs Recall Trade-off
axes[1].plot(threshold_df['Recall'], threshold_df['Accuracy'], marker='o')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Recall Trade-off')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# =====================================
# 7. 변수 중요도 분석
# =====================================
feature_importance_df = pd.DataFrame({
    '변수명': X_train.columns,
    '중요도': best_rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

top_10_features = feature_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_10_features['변수명'], top_10_features['중요도'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances (Optimized Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =====================================
# 8. 앙상블 전략 (옵션)
# =====================================
# 여러 random_state로 학습한 모델들의 앙상블
ensemble_models = []
random_states = [42, 123, 456, 789, 1011]

for rs in random_states:
    model = RandomForestClassifier(
        **{k: v for k, v in best_params.items() if k not in ['use_smote', 'random_state']},
        random_state=rs,
        n_jobs=-1
    )
    
    if use_smote:
        smote = SMOTE(random_state=rs, k_neighbors=min(5, (y_train==1).sum()-1))
        X_train_rs, y_train_rs = smote.fit_resample(X_train, y_train)
        model.fit(X_train_rs, y_train_rs)
    else:
        model.fit(X_train, y_train)
    
    ensemble_models.append(model)

# 앙상블 예측 (다수결 투표)
ensemble_predictions = np.array([model.predict(X_test) for model in ensemble_models])
y_pred_ensemble = np.round(np.mean(ensemble_predictions, axis=0)).astype(int)

# 앙상블 성능 평가
ensemble_results = {
    'Accuracy': accuracy_score(y_test, y_pred_ensemble),
    'Precision': precision_score(y_test, y_pred_ensemble),
    'Recall': recall_score(y_test, y_pred_ensemble),
    'F1-Score': f1_score(y_test, y_pred_ensemble)
}

# =====================================
# 9. 최종 모델 저장
# =====================================
import joblib

# 최적 모델 저장
model_save_path = BASE_DIR / "models" / "best_rf_model.pkl"
model_save_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(best_rf_model, model_save_path)

# =====================================
# 10. 제출 파일 생성
# =====================================
if True:  # 제출 파일 생성
    ORIGINAL_TEST_FILE = BASE_DIR / "data" / "raw" / "test.csv"
    original_test = pd.read_csv(ORIGINAL_TEST_FILE)
    
    X_final_test, _ = preprocess_data(original_test, is_train=False)
    X_final_test = X_final_test[common_cols]
    
    # 최적 임계값을 사용한 예측
    y_final_proba = best_rf_model.predict_proba(X_final_test)[:, 1]
    y_final_pred = (y_final_proba >= best_threshold).astype(int)
    
    submission = pd.DataFrame({
        'id': original_test['id'],
        'passorfail': y_final_pred
    })
    
    submission.to_csv(BASE_DIR / "submit_rf_optimized.csv", index=False)
    
# =====================================
# 11. 결과 요약 (변수로 저장)
# =====================================
# 모든 결과를 딕셔너리로 정리
final_results = {
    'best_params': best_params_info,
    'performance': performance_df,
    'confusion_matrix': cm_df,
    'threshold_analysis': threshold_df,
    'best_threshold': best_threshold,
    'best_threshold_performance': best_threshold_performance,
    'ensemble_results': ensemble_results,
    'top_features': top_10_features
}