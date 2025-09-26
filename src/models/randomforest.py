import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

### ==============================
# 0. 데이터 로드
### =====================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "train_v1.csv"

train_df = pd.read_csv(DATA_FILE)


# =====================================
# 1. 데이터 준비
# =====================================
# 타겟 변수와 특성 분리
X = train_df.drop(columns=['passorfail'])  # 타겟 변수 제외
y = train_df['passorfail']

# 1단계: 문자열 컬럼 찾기 및 Label Encoding
string_cols = X.select_dtypes(include=['object']).columns.tolist()

if len(string_cols) > 0:
    for col in string_cols:
        X[col] = X[col].astype('category').cat.codes

# 2단계: 혹시 남은 문자열 컬럼 제거
remaining_objects = X.select_dtypes(include=['object']).columns.tolist()
if len(remaining_objects) > 0:
    X = X.drop(columns=remaining_objects)

# 날짜/시간 변수 처리 (있는 경우)
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
        X['hour'] = X['time'].dt.hour
        X['minute'] = X['time'].dt.minute
        date_time_cols.append('time')

# 날짜/시간 원본 컬럼 제거
if len(date_time_cols) > 0:
    X = X.drop(columns=date_time_cols)

# 최종 확인: 모든 컬럼이 숫자형인지 체크
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if len(non_numeric_cols) > 0:
    X = X.drop(columns=non_numeric_cols)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# =====================================
# 2. 랜덤포레스트 모델 학습
# =====================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'  # 불균형 데이터 대응
)

rf_model.fit(X_train, y_train)

# =====================================
# 3. 기본 성능 평가
# =====================================
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

performance_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score': [accuracy, precision, recall, f1]
})

# =====================================
# 4. 변수 중요도 분석
# =====================================
# 기본 특성 중요도
feature_importance_df = pd.DataFrame({
    '변수명': X.columns,
    '중요도': rf_model.feature_importances_
}).sort_values('중요도', ascending=False)

# 상위 10개 변수만 추출
top_10_features = feature_importance_df.head(10)

# Permutation Importance (더 신뢰할 수 있는 중요도)
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    '변수명': X.columns,
    '순열중요도': perm_importance.importances_mean
}).sort_values('순열중요도', ascending=False)

top_10_perm = perm_importance_df.head(10)

import matplotlib.pyplot as plt

# 1. Feature Importance 시각화
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['변수명'], top_10_features['중요도'])
plt.xlabel('Importance')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 2. Permutation Importance 시각화
plt.figure(figsize=(10, 6))
plt.barh(top_10_perm['변수명'], top_10_perm['순열중요도'])
plt.xlabel('Permutation Importance')
plt.title('Top 10 Permutation Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# =====================================
# 5. 교차검증
# =====================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores_accuracy = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
cv_scores_f1 = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
cv_scores_precision = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='precision_weighted', n_jobs=-1)
cv_scores_recall = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='recall_weighted', n_jobs=-1)

cv_results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score', 'Precision', 'Recall'],
    'Mean': [cv_scores_accuracy.mean(), cv_scores_f1.mean(), 
             cv_scores_precision.hmean(), cv_scores_recall.mean()],
    'Std': [cv_scores_accuracy.std(), cv_scores_f1.std(), 
            cv_scores_precision.std(), cv_scores_recall.std()]
})


# =====================================
# 6. SHAP 값 분석
# =====================================
# SHAP Explainer 생성 (TreeExplainer 사용 - 빠름)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# 이진 분류인 경우 클래스 1(불량)에 대한 SHAP 값만 사용
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values

# SHAP 평균 절대값으로 중요도 계산
shap_importance = np.abs(shap_values_class1).mean(axis=0)
shap_importance_df = pd.DataFrame({
    '변수명': X.columns,
    'SHAP중요도': shap_importance
}).sort_values('SHAP중요도', ascending=False)


# =====================================
# 7. 혼동 행렬
# =====================================
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm)