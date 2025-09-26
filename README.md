# Casting-Process-Data-Modeling-Project

![Python](https://img.shields.io/badge/python-3.13%2B-blue)

## 개요
주조(다이캐스팅) 공정에서 수집된 센서 데이터를 바탕으로 제품의 합격/불합격(품질) 여부를 예측하는 이진 분류 모델을 개발합니다. 학습된 모델은 모니터링 대시보드에 연결해 실무에서 실시간 예측 및 주요 변수의 영향도를 확인할 수 있도록 합니다.

## 주요 목표
- 데이터를 이용한 불량(불합격) 예측 모델 개발
- 모델 성능 지표: ROC AUC, PR AUC, F1-score, Precision, Recall, Confusion Matrix
- 실무 적용 가능성(경고 알림, 실시간 추론) 검증

## 폴더 구조
```
├── README.md
├── environment.yml
├── app
│   └── modules
├── data
│   ├── raw
│   ├── processed
│   └── interim
├── src
│   ├── eda
│   ├── models
│   └── preprocessing
└── tests
```

## 분석 과정
1. **EDA** :
   - 목적: 데이터 분포·결측·편향·시간/금형(mold)별 패턴을 파악해 이후 전처리·모델링 의사결정 근거 확보.
   - 체크리스트:
     - 타깃(`passorfail`) 클래스 비율(전체, 라인/금형별) 확인 — 불균형 정도 파악.
     - 결측치 패턴(행·열 단위) 시각화(heatmap / missingno) — 드물게 관측되는 행(예: `emergency_stop`)은 개별 검토.
     - 주요 수치형 변수의 분포(히스토그램, 박스플롯) 및 이상치 탐지.
     - 범주형 변수 빈도(예: `mold_name`, `line`)와 타깃과의 관계(교차표).
     - 상관관계(피어슨/스피어만) 및 변수 중요 후보 탐색(간단한 단변량 분석, mutual_info).
2. **전처리** :
   - 목적: 모델이 잘 학습하도록 결측·이상치 처리.
   - 주요 단계:
     - 결측치 처리: 
       - 센서·연속 변수는 도메인성 고려(이전/이후 값으로 보간)
     - 이상치 처리: 

3. **모델링** :
   - 목적: 불량(positive) 검출 성능 극대화(특히 FN 감소).
   - 모델 후보:
     - Baseline: `LogisticRegression`
     - 트리/부스팅: `RandomForest`, `LightGBM`, `XGBoost`, `CatBoost`
   - 학습 전략:
     - 하이퍼파라미터 검색: `Optuna`.

4. **진단** :
   - 목적: 모델의 약점·오류 유형 파악 및 개선 포인트 도출.
   - 핵심 진단 항목:
     - Confusion Matrix
     - Precision-Recall Curve + PR AUC
     - ROC Curve, AUC


## 평가 지표
- **ROC AUC**: 클래스 분포가 불균형할 때 전반적인 분별력 확인
- **PR AUC**: 불균형 데이터에서 Positive 예측 성능을 더 민감하게 반영
- **Precision / Recall / F1**: 불량 누락 최소화에 따라 중요도 조정
- **Confusion matrix**: TP/FN/FP/TN 직접 확인
```
Confusion matrix:
[[TN, FP]
 [FN, TP]]
```
> **FN(실제 불량이지만 모델은 합격으로 예측)** 를 가능한 작게 유지하는 것이 중요 포인트로 결정 


---
