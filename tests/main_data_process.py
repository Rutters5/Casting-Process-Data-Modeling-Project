import pandas as pd
import numpy as np

train = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\train.csv")
test = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\test.csv")
submission = pd.read_csv("C:\\Users\\an100\\OneDrive\\바탕 화면\\test\\main_project1\\data\\submit.csv")

'''
# 컬럼명 매핑 딕셔너리
column_mapping = {
'line': '작업_라인',
'name': '제품명',
'mold_name': '금형명',
'time': '수집_시간',
'date': '수집_일자',--> 나중에
'count': '일자별_생산_번호', ---> 해석 불가
'working': '가동_여부', ---> 38개만 정지였는데, 결측치 1개 드랍하고 그대로 쓰기
'emergency_stop': '비상_정지_여부', ---> 값이 통일되어있어서 통으로 날린다?
'molten_temp': '용탕_온도', ---> 결측치가 2261개 있음
'facility_operation_cycleTime': '설비_작동_사이클_시간', ---> 결측치 없어서 그대로 두기
'production_cycletime': '제품_생산_사이클_시간', ---> 위의 컬럼이랑 그래프가 비슷함 뭔가 연관
'low_section_speed': '저속_구간_속도', ---> 결측치, 이상치 다 드랍?
'high_section_speed': '고속_구간_속도', ---> 
'molten_volume': '용탕량',
'cast_pressure': '주조_압력',
'biscuit_thickness': '비스켓_두께',
'upper_mold_temp1': '상금형_온도1',
'upper_mold_temp2': '상금형_온도2',
'upper_mold_temp3': '상금형_온도3',
'lower_mold_temp1': '하금형_온도1',
'lower_mold_temp2': '하금형_온도2',
'lower_mold_temp3': '하금형_온도3',
'sleeve_temperature': '슬리브_온도',
'physical_strength': '형체력',
'Coolant_temperature': '냉각수_온도',
'EMS_operation_time': '전자교반_가동_시간',
'registration_time': '등록_일시',
'passorfail': '양품불량_판정',
'tryshot_signal': '사탕_신호',
'mold_code': '금형_코드',
'heating_furnace': '가열로_구분' -> a/b가 있고 38000정도로 결측치 많아서 확인
'''

train.shape

#--------------------------------------------------------------------------------
#Coolant_temperature: 냉각수_온도
#금형이 너무 뜨거우면 제품 품질이 나빠지기 때문에, 차가운 물로 금형을 식혀줘야함.
train.loc[(train["molten_temp"]-train["Coolant_temperature"]) < 0,"passorfail"].value_counts()

#시각화
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 기본 통계 및 결측치 확인
train['Coolant_temperature'] #냉각수 온도
train['Coolant_temperature'].unique()
basic_stats = train['Coolant_temperature'].describe()

#결측치 확인
missing_count = train['Coolant_temperature'].isnull().sum() #1개의 결측치
missing_ratio = missing_count / len(train) * 100
train['Coolant_temperature'].fillna(train['Coolant_temperature'].mean(), inplace=True)
train['Coolant_temperature'].isnull().sum()
#결측치는 없애거나 평균으로 채우자!

#이상치 확인
# 왜도와 첨도 계산
skewness = stats.skew(train['Coolant_temperature'])
kurtosis = stats.kurtosis(train['Coolant_temperature'])
'''
결과 
왜도 : 86.31348636701962
첨도 : 7682.66067388381
-왜도가 극도로 오른쪽으로 치우쳐있다, 대부분의 값이 낮은 범위에 몰려있고, 일부 매우 높은 값들이 있다.
-첨도 극단적인 이상치가 존재
'''

# 데이터 박스플롯
plt.figure(figsize=(8, 6))
plt.boxplot(train['Coolant_temperature'])
plt.title('전체 데이터 박스플롯')
plt.ylabel('Coolant_temperature')
plt.show()

# Coolant_temperature 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(train['Coolant_temperature'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Coolant_temperature 히스토그램')
plt.xlabel('Coolant_temperature')
plt.ylabel('빈도')
plt.axvline(train['Coolant_temperature'].mean(), color='red', linestyle='--', label='평균')
plt.axvline(train['Coolant_temperature'].median(), color='green', linestyle='--', label='중앙값')
plt.legend()
plt.show()

#이상치제거
# 99분위수 기준 이상치 제거
percentile_99_threshold = train['Coolant_temperature'].quantile(0.99) #39도를 넘어가면 이상값이다.
train_cleaned = train[train['Coolant_temperature'] <= percentile_99_threshold].copy()

# 제거 전후 비교
original_count = len(train)
cleaned_count = len(train_cleaned)
removed_count = original_count - cleaned_count
removal_ratio = (removed_count / original_count) * 100

# 제거된 이상치들
removed_outliers = train[train['Coolant_temperature'] > percentile_99_threshold]

# 정제된 데이터의 새로운 통계량
new_skewness = stats.skew(train_cleaned['Coolant_temperature'])
new_kurtosis = stats.kurtosis(train_cleaned['Coolant_temperature'])

train.loc[train["Coolant_temperature"]>1400,:]

# 정제된 데이터 히스토그램
plt.figure(figsize=(10, 6))
plt.hist(train_cleaned['Coolant_temperature'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('99분위수 기준 이상치 제거 후 히스토그램')
plt.xlabel('Coolant_temperature')
plt.ylabel('빈도')
plt.axvline(train_cleaned['Coolant_temperature'].mean(), color='red', linestyle='--', label='평균')
plt.axvline(train_cleaned['Coolant_temperature'].median(), color='blue', linestyle='--', label='중앙값')
plt.legend()
plt.show()

# 정제된 데이터 박스플롯
plt.figure(figsize=(8, 6))
plt.boxplot(train_cleaned['Coolant_temperature'])
plt.title('이상치 제거 후 박스플롯')
plt.ylabel('Coolant_temperature')
plt.show()

# test 데이터도 동일하게 처리
test['Coolant_temperature'].fillna(test['Coolant_temperature'].mean(), inplace=True)
test_cleaned = test[test['Coolant_temperature'] <= percentile_99_threshold].copy()


'''
[결론]
실무적 의미:
냉각수 온도 39℃ 이하: 정상 운영 범위
냉각수 온도 39℃ 초과: 설비 이상 상황으로 간주하여 분석에서 제외
'''

# ----------------------------------------------
# EMS_operation_time : '전자교반_가동_시간'
# 전자기장을 이용해서 용융된 금속을 저어주는 시스템
# 물리적인 막대로 젓는 것이 아니라 전기적/자기적 힘으로 교반

# 1. 기본 정보 확인
train['EMS_operation_time'].dtype
train['EMS_operation_time'].unique()
basic_stats_ems = train['EMS_operation_time'].describe()
# 이 데이터는 연속형 데이터라기보다 범주형 데이터에 가까움

# 2. 결측치 확인
missing_count_ems = train['EMS_operation_time'].isnull().sum()
missing_ratio_ems = missing_count_ems / len(train) * 100
#결측치 없어

#3. 빈도 확인
train['EMS_operation_time'].value_counts().sort_index()

#4. 범주형으로 변경
# 데이터 타입을 범주형으로 변경
train['EMS_operation_time'] = train['EMS_operation_time'].astype('category')
test['EMS_operation_time'] = test['EMS_operation_time'].astype('category')

# 교반 시간별 품질 비율
quality_by_ems = train.groupby('EMS_operation_time')['passorfail'].agg(['count', 'mean'])

# 시각화
plt.figure(figsize=(10, 6))
quality_by_ems['mean'].plot(kind='bar')
plt.title('교반 시간별 불량률')
plt.xlabel('EMS_operation_time')
plt.ylabel('불량률 (passorfail)')
plt.show()

'''
결론
전처리 필요 없나..?
'''

# ---------------------------------------------------------
# registration_time': '등록_일시
# 1. 기본 정보 확인
train['registration_time'].dtype
train['registration_time'].head(10)
train['registration_time'].unique()[:10]  # 처음 10개만 확인

# 2. 결측치 확인
missing_count_reg = train['registration_time'].isnull().sum()
missing_ratio_reg = missing_count_reg / len(train) * 100

# 3. 날짜/시간 형식으로 변환
train['registration_time'] = pd.to_datetime(train['registration_time'])
test['registration_time'] = pd.to_datetime(test['registration_time'])

# 4. 날짜/시간 정보 추출
train['reg_year'] = train['registration_time'].dt.year
train['reg_month'] = train['registration_time'].dt.month
train['reg_day'] = train['registration_time'].dt.day
train['reg_hour'] = train['registration_time'].dt.hour
train['reg_minute'] = train['registration_time'].dt.minute
train['reg_weekday'] = train['registration_time'].dt.weekday  # 0=월요일, 6=일요일
train['reg_week'] = train['registration_time'].dt.isocalendar().week

# 5. 기간 정보 확인
date_range = f"{train['registration_time'].min()} ~ {train['registration_time'].max()}"
total_days = (train['registration_time'].max() - train['registration_time'].min()).days

# 6. 시간대별 데이터 분포 확인
hour_distribution = train['reg_hour'].value_counts().sort_index()
weekday_distribution = train['reg_weekday'].value_counts().sort_index()

# 7. 시간대별 시각화
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
hour_distribution.plot(kind='bar')
plt.title('시간대별 데이터 분포')
plt.xlabel('시간 (Hour)')
plt.ylabel('빈도')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
weekday_distribution.plot(kind='bar')
plt.title('요일별 데이터 분포')
plt.xlabel('요일 (0=월, 6=일)')
plt.ylabel('빈도')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

'''
시간대별 데이터 분포
-7시에 등록된 제품의 개수가 제일 적었다.
-야간시간에 등록되는 제품의 개수가 많음.(야간근무가 활발)
-야간근무가 낮근무보다 생산량이 많음.

요일별 데이터 분포
-일요일에 등록된 제품의 개수가 제일 적음
-수요일에 등록된 제품의 개수가 제일 많음

*야간 중심의 생산 패턴을 보이는 공장
*시간대별 요일이 품질에 어떤 영향을 미치는지
'''

# -----------------------------------------
#  'tryshot_signal': '사탕_신호'
#  정상 생산인지 시험 생산인지를 구분하는 플래그 변수

# 1. 기본 정보 확인
train['tryshot_signal'].dtype
tryshot_unique_values = train['tryshot_signal'].unique()
tryshot_value_counts = train['tryshot_signal'].value_counts()

# 2. 결측치 확인
missing_count_tryshot = train['tryshot_signal'].isnull().sum()
missing_ratio_tryshot = missing_count_tryshot / len(train) * 100

# 4. 사탕신호별 분포 시각화
plt.figure(figsize=(8, 6))
tryshot_value_counts.plot(kind='bar')
plt.title('사탕신호별 분포')
plt.xlabel('tryshot_signal')
plt.ylabel('개수')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# mold_code : 금형코드
# 1. 기본 정보 확인
# 1. 기본 정보 확인
train['mold_code'].dtype
train['mold_code'].unique()

# 2. 결측치 확인
missing_count_mold = train['mold_code'].isnull().sum()
missing_ratio_mold = missing_count_mold / len(train) * 100

# 4. 금형코드별 빈도 확인
mold_frequency = train['mold_code'].value_counts().sort_index()

# 5. 금형코드 분포 시각화
plt.figure(figsize=(12, 6))
mold_frequency.plot(kind='bar')
plt.title('금형코드별 생산 빈도')
plt.xlabel('mold_code')
plt.ylabel('생산 개수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

'''
-금형별 생산량 편차가 크다
-8917금형 : 가장 많이 사용되는건데 
-8600금형 : 가장 적게 사용되니깐 특수 목적으로 사용될 가능성이 있다.

금형별 생산량
8917> 8722 > 8412 > 8573 > 8600

불량률 높은 순서
8722 > 8600 > 8917 > 8573 > 8412


금형별로 별도 분석이 필요하겠다
'''

# 금형별 불량률
mold_defect_rate = train.groupby('mold_code')['passorfail'].mean().sort_values(ascending=False)

# 시각화
plt.figure(figsize=(10, 6))
mold_defect_rate.plot(kind='bar')
plt.title('금형별 불량률')
plt.xlabel('mold_code')
plt.ylabel('불량률')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


