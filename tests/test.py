import numpy as np
import pandas as pd

train_df = pd.read_csv("./data/raw/train.csv")
test_df = pd.read_csv("./data/raw/test.csv")
train_df.info()

train_df.isna().sum()

train_df[train_df['passorfail'] == 1]
train_df.columns
# ['id', 'line', 'name', 'mold_name', 'time', 'date', 'count', 'working',
#       'emergency_stop', 'molten_temp', 'facility_operation_cycleTime',
#       'production_cycletime', 'low_section_speed', 'high_section_speed',
#       'molten_volume', 'cast_pressure', 'biscuit_thickness',
#       'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
#       'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
#       'sleeve_temperature', 'physical_strength', 'Coolant_temperature',
#       'EMS_operation_time', 'registration_time', 'passorfail',
#       'tryshot_signal', 'mold_code', 'heating_furnace']
train_df['line'].unique() # '전자교반 3라인 2호기' 하나만 존재 -> 생략
train_df['name'].unique() # 'TM Carrier RH' 하나만 존재 -> 생략
train_df['mold_name'].unique() # 'TM Carrier RH-Semi-Solid DIE-06' 하나만 존재 -> 생략
train_df['time'].unique() # 2019년 1월 2일부터 (1월 13일, 2월 4/5/6일 제외) 3월 12일까지 존재
train_df['date'].unique() # 시:분:초 단위
train_df['count'].unique() # 일자별 생산번호 -> time과 동일
train_df['working'].unique() # 가동, 정지, nan
train_df['emergency_stop'].unique() # ON, nan -> 비상정지 여부
train_df['molten_temp'].unique() # 용탕 온도
train_df['cast_pressure'].unique() # 주조 압력
train_df['biscuit_thickness'].unique() # 비스켓 두께
train_df['upper_mold_temp1'].unique() # 상금형 온도 1
train_df['upper_mold_temp2'].unique() # 상금형 온도 2
train_df['upper_mold_temp3'].unique() # 상금형 온도 3
train_df['lower_mold_temp1'].unique() # 하금형 온도 1
train_df['lower_mold_temp2'].unique() # 하금형 온도 2
train_df['lower_mold_temp3'].unique() # 하금형 온도 3
train_df['sleeve_temperature'].unique() # 슬리브 온도
train_df['physical_strength'].unique() # 형체력
train_df['Coolant_temperature'].unique() # 냉각수 온도 -> 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1449, nan
train_df['EMS_operation_time'].unique() # 전자교반 가동 시간 # [23, 25, 0, 3, 6]
train_df['registration_time'].unique() # 등록 일시 '연-월-일 시:분:초' 형식
train_df['passorfail'].unique() # 양품/불량판정, 0 또는 1 (1이면 불량)
train_df['tryshot_signal'].unique() # 시운전 여부, 'D' 또는 nan -> D일 경우 시험 사출 데이터
train_df['mold_code'].unique() # 금형 코드, 8722/8412/8573/8917/8600
train_df['heating_furnace'].unique() # 가열로 구분, A/B/nan
train_df[['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3']]
train_df['upper_mold_temp1'].value_counts()
train_df['mold_code'].unique()
train_df.loc[train_df['mold_code'] == 8722]['molten_volume'].value_counts()
train_df.loc[train_df['mold_code'] == 8412]['molten_volume'].value_counts()
train_df.loc[train_df['mold_code'] == 8573]['molten_volume'].value_counts()
train_df.loc[train_df['mold_code'] == 8917]['molten_volume'].value_counts()
train_df.loc[train_df['mold_code'] == 8600]['molten_volume'].value_counts()
train_df.loc[train_df['molten_volume'] == 84.0]['passorfail'].value_counts()
test_df['mold_code'].unique()


train_df.loc[train_df['molten_volume'] > 2500]['tryshot_signal'].value_counts()

train_df['molten_temp'].isna().sum()


train_df['count']
train_df.groupby('date')['count'].value_counts()
train_df.groupby('time')['count'].value_counts()
train_df.groupby('time')['count'].max()

# 일자(time), 제품(mold_code)별 count 중복 횟수 확인
dup_stats = (
    train_df
    .groupby(['time','mold_code'])['count']
    .value_counts()   # count별 등장 횟수
    .reset_index(name='dup_freq')  # 컬럼 이름 정리
    .sort_values(['dup_freq'], ascending=False)  # 중복 많은 순 정렬
)

dup_stats
print(dup_stats.head(20))  # 상위 20개 확인

train_df[(train_df['time'] == '2019-01-07') & (train_df['mold_code'] == 8573) & (train_df['count'] == 312)]
train_df.iloc[51577, :]
train_df.iloc[51578, :]
train_df.iloc[51576, :]



# ============================================================================================
# working 열 파악 시도
# ============================================================================================
# 시간 순으로 정렬
df_sorted = train_df.sort_values(['time','count','id'])

# 이전 working 상태와 비교
df_sorted['prev_working'] = df_sorted['working'].shift()

# "정지 -> 가동" 전환 이벤트 플래그
df_sorted['restart_event'] = (
    (df_sorted['prev_working'] == '정지') & (df_sorted['working'] == '가동')
)

# mold_code별 재가동 횟수 집계
restart_stats = (
    df_sorted.groupby('mold_code')['restart_event']
    .sum()
    .reset_index(name='restart_count')
    .sort_values('restart_count', ascending=False)
)

print(restart_stats)

import pandas as pd

# datetime으로 변환 (등록 일시 기준)
df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['registration_time'])

# 시간순 정렬
df = df.sort_values(['mold_code', 'timestamp', 'id'])

# 이전 상태와 비교
df['prev_working'] = df.groupby('mold_code')['working'].shift()

# "정지 -> 가동" 구간만 추출
restart_events = df[
    (df['prev_working'] == '정지') & (df['working'] == '가동')
].copy()

# 직전 "정지" 시점 구하기
restart_events['prev_timestamp'] = df.groupby('mold_code')['timestamp'].shift()[restart_events.index]

# 다운타임 계산 (초 단위)
restart_events['downtime_sec'] = (restart_events['timestamp'] - restart_events['prev_timestamp']).dt.total_seconds()

# mold_code별 평균/최대/최소 다운타임 요약
downtime_stats = restart_events.groupby('mold_code')['downtime_sec'].agg(['count','mean','min','max']).reset_index()

print(downtime_stats.head())
restart_events[['mold_code','prev_timestamp','timestamp','downtime_sec']]


import pandas as pd

df = train_df.copy()
df['timestamp'] = pd.to_datetime(df['registration_time'])
df = df.sort_values(['mold_code','timestamp','id'])

# 이전 상태와 품질 기록
df['prev_working'] = df.groupby('mold_code')['working'].shift()
df['prev_passorfail'] = df.groupby('mold_code')['passorfail'].shift()

# "정지 -> 가동" 전환 행만 추출
restart_events = df[
    (df['prev_working'] == '정지') & (df['working'] == '가동')
].copy()

# 정지 직전 품질, 재가동 직후 품질
restart_events = restart_events[['mold_code','timestamp','prev_passorfail','passorfail']]

# 교차 집계표 (정지 전 vs 재가동 후)
compare_table = pd.crosstab(
    restart_events['prev_passorfail'],
    restart_events['passorfail'],
    rownames=['정지 전 판정'],
    colnames=['재가동 후 판정'],
    normalize='index'  # 비율로 보기 (횟수로 보려면 빼기)
)

print("=== 샘플 이벤트 비교 ===")
print(restart_events.head())

print("\n=== 정지 전 vs 재가동 후 판정 분포 (비율) ===")
print(compare_table)
# ===========================================================================================





# ===========================================================================================
# molten_temp 결측치 채우기 시도
# ===========================================================================================
df = train_df.copy()

# 🔹 원본 molten_temp를 새로운 열로 복사
df['molten_temp_filled'] = df['molten_temp']

# 🔹 금형별 시간 순 정렬 후 선형 보간
df['molten_temp_filled'] = (
    df.groupby('mold_code')['molten_temp_filled']
      .transform(lambda x: x.interpolate(method='linear'))
)

# 🔹 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
df['molten_temp_filled'] = (
    df.groupby('mold_code')['molten_temp_filled']
      .transform(lambda x: x.fillna(x.median()))
)
df[['molten_temp', 'molten_temp_filled']]
df['molten_temp'].isna().sum()
df['molten_temp_filled'].isna().sum()
# ===========================================================================================




# ===========================================================================================
# cast_pressure, biscuit_thickness, upper_mold_temp1~3 확인
# ===========================================================================================
import matplotlib.pyplot as plt
import seaborn as sns

# 분석 대상 칼럼
cols = ['cast_pressure', 'biscuit_thickness',
        'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3']

for col in cols:
    print(f"\n===== {col} =====")
    print("결측치 개수:", train_df[col].isna().sum())
    print("고유값 개수:", train_df[col].nunique())
    print(train_df[col].describe())  # 기초통계량 (count, mean, std, min, 25%, 50%, 75%, max)
    
    plt.figure(figsize=(12,4))
    
    # 1) 히스토그램
    plt.subplot(1,2,1)
    sns.histplot(train_df[col], bins=50, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}")
    
    # 2) 박스플롯 (이상치 확인)
    plt.subplot(1,2,2)
    sns.boxplot(x=train_df[col], color="lightcoral")
    plt.title(f"Boxplot of {col}")
    
    plt.tight_layout()
    plt.show()

# 
import seaborn as sns
# 
df = train_df.copy()
df = df[(df['upper_mold_temp1'] < 1000) & (df['upper_mold_temp2'] < 4000)]
sns.scatterplot(data = df, x= 'upper_mold_temp1', y='upper_mold_temp2', hue='mold_code')





import plotly.express as px
import pandas as pd

# 🔹 데이터 전처리 (온도 이상치 제거)
df = train_df.copy()
df = df[(df['upper_mold_temp1'] < 1000) & 
        (df['upper_mold_temp2'] < 4000) & 
        (df['upper_mold_temp3'] < 2000)]
df = df.dropna(subset=['upper_mold_temp1','upper_mold_temp2','upper_mold_temp3','passorfail'])

# 🔹 Plotly 3D 산점도
fig = px.scatter_3d(
    df,
    x='upper_mold_temp1',
    y='upper_mold_temp2',
    z='upper_mold_temp3',
    color='mold_code',   # 색상 구분
    symbol='mold_code',  # 모양 구분 (0,1)
    opacity=0.6,
    title="3D Scatter: Upper Mold Temperatures vs Pass/Fail"
)

fig.update_traces(marker=dict(size=3))  # 점 크기 조정
fig.show()





# ===========================================================================================
# cast_pressure와 passorfail 비교
# ===========================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 한글 폰트 설정 (Windows 기본: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 🔹 데이터 준비
df = train_df[['cast_pressure', 'passorfail']].copy()
df = df.dropna(subset=['cast_pressure', 'passorfail'])
df['passorfail'] = df['passorfail'].astype(int)

print("=== 전체/클래스별 요약통계 ===")
print(df.groupby('passorfail')['cast_pressure'].describe(), "\n")

# 1) 히스토그램 + KDE
fig, ax = plt.subplots(1, 2, figsize=(14,4))
sns.histplot(data=df, x='cast_pressure', hue='passorfail',
             bins=60, stat='density', common_norm=False,
             element='step', fill=False, ax=ax[0])
ax[0].set_title("Histogram (밀도) by passorfail\n0=양품, 1=불량")

sns.kdeplot(data=df[df['passorfail']==0]['cast_pressure'], label='0(양품)', ax=ax[1])
sns.kdeplot(data=df[df['passorfail']==1]['cast_pressure'], label='1(불량)', ax=ax[1])
ax[1].legend()
ax[1].set_title("KDE (밀도추정) by passorfail")

plt.tight_layout()
plt.show()

# 2) 박스플롯 / 바이올린 플롯
fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.boxplot(data=df, x='passorfail', y='cast_pressure', ax=ax[0])
ax[0].set_title("Boxplot by passorfail")
sns.violinplot(data=df, x='passorfail', y='cast_pressure', cut=0, inner='quartile', ax=ax[1])
ax[1].set_title("Violin plot by passorfail")
plt.tight_layout()
plt.show()

# 3) ECDF (누적분포)
plt.figure(figsize=(8,4))
sns.ecdfplot(data=df, x='cast_pressure', hue='passorfail')
plt.title("ECDF (누적분포) by passorfail")
plt.show()

# 4) 구간별 불량률 표 (10 분위 구간)
qbins = pd.qcut(df['cast_pressure'], q=10, duplicates='drop')
bin_stats = (df
             .assign(bin=qbins)
             .groupby('bin')
             .agg(n=('cast_pressure','size'),
                  mean_cp=('cast_pressure','mean'),
                  fail_rate=('passorfail','mean'))
             .reset_index())
bin_stats['fail_rate'] = (bin_stats['fail_rate']*100).round(2)

print("=== 구간별 불량률 (%) ===")
print(bin_stats)






# ============================================================================================
# biscuit_thickness와 passorfail 비교
# ============================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 한글 폰트 설정 (Windows 기본: Malgun Gothic)
plt.rcParams['font.family'] = 'Malgun Gothic'   # 그래프 폰트
plt.rcParams['axes.unicode_minus'] = False      # 음수 기호 깨짐 방지

# 🔹 데이터 준비
df = train_df[['biscuit_thickness', 'passorfail']].copy()
df = df.dropna(subset=['biscuit_thickness', 'passorfail'])
df['passorfail'] = df['passorfail'].astype(int)

print("=== 전체/클래스별 요약통계 ===")
print(df.groupby('passorfail')['biscuit_thickness'].describe(), "\n")

# 1) 히스토그램 + KDE
fig, ax = plt.subplots(1, 2, figsize=(14,4))
sns.histplot(data=df, x='biscuit_thickness', hue='passorfail',
             bins=60, stat='density', common_norm=False,
             element='step', fill=False, ax=ax[0])
ax[0].set_title("Histogram (밀도) by passorfail\n0=양품, 1=불량")

sns.kdeplot(data=df[df['passorfail']==0]['biscuit_thickness'], label='0(양품)', ax=ax[1])
sns.kdeplot(data=df[df['passorfail']==1]['biscuit_thickness'], label='1(불량)', ax=ax[1])
ax[1].legend()
ax[1].set_title("KDE (밀도추정) by passorfail")

plt.tight_layout()
plt.show()

# 2) 박스플롯 / 바이올린 플롯
fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.boxplot(data=df, x='passorfail', y='biscuit_thickness', ax=ax[0])
ax[0].set_title("Boxplot by passorfail")
sns.violinplot(data=df, x='passorfail', y='biscuit_thickness', cut=0, inner='quartile', ax=ax[1])
ax[1].set_title("Violin plot by passorfail")
plt.tight_layout()
plt.show()

# 3) ECDF (누적분포)
plt.figure(figsize=(8,4))
sns.ecdfplot(data=df, x='biscuit_thickness', hue='passorfail')
plt.title("ECDF (누적분포) by passorfail")
plt.show()

# 4) 구간별 불량률 표 (10 분위 구간)
qbins = pd.qcut(df['biscuit_thickness'], q=10, duplicates='drop')
bin_stats = (df
             .assign(bin=qbins)
             .groupby('bin')
             .agg(n=('biscuit_thickness','size'),
                  mean_bt=('biscuit_thickness','mean'),
                  fail_rate=('passorfail','mean'))
             .reset_index())
bin_stats['fail_rate'] = (bin_stats['fail_rate']*100).round(2)

print("=== 구간별 불량률 (%) ===")
print(bin_stats)





# ============================================================================================
# upper_mold_temp와 passorfail 비교
# ============================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_feature(df, col, target='passorfail', bins=60):
    """특정 칼럼에 대해 그룹별 분포/박스플롯/ECDF/구간별 불량률 확인"""
    print(f"\n=== {col} ===")
    tmp = df[[col, target]].dropna()
    tmp[target] = tmp[target].astype(int)

    # 요약통계
    print(tmp.groupby(target)[col].describe(), "\n")

    # 1) 히스토그램 + KDE
    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    sns.histplot(data=tmp, x=col, hue=target,
                 bins=bins, stat='density', common_norm=False,
                 element='step', fill=False, ax=ax[0])
    ax[0].set_title(f"Histogram (밀도) by {target}")

    sns.kdeplot(data=tmp[tmp[target]==0][col], label='0(양품)', ax=ax[1])
    sns.kdeplot(data=tmp[tmp[target]==1][col], label='1(불량)', ax=ax[1])
    ax[1].legend()
    ax[1].set_title(f"KDE (밀도추정) by {target}")
    plt.tight_layout()
    plt.show()

    # 2) Boxplot / Violin plot
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    sns.boxplot(data=tmp, x=target, y=col, ax=ax[0])
    ax[0].set_title(f"Boxplot by {target}")
    sns.violinplot(data=tmp, x=target, y=col, cut=0, inner='quartile', ax=ax[1])
    ax[1].set_title(f"Violin plot by {target}")
    plt.tight_layout()
    plt.show()

    # 3) ECDF
    plt.figure(figsize=(8,4))
    sns.ecdfplot(data=tmp, x=col, hue=target)
    plt.title(f"ECDF (누적분포) by {target}")
    plt.show()

    # 4) 구간별 불량률 표 (10 분위 구간)
    try:
        qbins = pd.qcut(tmp[col], q=10, duplicates='drop')
        bin_stats = (tmp
                     .assign(bin=qbins)
                     .groupby('bin')
                     .agg(n=(col,'size'),
                          mean_val=(col,'mean'),
                          fail_rate=(target,'mean'))
                     .reset_index())
        bin_stats['fail_rate'] = (bin_stats['fail_rate']*100).round(2)
        print("=== 구간별 불량률 (%) ===")
        print(bin_stats, "\n")
    except Exception as e:
        print("구간 분할 불가:", e)

# 🔹 세 변수 각각 분석 실행
for col in ['upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3']:
    analyze_feature(train_df, col)


train_df['upper_mold_temp3'].value_counts()



# ============================================================================================
# cast_pressure, biscuit_thickness, upper_mold_temp1,2,3 전처리 후 passorfail 그룹별 비교
# ============================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 0) 한글 폰트(Windows: Malgun Gothic)
# -----------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 1) 전처리
# -----------------------------
df = train_df.copy()

# (1) upper_mold_temp1/2에서 1400 이상인 비현실적 이상치 행 제거
df = df[(df['upper_mold_temp1'] < 1400) & (df['upper_mold_temp2'] < 1400)]

# (2) upper_mold_temp3 열 삭제
if 'upper_mold_temp3' in df.columns:
    df = df.drop(columns=['upper_mold_temp3'])

# (3) 분석에 필요한 열의 결측치 제거
cols_to_use = ['cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 'passorfail']
df = df.dropna(subset=cols_to_use).copy()

# (4) 타깃 정수 변환
df['passorfail'] = df['passorfail'].astype(int)

print("전처리 후 데이터 크기:", df.shape)

# -----------------------------
# 2) 공통 분석 함수
# -----------------------------
def analyze_feature(data, col, target='passorfail', bins=60):
    """특정 칼럼에 대해: 요약통계, 히스토그램+KDE, Boxplot/Violin, ECDF, 10-분위 불량률"""
    print(f"\n=== {col} ===")
    tmp = data[[col, target]].dropna().copy()
    tmp[target] = tmp[target].astype(int)

    # 요약통계(클래스별)
    print(tmp.groupby(target)[col].describe(), "\n")

    # 1) 히스토그램 + KDE
    fig, ax = plt.subplots(1, 2, figsize=(14,4))
    sns.histplot(data=tmp, x=col, hue=target,
                 bins=bins, stat='density', common_norm=False,
                 element='step', fill=False, ax=ax[0])
    ax[0].set_title(f"Histogram (밀도) by {target}\n0=양품, 1=불량")

    sns.kdeplot(tmp.loc[tmp[target]==0, col], label='0(양품)', ax=ax[1])
    sns.kdeplot(tmp.loc[tmp[target]==1, col], label='1(불량)', ax=ax[1])
    ax[1].legend()
    ax[1].set_title(f"KDE (밀도추정) by {target}")
    plt.tight_layout()
    plt.show()

    # 2) Boxplot / Violin plot
    fig, ax = plt.subplots(1, 2, figsize=(12,4))
    sns.boxplot(data=tmp, x=target, y=col, ax=ax[0])
    ax[0].set_title(f"Boxplot by {target}")
    sns.violinplot(data=tmp, x=target, y=col, cut=0, inner='quartile', ax=ax[1])
    ax[1].set_title(f"Violin plot by {target}")
    plt.tight_layout()
    plt.show()

    # 3) ECDF
    plt.figure(figsize=(8,4))
    sns.ecdfplot(data=tmp, x=col, hue=target)
    plt.title(f"ECDF (누적분포) by {target}")
    plt.tight_layout()
    plt.show()

    # 4) 10-분위 구간 불량률 표 (중복 경계 자동 제거)
    try:
        qbins = pd.qcut(tmp[col], q=10, duplicates='drop')
        bin_stats = (tmp
                     .assign(bin=qbins)
                     .groupby('bin')
                     .agg(n=(col,'size'),
                          mean_val=(col,'mean'),
                          fail_rate=(target,'mean'))
                     .reset_index())
        bin_stats['fail_rate'] = (bin_stats['fail_rate']*100).round(2)
        print("=== 구간별 불량률 (%) ===")
        print(bin_stats, "\n")
    except Exception as e:
        print("구간 분할 불가:", e)

# -----------------------------
# 3) 각각 실행
# -----------------------------
for col in ['cast_pressure', 'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2']:
    analyze_feature(df, col)





# 
train_df['mold_code'].value_counts()
train_df['molten_volume'].value_counts()
train_df['passorfail'].value_counts()
train_df = train_df.drop(index=19327)

test_df[['heating_furnace', 'molten_volume']]
test_df.loc[~test_df['molten_volume'].isna()]['heating_furnace'].isna()





# ================================================================================================
# molten_volume 결측치 채우기 iterative_imputer
# ================================================================================================
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

