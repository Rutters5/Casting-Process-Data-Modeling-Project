# %% 라이브러리 호출
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer
# %% 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
OUTPUT_FILE_TRAIN = BASE_DIR / "data" / "processed" / "train_v2.csv"
OUTPUT_FILE_TEST = BASE_DIR / "data" / "processed" / "test_v2.csv"
# %% raw 데이터 로드
df_raw = pd.read_csv(DATA_FILE)
df = df_raw.copy()
# %% 공통 처리
df['tryshot_signal'].fillna('N', inplace=True)
df = df.rename(columns={'date': '__tmp_swap__'})
df = df.rename(columns={'time': 'date', '__tmp_swap__': 'time'})

df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
df['registration_time'] = pd.to_datetime(df['registration_time'])
# ==================================================================================================
# 범주형으로 처리할 컬럼
# ==================================================================================================
df["mold_code"] = df["mold_code"].astype('object')
df["EMS_operation_time"] = df["EMS_operation_time"].astype('object')
# %% 대부분이 결측치인 행 확인 및 제거
# 해당 행이 유일한 emergency_stop 결측행이여서 이 행이 긴급중단을 나타내는 행이라고 판단
# 모델 예측 끝난 후에 ‘emergency_stop’이 결측인 경우 무조건 불량이라고 판정 내도록 만들기
# ==================================================================================================
df[df["emergency_stop"].isna()]
df.drop(df[df["emergency_stop"].isna()].index, inplace=True)
# %%
# ID 컬럼 제거
df.drop(columns=["id"], inplace=True)
# 단일값 컬럼 제거
df.drop(columns=["line", "name", "mold_name"], inplace=True)
# %% mold_temp3 제거
df.drop(columns=[ "upper_mold_temp3"], inplace=True)
df.drop(columns=[ "lower_mold_temp3"], inplace=True)

# %%
# ==================================================================================================
# 온도 센서 이상(1449) 결측치로 대치 
# ==================================================================================================
temp_cols = [c for c in df.columns if "temp" in c.lower()]

# 각 컬럼별 1400 이상 값 개수 집계
counts_over_1400 = (df[temp_cols] >= 1400).sum()

print(counts_over_1400)
# 온도 컬럼
temp_cols2 = [
    "molten_temp",
    "upper_mold_temp1","upper_mold_temp2",
    "lower_mold_temp1","lower_mold_temp2",
    "sleeve_temperature","Coolant_temperature"
]
df[temp_cols2] = df[temp_cols2].mask(df[temp_cols2] >= 1400, np.nan)
# ==================================================================================================
# 형체력 이상 결측치로 대치 
# ==================================================================================================
df["physical_strength"] = df["physical_strength"].mask(df["physical_strength"] >= 10000, np.nan)
# %%
df['physical_strength'].isnull().sum()
# %% 데이터 분할
n = len(df)
split_idx = int(n * 0.8)

train_df = df.iloc[:split_idx].copy()
test_df  = df.iloc[split_idx:].copy()

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
# %%
# ==================================================================================================
# 결측치 대치(시간, mold_code로 선형 보간)
# ==================================================================================================
cols_interp = [
    "molten_temp",
    "upper_mold_temp1","upper_mold_temp2",
    "lower_mold_temp1","lower_mold_temp2",
    "sleeve_temperature","Coolant_temperature",
    "physical_strength",
]

# 세그먼트 분할 함수(구간 요조정)
def add_segment(g, time_col="registration_time", gap_minutes=30):
    g = g.sort_values(time_col).copy()
    dt = g[time_col].diff().dt.total_seconds().div(60)
    g["__seg__"] = (dt.isna() | (dt > gap_minutes)).cumsum()
    return g
out = []
for _, g0 in train_df.groupby("mold_code", dropna=False):
    g1 = add_segment(g0, "registration_time", gap_minutes=30)
    for _, seg in g1.groupby("__seg__", dropna=False):
        seg = seg.sort_values("registration_time").set_index("registration_time")

        # 시간축 기반 선형 보간 (양끝단 보간은 하지 않음: limit_area='inside')
        seg[cols_interp] = seg[cols_interp].interpolate(
            method="time", limit_area="inside"
        )

        # 필요 시 양끝단도 채우고 싶다면 아래 두 줄 중 택1 추가:
        seg[cols_interp] = seg[cols_interp].ffill().bfill()          # 계단식 보정
        seg[cols_interp] = seg[cols_interp].interpolate(method="time", limit_direction="both")

        out.append(seg.reset_index())

# 병합
train_df = pd.concat(out, ignore_index=True).sort_values(["mold_code","registration_time"])
train_df.drop(columns='__seg__', inplace=True)
# 남은 결측치는 중앙값
train_df["physical_strength"] = (
    train_df["physical_strength"]
    .fillna(train_df.groupby("mold_code")["physical_strength"].transform("median"))
)

# test 데이터 처리
# 세그먼트 분할 함수(구간 요조정)
out = []
for _, g0 in test_df.groupby("mold_code", dropna=False):
    g1 = add_segment(g0, "registration_time", gap_minutes=30)
    for _, seg in g1.groupby("__seg__", dropna=False):
        seg = seg.sort_values("registration_time").set_index("registration_time")

        # 시간축 기반 선형 보간 (양끝단 보간은 하지 않음: limit_area='inside')
        seg[cols_interp] = seg[cols_interp].interpolate(
            method="time", limit_area="inside"
        )

        out.append(seg.reset_index())

test_df = pd.concat(out, ignore_index=True).sort_values(["mold_code","registration_time"])
test_df.drop(columns='__seg__', inplace=True)
# %%
# ==================================================================================================
# 결측치 처리 (molten_temp)
# 처리 방법 : 동일코드 앞 생산 온도, 동일 코드 뒤 생산 온도 평균
# ==================================================================================================
# 원본 molten_temp를 새로운 열로 복사
train_df['molten_temp_filled'] = train_df['molten_temp']

# 코드별 시간 순 정렬 후 선형 보간
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)

# 채워진 컬럼으로 교체
train_df.drop(columns=["molten_temp"], inplace=True)
train_df = train_df.rename(columns={'molten_temp_filled': 'molten_temp'})
# test 데이터 처리
test_df['molten_temp_filled'] = test_df['molten_temp']

# 코드별 시간 순 정렬 후 선형 보간
test_df['molten_temp_filled'] = (
    test_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
test_df['molten_temp_filled'] = (
    test_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)

# 채워진 컬럼으로 교체
test_df.drop(columns=["molten_temp"], inplace=True)
test_df = test_df.rename(columns={'molten_temp_filled': 'molten_temp'})

# %% 변수 추가
upper_cols = ["upper_mold_temp1", "upper_mold_temp2"]
lower_cols = ["lower_mold_temp1", "lower_mold_temp2"]

# 표준편차 (4포인트 전체)
train_df["uniformity"] =train_df[upper_cols + lower_cols].std(axis=1)
# 상부-하부 평균 차이
train_df["mold_temp_udiff"] =train_df[upper_cols].mean(axis=1) - train_df[lower_cols].mean(axis=1)
# 파스칼 원리
train_df['P_diff'] = train_df['physical_strength'] - train_df['cast_pressure']
# 사이클 타임 이상
train_df['Cycle_diff'] = train_df['production_cycletime'] - train_df['facility_operation_cycleTime']

# test
test_df["uniformity"] =test_df[upper_cols + lower_cols].std(axis=1)
test_df["mold_temp_udiff"] =test_df[upper_cols].mean(axis=1) - test_df[lower_cols].mean(axis=1)
test_df['P_diff'] = test_df['physical_strength'] - test_df['cast_pressure']
test_df['Cycle_diff'] = test_df['production_cycletime'] - test_df['facility_operation_cycleTime']
# %%
# ==================================================================================================
# 컬럼 제거 (heating_furnace)
# 결측치 총 40880개 (mold_code 8600은 전부 다 결측치(2960개), 8722도 전부 다 결측치(19664개))
# 일단은 제외 (3개 이상의 종류이지만 구분이 어려움, 결과에 큰 영향을 미치지 않을 것이라 판단)
# ==================================================================================================
train_df.drop(columns=["heating_furnace"], inplace=True)
test_df.drop(columns=["heating_furnace"], inplace=True)
# ==================================================================================================
# 컬럼 제거 (molten_volume)
# ==================================================================================================
train_df.drop(columns=["molten_volume"], inplace=True)
test_df.drop(columns=["molten_volume"], inplace=True)
# 인코딩, 스케일링
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer

# categorical_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
# numeric_cols = df.select_dtypes(include=["int64","float64"]).columns.tolist()

# categorical_transformer = OneHotEncoder(handle_unknown="ignore")
# numeric_transformer = StandardScaler()

# %%
train_df[train_df['physical_strength'].isnull()]
train_df.columns
# %% mold 별 df 생성
# df_8412 = df[df['mold_code'] == 8412]
# df_8722 = df[df['mold_code'] == 8722]
# df_8573 = df[df['mold_code'] == 8573]
# df_8917 = df[df['mold_code'] == 8917]
# df_8600 = df[df['mold_code'] == 8600]
# %%
#df_8600["upper_mold_temp1"].corr(df_8600["lower_mold_temp1"], method='pearson')
# %% 8412
train_df.to_csv(OUTPUT_FILE_TRAIN)
test_df.to_csv(OUTPUT_FILE_TEST)

