import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv("../../data/raw/train.csv")
test_df = pd.read_csv("../../data/raw/test.csv")

train_df.info()
train_df.head()
train_df.isna().sum()

# 대부분이 결측치인 행 제거
train_df.drop(19327, inplace=True)

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop", "registration_time"], inplace=True)

'''
결측치 처리 (molten_temp)
동일코드 앞 생산 온도, 동일 코드 뒤 생산 온도 평균
'''
# 🔹 원본 molten_temp를 새로운 열로 복사
train_df['molten_temp_filled'] = train_df['molten_temp']

# 🔹 금형별 시간 순 정렬 후 선형 보간
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.interpolate(method='linear'))
)

# 🔹 여전히 남아있는 결측치(맨 앞/뒤)는 그룹별 중앙값으로 채우기
train_df['molten_temp_filled'] = (
    train_df.groupby('mold_code')['molten_temp_filled'].transform(lambda x: x.fillna(x.median()))
)
train_df[['molten_temp', 'molten_temp_filled']]
train_df['molten_temp'].isna().sum()
train_df['molten_temp_filled'].isna().sum()
train_df.drop(columns=["molten_temp"], inplace=True)

'''
결측치 처리 (molten_volume)
'''
train_df.loc[train_df["molten_volume"].isna(), :]

custom_colors = {
    8412 : '#2ca02c',
    8573 : '#ff7f0e',
    8600 : "#ff0e0e",
    8722 : "#ffd70e",
    8917 : '#2ca02c'
}

# 코드별 전자교반 시간
train_df.groupby(["mold_code", "EMS_operation_time"])["passorfail"].count()

# 코드별 형체력 
sns.histplot(data=train_df.loc[(train_df["physical_strength"]<10000) & (train_df["physical_strength"]>600), :], x='physical_strength', hue='mold_code', kde=True)

# 코드별 주조 압력
sns.histplot(data=train_df.loc[train_df["cast_pressure"]>300, :], x='cast_pressure', hue='mold_code', kde=True)
sns.histplot(data=train_df.loc[(train_df["cast_pressure"]>300) & train_df["mold_code"].isin([8573, 8600, 8722]), :], x='cast_pressure', hue='mold_code', kde=True)

# 코드별 냉각수 온도
sns.histplot(data=train_df.loc[train_df["Coolant_temperature"] < 150, :], x='Coolant_temperature', hue='mold_code', palette=custom_colors, kde=True)
sns.histplot(data=train_df.loc[(train_df["Coolant_temperature"] < 150) & (train_df["mold_code"]).isin([8573, 8600, 8722]), :], x='Coolant_temperature', hue='mold_code', kde=True)

# 코드별 설비 작동 사이클 시간
sns.histplot(data=train_df.loc[(train_df["facility_operation_cycleTime"]<150) & (train_df["facility_operation_cycleTime"]>80), :], x='facility_operation_cycleTime', hue='mold_code', palette=custom_colors, kde=True)



# ==================================================================================================
# mold_code별 molten_volume 결측치 개수 확인
# ==================================================================================================
# 전체 개수 (결측 포함)
total = train_df.groupby('mold_code').size()

# 결측치 제외한 개수
non_null = train_df.groupby('mold_code')['molten_volume'].count()

# 결측치 개수 = 전체 - 결측 아닌 값
missing = total - non_null

# 하나의 데이터프레임으로 합치기
missing_df = pd.DataFrame({
    'total_rows': total,
    'non_null': non_null,
    'missing': missing
})

print(missing_df)





# ===========================================================================================
# mold_code별 molten_volume과 count의 관계 확인
# ===========================================================================================
# 보고 싶은 컬럼만 선택
train_df = train_df[~(train_df['tryshot_signal']=="D")] # tryshot_signal이 결측치인 경우(정상동작)이 아닌 경우만 고름
train_df['tryshot_signal'].value_counts() # 시험생산인 경우 확인: 1244개 -> 0개
df_selected = train_df[['time','date','count','molten_volume','mold_code','sleeve_temperature','lower_mold_temp2','passorfail']].copy()
df_selected.dropna(subset=['molten_volume'], inplace=True) # molten_volume이 결측치인 경우 제외함
df_selected = df_selected[df_selected['molten_volume']<2000] # 이 중 molten_volume이 2000 미만인 경우만 고름

# mold_code별로 그래프 그리기
mold_codes = df_selected['mold_code'].unique()

plt.figure(figsize=(15, 10))
for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = df_selected [df_selected['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='molten_volume', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('Molten Volume')

plt.tight_layout()
plt.show()

# molten을 한 번 채운 뒤 쭉 사용하다가 일정 수준 이하로 떨어지면 다시 채워넣음





# ==================================================================================================
# mold_code별 Sleeve temperature와 count의 관계 확인
# ==================================================================================================
# mold_code별로 그래프 그리기
mold_codes = df_selected['mold_code'].unique()

plt.figure(figsize=(15, 10))

for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = df_selected [df_selected ['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='sleeve_temperature', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('Sleeve Temperature')

plt.tight_layout()
plt.show()

# ==================================================================================================
# mold_code별 lower_mold_temp2와 count의 관계 확인
# ==================================================================================================
# mold_code별로 그래프 그리기

mold_codes = df_selected['mold_code'].unique()

plt.figure(figsize=(15, 10))

for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = df_selected [df_selected ['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='lower_mold_temp2', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('lower_mold_temp2')

plt.tight_layout()
plt.show()




# ==================================================================================================
# heating_furnace 열을 버리는 이유
# (1) NaN이 2개 이상의 그룹으로 나뉨
# (2) molten_volume을 한 번 채울 때마다 count가 새로 시작되는데, 그때마다 furnace를 바꾸지 않는다고 확신할 수 없음
# ==================================================================================================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_FILE_train = BASE_DIR / "data" / "processed" / "train_v1.csv"
train_df = pd.read_csv(DATA_FILE_train)

pd.set_option('display.max_rows', None)
#train_df.loc[~(train_df['heating_furnace'].isna())][['mold_code', 'heating_furnace']].tail(70)
#train_df.loc[73406:73450, ['heating_furnace', 'mold_code', 'time', 'date', 'molten_volume', 'count']]

train_df.info()

# ==================================================================================================

# 한글 폰트 설정 (윈도우: 맑은 고딕 / 맥: AppleGothic / 리눅스: 나눔고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'   # 또는 'AppleGothic', 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지

# ==================================================================================================

# ==================================================================================================
# lower_temp1과 lower_temp2 유사성 및 상관관계 확인
# (1) 트리 기반 모델은 다른 모델에 비해 다중 공선성에 강하지만 변수 중요도가 왜곡 될 수 있음 (어떨떈 a가 중요하고 어떨떈 b가 중요하고)
# (2) 중요도가 분산돼버림.
# ==================================================================================================
# ====================================
# 1. 상관관계 계산
# ====================================
corr = train_df["lower_mold_temp1"].corr(train_df["lower_mold_temp2"])
print(f"상관계수 (lower_mold_temp1 vs lower_mold_temp2): {corr:.4f}")

# -0.06으로 상관관계가 없음

# ====================================
# 2. 분포 비교 (히스토그램 + KDE)
# ====================================
plt.figure(figsize=(10,6))
sns.kdeplot(train_df["lower_mold_temp1"].dropna(), label="lower_mold_temp1", shade=True)
sns.kdeplot(train_df["lower_mold_temp2"].dropna(), label="lower_mold_temp2", shade=True)
plt.title("Distribution Comparison (KDE): lower_mold_temp1 vs lower_mold_temp2")
plt.xlabel("Temperature")
plt.ylabel("Density")
plt.legend()
plt.show()

# 히스토그렘 결과상 두 열은 서로 다른 열이라고 확인하고 두개 열 모두 모델 학습에 반영

# ====================================
# 2. 산점도
# ====================================
plt.figure(figsize=(6,6))

# 먼저 불량(1) 찍기 → 뒤에 정상(0)이 덮어쓰게 됨
sns.scatterplot(
    x="lower_mold_temp1",
    y="lower_mold_temp2",
    data=train_df[train_df["passorfail"] == 1],
    color="red",
    alpha=0.4,
    label="불량(1)"
)

# 정상(0) 찍기 → 위에 표시됨
sns.scatterplot(
    x="lower_mold_temp1",
    y="lower_mold_temp2",
    data=train_df[train_df["passorfail"] == 0],
    color="blue",
    alpha=0.4,
    label="정상(0)"
)

plt.title("Scatter Plot: lower_mold_temp1 vs lower_mold_temp2 by Pass/Fail")
plt.xlabel("lower_mold_temp1")
plt.ylabel("lower_mold_temp2")
plt.legend(title="Pass/Fail")
plt.show()

                                                   
# 불량 정상 나누어서 데이터 분포 비교

# 데이터 분리
normal_df = train_df[train_df['passorfail'] == 0]
defect_df = train_df[train_df['passorfail'] == 1]

# 그래프 크기 설정
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# 정상 제품 산점도
sns.scatterplot(
    x="lower_mold_temp1",
    y="lower_mold_temp2",
    data=normal_df,
    color="blue",
    alpha=0.4,
    ax=axes[0]
)
axes[0].set_title("정상 제품 (passorfail=0)")
axes[0].set_xlabel("하부 금형 온도 1")
axes[0].set_ylabel("하부 금형 온도 2")

# 불량 제품 산점도
sns.scatterplot(
    x="lower_mold_temp1",
    y="lower_mold_temp2",
    data=defect_df,
    color="red",
    alpha=0.4,
    ax=axes[1]
)
axes[1].set_title("불량 제품 (passorfail=1)")
axes[1].set_xlabel("하부 금형 온도 1")
axes[1].set_ylabel("하부 금형 온도 2")

plt.suptitle("정상 vs 불량: 하부 금형 온도 관계", fontsize=14)
plt.tight_layout()
plt.show()



