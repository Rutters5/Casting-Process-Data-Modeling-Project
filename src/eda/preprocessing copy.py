import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 한글 폰트 설정
#plt.rcParams['font.family'] = 'Malgun Gothic'
#plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
train_df = pd.read_csv("../../data/raw/train.csv")
test_df = pd.read_csv("../../data/raw/test.csv")

train_df.info()
train_df.head()
train_df.isna().sum()

# 대부분이 결측치인 행 제거
train_df.drop(19327, inplace=True)

# 이상치 처리

# upper_mold_temp2
train_df['upper_mold_temp2'].hist()
train_df['upper_mold_temp2'].describe()
train_df[train_df['upper_mold_temp2']==4232]
train_df.drop(42632,inplace=True)

# 분석에서 필요없는 컬럼 제거
train_df.drop(columns=["id", "line", "name", "mold_name", "emergency_stop"], inplace=True)

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

############################################################################################################

# moltn_volume 2500기준으로 나누어서 보기
train_df['molten_volume'].hist()

train_df[train_df['molten_volume']>2000]['mold_code'].value_counts() # 99퍼센트가 8412

# molten_volume > 2000 & mold_code != 8412
df_other_molds = train_df[(train_df['molten_volume'] > 2000) & (train_df['mold_code'] != 8412)].copy()

# mold_code별 DataFrame 나누기
mold_dfs = {mold: df for mold, df in df_other_molds.groupby('mold_code')}

# 현재 df_other_molds에 있는 mold_code 확인
print("Available mold_code:", list(mold_dfs.keys()))

# 실제 존재하는 mold_code 데이터 확인
some_mold = list(mold_dfs.keys())[1]  # 첫 번째 mold_code 선택
print(f"Mold {some_mold} 데이터:")
mold_dfs[some_mold]
# 다른 코드들도 과도하게 많이 부어도 반드시 오류는 아님

# 분석의 편의를 위해 우선 8412코드를 제외하고 2500이상인 애들은 제외
# df_other_molds의 index 가져오기
remove_index = df_other_molds.index

# train_df에서 해당 index 제거
train_df = train_df.drop(remove_index).reset_index(drop=True)
# 
# 보고 싶은 컬럼만 선택
train_df = train_df[~(train_df['tryshot_signal']=="D")]
train_df['tryshot_signal'].value_counts()
df_selected = train_df[['time','date','count','molten_volume','mold_code','sleeve_temperature','passorfail']].copy()
df_selected.dropna(subset=['molten_volume'], inplace=True)
df_selected = df_selected[df_selected['molten_volume']<2000]

df_selected_8412 = df_selected[df_selected['mold_code']==8412].reset_index(drop=True)

# 예시: 데이터 불러오기
# df = pd.read_csv("train.csv")  

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
# 중간에 한번 채우고 다시 감소하고 다시 채우고
# 
train_8412 = train_df[train_df['mold_code']==8412]
train_8412[train_8412['molten_volume']>2500]['count'].head(30)

train_df.info()

import pandas as pd

# molten_volume이 결측치가 아닌 데이터만 선택
df_notnull = train_df.dropna(subset=['molten_volume'])

# 수치형 열만 선택
numeric_cols = df_notnull.select_dtypes(include=['int64', 'float64']).columns

# molten_volume과 나머지 수치형 열의 상관계수 계산
corr_with_molten = df_notnull[numeric_cols].corr()['molten_volume'].sort_values(ascending=False)

print(corr_with_molten)

# 상관관계가 낮다고 나오는데 선형관계가 아니여서 그럼
train_df

import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression

# molten_volume과 비교할 숫자형 컬럼만 선택
numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
numeric_cols.remove('molten_volume')  # molten_volume 제외

# 결과를 담을 DataFrame
results = pd.DataFrame(columns=['Feature', 'Pearson', 'Spearman', 'Mutual_Info'])

for feature in numeric_cols:
    # Pearson 상관계수
    pearson_corr, _ = pearsonr(train_df[feature], train_df['molten_volume'])
    
    # Spearman 상관계수
    spearman_corr, _ = spearmanr(train_df[feature], train_df['molten_volume'])
    
    # Mutual Information
    mi = mutual_info_regression(train_df[[feature]], train_df['molten_volume'])
    
    # 결과 저장
    results = pd.concat([results, pd.DataFrame({
        'Feature':[feature],
        'Pearson':[pearson_corr],
        'Spearman':[spearman_corr],
        'Mutual_Info':[mi[0]]
    })], ignore_index=True)

# Mutual Info 기준 내림차순 정렬
results = results.sort_values(by='Mutual_Info', ascending=False).reset_index(drop=True)

print(results)

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 8412 데이터 선택
df_8412 = train_df[train_df['mold_code'] == 8412].copy()
df_8412 = df_8412[df_8412['molten_volume']>2000]

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 두 그룹 나누기 (조건은 필요에 맞게 수정 가능)
df_8412['mold_group'] = df_8412['molten_volume'].apply(lambda v: 'High (~2000)' if v > 1500 else 'Low (~700)')

features = ['lower_mold_temp1', 'high_section_speed', 'cast_pressure']

# 각 그룹 + 각 feature별 독립 그래프
for group in df_8412['mold_group'].unique():
    sub = df_8412[df_8412['mold_group'] == group]
    
    for feature in features:
        plt.figure(figsize=(6,5))
        sns.scatterplot(x=sub[feature], y=sub['molten_volume'], alpha=0.6)
        sns.regplot(x=sub[feature], y=sub['molten_volume'], scatter=False, color='red')
        plt.xlabel(feature)
        plt.ylabel('Molten Volume')
        
        # Spearman 상관계수
        corr, _ = spearmanr(sub[feature], sub['molten_volume'])
        plt.title(f'Group: {group}\n{feature} vs Molten Volume\nSpearman r = {corr:.3f}')
        
        plt.tight_layout()
        plt.show()



# mold_code별로 그래프 그리기
mold_codes = df_selected['mold_code'].unique()

plt.figure(figsize=(15, 10))

for i, mold in enumerate(mold_codes, 1):
    plt.subplot(len(mold_codes), 1, i)
    mold_df = df_selected [df_selected ['mold_code'] == mold].head(300)
    sns.scatterplot(data=mold_df, x='count', y='sleeve_temperature', hue='passorfail', palette='Set1', alpha=0.6)
    plt.title(f'Mold Code: {mold}')
    plt.xlabel('Count')
    plt.ylabel('Molten Volume')

plt.tight_layout()
plt.show()

train_df.groupby(['time','count'])['passorfail'].count().sort_values()


train_df[train_df['']]

train_df[train_df['mold_code']==8412]['count'].head(30)
train8412 = train_df[train_df['mold_code']==8412]
train8412[train8412['count']==32]
train8412['tryshot_signal'].value_counts()


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
