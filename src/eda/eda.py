import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 경로 설정
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

# 데이터 로드
train_df = pd.read_csv(DATA_FILE)

# 데이터 정보
train_df.info()
train_df.columns

# 시간별 생산량
# 확인결과 7시, 19시는 생산량이 매우 적음
df = train_df.copy()
df["hour"] = df["time"].dt.hour
by_hour = df.groupby('hour').size().reindex(range(24), fill_value=0)

ax = by_hour.plot(kind='bar', figsize=(10,4))
ax.set_xlabel('Hour of day (0-23)')
ax.set_ylabel('Count (rows)')
ax.set_title('Production Count by Hour')
plt.tight_layout()
plt.show()