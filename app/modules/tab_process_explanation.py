from shiny import ui, render, reactive
import pandas as pd
from pathlib import Path
import numpy as np

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"
SCALER_FILE = BASE_DIR / "data" / "interim" / "std_scaler_v1.joblib"
MODEL_FILE = BASE_DIR / "data" / "interim" / "rf_model_v1.joblib"
PDP_IMAGE_FILE = BASE_DIR / "data" / "png" / "RF_basic_PDP.png"

df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)

# 숫자지만 범주형으로 취급할 컬럼들을 문자열로 변환
df['mold_code'] = df['mold_code'].astype(str)
df['passorfail'] = df['passorfail'].astype(str)

def get_variable_mapping():
    """영문 변수명을 한글(영문)-(타입) 형식으로 매핑"""
    
    # 수치형 변수 매핑
    numeric_mapping = {
        'count': '일자별 생산 번호(count)-수치형',
        'molten_temp': '용탕 온도(molten_temp)-수치형',
        'facility_operation_cycleTime': '설비 작동 사이클 시간(facility_operation_cycleTime)-수치형',
        'production_cycletime': '제품 생산 사이클 시간(production_cycletime)-수치형',
        'low_section_speed': '저속 구간 속도(low_section_speed)-수치형',
        'high_section_speed': '고속 구간 속도(high_section_speed)-수치형',
        'molten_volume': '용탕량(molten_volume)-수치형',
        'cast_pressure': '주조 압력(cast_pressure)-수치형',
        'biscuit_thickness': '비스켓 두께(biscuit_thickness)-수치형',
        'upper_mold_temp1': '상금형 온도1(upper_mold_temp1)-수치형',
        'upper_mold_temp2': '상금형 온도2(upper_mold_temp2)-수치형',
        'upper_mold_temp3': '상금형 온도3(upper_mold_temp3)-수치형',
        'lower_mold_temp1': '하금형 온도1(lower_mold_temp1)-수치형',
        'lower_mold_temp2': '하금형 온도2(lower_mold_temp2)-수치형',
        'lower_mold_temp3': '하금형 온도3(lower_mold_temp3)-수치형',
        'sleeve_temperature': '슬리브 온도(sleeve_temperature)-수치형',
        'physical_strength': '형체력(physical_strength)-수치형',
        'Coolant_temperature': '냉각수 온도(Coolant_temperature)-수치형',
        'EMS_operation_time': '전자교반 가동 시간(EMS_operation_time)-수치형'
    }
    
    # 범주형 변수 매핑
    categorical_mapping = {
        'working': '가동 여부(working)-범주형',
        'passorfail': '양품/불량 판정(passorfail)-범주형',
        'tryshot_signal': '사탕 신호(tryshot_signal)-범주형',
        'heating_furnace': '가열로 구분(heating_furnace)-범주형',
        'mold_code': '금형 코드(mold_code)-범주형'
    }
    
    return numeric_mapping, categorical_mapping

def classify_columns(df):
    """데이터의 실제 의미에 따라 컬럼을 분류"""
    
    # 제외할 컬럼들 (ID, 시간, 불필요한 범주형)
    exclude_columns = ['id', 'line', 'name', 'mold_name', 'emergency_stop', 
                      'time', 'date', 'registration_time']
    
    # 실제 수치형 컬럼 (연속형 측정값들)
    numeric_columns = [
        'count', 'molten_temp', 'facility_operation_cycleTime', 'production_cycletime',
        'low_section_speed', 'high_section_speed', 'molten_volume', 'cast_pressure',
        'biscuit_thickness', 'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
        'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3', 'sleeve_temperature',
        'physical_strength', 'Coolant_temperature', 'EMS_operation_time'
    ]
    
    # 범주형 컬럼 (object 타입)
    categorical_columns = ['working', 'passorfail', 'tryshot_signal', 'heating_furnace', 'mold_code']
    
    # 실제 존재하는 컬럼만 필터링
    numeric_columns = [col for col in numeric_columns if col in df.columns and col not in exclude_columns]
    categorical_columns = [col for col in categorical_columns if col in df.columns and col not in exclude_columns]
    
    # 한글 매핑 가져오기
    numeric_mapping, categorical_mapping = get_variable_mapping()
    
    # 선택지 딕셔너리 생성 (수치형 먼저, 범주형 나중)
    choices_dict = {}
    
    # 수치형 변수 추가
    for col in numeric_columns:
        choices_dict[col] = numeric_mapping.get(col, col)
    
    # 범주형 변수 추가
    for col in categorical_columns:
        choices_dict[col] = categorical_mapping.get(col, col)
    
    return {
        'numeric': numeric_columns,
        'categorical': categorical_columns,
        'all': numeric_columns + categorical_columns,  # 수치형 먼저
        'choices': choices_dict
    }

def panel():
    # 컬럼 분류
    col_types = classify_columns(df)
    
    return ui.nav_panel(
        "EDA 분석",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("📊 변수 선택"),
                ui.p("두 변수가 같으면 히스토그램, 다르면 산점도/박스플롯/히트맵이 표시됩니다.", 
                     style="font-size: 0.9em; color: #666;"),
                
                # 첫 번째 변수 선택
                ui.input_selectize(
                    "var1",
                    "변수 1",
                    choices=col_types['choices'],
                    selected=col_types['all'][0] if len(col_types['all']) > 0 else None
                ),
                
                # 두 번째 변수 선택
                ui.input_selectize(
                    "var2",
                    "변수 2",
                    choices=col_types['choices'],
                    selected=col_types['all'][0] if len(col_types['all']) > 0 else None
                ),
                
                ui.hr(),
                
                # 선택된 변수 정보 표시
                ui.output_text("selection_info"),
                
                width=350
            ),
            
            ui.div(
                ui.h3("🔍 탐색적 데이터 분석 (EDA)"),
                
                # 데이터셋 정보 + 선택된 변수 통계 (통합)
                ui.card(
                    ui.card_header("📋 데이터셋 정보 및 통계"),
                    ui.div(
                        ui.output_text("data_info"),
                        ui.hr(),
                        ui.output_data_frame("selected_stats")
                    )
                ),
                
                ui.br(),
                
                # 시각화 결과
                ui.card(
                    ui.card_header("📈 시각화 결과"),
                    ui.output_plot("eda_plots", height="600px")
                )
            )
        )
    )

def server(input, output, session):
    
    # 영문 변수명을 한글명으로 변환
    def get_korean_name(eng_name):
        numeric_mapping, categorical_mapping = get_variable_mapping()
        all_mapping = {**numeric_mapping, **categorical_mapping}
        return all_mapping.get(eng_name, eng_name)
    
    # 선택된 변수들을 반환하는 reactive
    @reactive.Calc
    def get_selected_vars():
        var1 = input.var1()
        var2 = input.var2()
        
        if var1 and var2:
            if var1 == var2:
                return [var1]
            else:
                return [var1, var2]
        elif var1:
            return [var1]
        elif var2:
            return [var2]
        else:
            return []
    
    @output
    @render.text
    def selection_info():
        var1 = input.var1()
        var2 = input.var2()
        
        if not var1 or not var2:
            return "⚠️ 두 변수를 모두 선택해주세요"
        elif var1 == var2:
            return f"✅ 동일 변수 선택\n→ 히스토그램 표시"
        else:
            col_types = classify_columns(df)
            is_numeric1 = var1 in col_types['numeric']
            is_numeric2 = var2 in col_types['numeric']
            
            if is_numeric1 and is_numeric2:
                viz_type = "산점도"
            elif is_numeric1 or is_numeric2:
                viz_type = "박스플롯"
            else:
                viz_type = "히트맵"
            
            return f"✅ 두 변수 선택\n→ {viz_type} 표시"
    
    @output
    @render.text
    def data_info():
        selected = get_selected_vars()
        
        info_text = f"📊 전체 데이터 행 수: {len(df):,}개"
        
        if len(selected) > 0:
            info_text += "\n\n❌ 선택된 변수의 결측값:"
            for col in selected:
                korean_name = get_korean_name(col)
                missing_count = df[col].isnull().sum()
                missing_pct = (missing_count / len(df)) * 100
                info_text += f"\n   • {korean_name}: {missing_count:,}개 ({missing_pct:.1f}%)"
        
        return info_text
    
    @output
    @render.data_frame
    def selected_stats():
        selected = get_selected_vars()
        
        if len(selected) == 0:
            return pd.DataFrame()
        
        # 선택된 컬럼의 데이터
        selected_df = df[selected].copy()
        
        # 컬럼명을 한글로 변경
        column_rename = {col: get_korean_name(col) for col in selected}
        selected_df.columns = [column_rename.get(col, col) for col in selected_df.columns]
        
        # 수치형 데이터 기술통계
        numeric_data = selected_df.select_dtypes(include=[np.number])
        categorical_data = selected_df.select_dtypes(include=['object'])
        
        result = pd.DataFrame()
        
        if not numeric_data.empty:
            stats = numeric_data.describe().round(3)
            # 인덱스를 리셋하여 통계량 이름을 컬럼으로 변환
            stats = stats.reset_index()
            stats = stats.rename(columns={'index': '통계량'})
            
            # 통계량에 한글 설명 매핑
            stats_mapping = {
                'count': 'count (데이터 개수)',
                'mean': 'mean (평균)',
                'std': 'std (표준편차)',
                'min': 'min (최솟값)',
                '25%': '25% (1사분위수)',
                '50%': '50% (중앙값)',
                '75%': '75% (3사분위수)',
                'max': 'max (최댓값)'
            }
            stats['통계량'] = stats['통계량'].map(stats_mapping)
            
            return stats
        
        if not categorical_data.empty:
            cat_result = []
            for col in categorical_data.columns:
                cat_result.append({
                    '변수': col,
                    '고유값 개수': categorical_data[col].nunique(),
                    '최빈값': categorical_data[col].mode()[0] if len(categorical_data[col].mode()) > 0 else 'N/A',
                    '최빈값 빈도': categorical_data[col].value_counts().iloc[0] if len(categorical_data[col]) > 0 else 0
                })
            return pd.DataFrame(cat_result)
        
        return pd.DataFrame()
    
    @output
    @render.plot
    def eda_plots():
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # 한글 폰트 설정 (맥/윈도우 호환)
        try:
            plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우
        except:
            try:
                plt.rcParams['font.family'] = 'AppleGothic'  # 맥
            except:
                plt.rcParams['font.family'] = 'DejaVu Sans'  # 기본
        
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        var1 = input.var1()
        var2 = input.var2()
        
        # 변수가 선택되지 않은 경우
        if not var1 or not var2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, '두 변수를 모두 선택해주세요', 
                   ha='center', va='center', fontsize=16)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        col_types = classify_columns(df)
        
        # 한글 이름 가져오기
        korean_name1 = get_korean_name(var1)
        korean_name2 = get_korean_name(var2)
        
        # 같은 변수 선택: 히스토그램
        if var1 == var2:
            col = var1
            korean_name = korean_name1
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 변수 타입 확인
            is_numeric = col in col_types['numeric']
            
            # 수치형 변수: 히스토그램
            if is_numeric:
                data_clean = df[col].dropna()
                ax.hist(data_clean, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
                ax.set_title(f'{korean_name} 분포', fontsize=14, pad=20)
                ax.set_xlabel(korean_name, fontsize=12)
                ax.set_ylabel('빈도', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
            
            # 범주형 변수: 막대그래프
            else:
                data_with_nan = df[col].fillna('NaN')
                value_counts = data_with_nan.value_counts().head(15)
                
                bars = ax.bar(range(len(value_counts)), value_counts.values, color='coral', alpha=0.7)
                ax.set_title(f'{korean_name} 분포', fontsize=14, pad=20)
                ax.set_xlabel(korean_name, fontsize=12)
                ax.set_ylabel('빈도', fontsize=12)
                ax.set_xticks(range(len(value_counts)))
                ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # 다른 변수 선택: 산점도/박스플롯/히트맵
        else:
            col1, col2 = var1, var2
            
            # 변수 타입 확인
            is_numeric1 = col1 in col_types['numeric']
            is_numeric2 = col2 in col_types['numeric']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Case 1: 둘 다 수치형 → 산점도
            if is_numeric1 and is_numeric2:
                plot_df = df[[col1, col2]].dropna()
                
                ax.scatter(plot_df[col1], plot_df[col2], alpha=0.5, s=20, color='steelblue')
                ax.set_xlabel(korean_name1, fontsize=12)
                ax.set_ylabel(korean_name2, fontsize=12)
                ax.set_title(f'{korean_name1} vs {korean_name2} (산점도)\n(유효 데이터: {len(plot_df):,}개)', fontsize=14, pad=20)
                ax.grid(True, alpha=0.3)
                
                # 상관계수
                correlation = plot_df[col1].corr(plot_df[col2])
                ax.text(0.05, 0.95, f'상관계수: {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Case 2: 하나는 범주형, 하나는 수치형 → 박스플롯
            elif (is_numeric1 and not is_numeric2) or (not is_numeric1 and is_numeric2):
                # 범주형과 수치형 구분
                if is_numeric1:
                    num_col = col1
                    cat_col = col2
                    num_korean = korean_name1
                    cat_korean = korean_name2
                else:
                    num_col = col2
                    cat_col = col1
                    num_korean = korean_name2
                    cat_korean = korean_name1
                
                plot_df = df[[cat_col, num_col]].dropna()
                
                # 카테고리별 데이터 준비
                categories = plot_df[cat_col].unique()[:10]
                data_to_plot = [plot_df[plot_df[cat_col] == cat][num_col].values for cat in categories]
                
                bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)
                
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                
                ax.set_xlabel(cat_korean, fontsize=12)
                ax.set_ylabel(num_korean, fontsize=12)
                ax.set_title(f'{cat_korean}별 {num_korean} 분포 (박스플롯)', fontsize=14, pad=20)
                ax.grid(axis='y', alpha=0.3)
                plt.xticks(rotation=45, ha='right')
            
            # Case 3: 둘 다 범주형 → 히트맵
            else:
                plot_df = df[[col1, col2]].dropna()
                
                # 카테고리 수 제한
                if plot_df[col1].nunique() > 10:
                    top_cats1 = plot_df[col1].value_counts().head(10).index
                    plot_df = plot_df[plot_df[col1].isin(top_cats1)]
                
                if plot_df[col2].nunique() > 10:
                    top_cats2 = plot_df[col2].value_counts().head(10).index
                    plot_df = plot_df[plot_df[col2].isin(top_cats2)]
                
                crosstab = pd.crosstab(plot_df[col1], plot_df[col2])
                
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax, 
                           cbar_kws={'label': '빈도'})
                ax.set_title(f'{korean_name1} vs {korean_name2} 교차표 (히트맵)', fontsize=14, pad=20)
                ax.set_xlabel(korean_name2, fontsize=12)
                ax.set_ylabel(korean_name1, fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
        
        plt.tight_layout()
        return fig