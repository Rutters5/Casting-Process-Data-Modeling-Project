from shiny import ui, render, reactive
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# =========================
# 경로 설정 및 데이터 로드
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)
df['mold_code'] = df['mold_code'].astype(str)
df['registration_time'] = pd.to_datetime(df['registration_time'])

# passorfail 처리 (숫자형으로 유지)
if 'passorfail' in df.columns:
    df['passorfail'] = pd.to_numeric(df['passorfail'], errors='coerce')

# 날짜 정보 생성 (한 번만 계산)
df['date'] = df['registration_time'].dt.date
date_choices = sorted(df['date'].unique(), reverse=True)
DATE_CHOICES = {str(d): str(d) for d in date_choices}

# passorfail 존재 여부 미리 확인
HAS_PASSORFAIL = 'passorfail' in df.columns

# =========================
# 변수 정의
# =========================
NUMERIC_VARS = {
    'count': '일자별 생산 번호', 'molten_temp': '용탕 온도',
    'facility_operation_cycleTime': '설비 작동 사이클 시간',
    'production_cycletime': '제품 생산 사이클 시간',
    'low_section_speed': '저속 구간 속도', 'high_section_speed': '고속 구간 속도',
    'molten_volume': '용탕량', 'cast_pressure': '주조 압력',
    'biscuit_thickness': '비스켓 두께', 'upper_mold_temp1': '상금형 온도1',
    'upper_mold_temp2': '상금형 온도2', 'upper_mold_temp3': '상금형 온도3',
    'lower_mold_temp1': '하금형 온도1', 'lower_mold_temp2': '하금형 온도2',
    'lower_mold_temp3': '하금형 온도3', 'sleeve_temperature': '슬리브 온도',
    'physical_strength': '형체력', 'Coolant_temperature': '냉각수 온도',
    'EMS_operation_time': '전자교반 가동 시간'
}

CATEGORICAL_VARS = {
    'working': '가동 여부', 'passorfail': '양품/불량 판정',
    'tryshot_signal': '사탕 신호', 'heating_furnace': '가열로 구분',
    'mold_code': '금형 코드'
}

ALL_VARS = {**{k: f"{v}({k})-수치형" for k, v in NUMERIC_VARS.items()},
            **{k: f"{v}({k})-범주형" for k, v in CATEGORICAL_VARS.items()}}

# =========================
# 유틸리티 함수
# =========================
def get_korean_name(col):
    return {**NUMERIC_VARS, **CATEGORICAL_VARS}.get(col, col)

def is_numeric_var(col):
    return col in NUMERIC_VARS

def setup_matplotlib():
    plt.rcParams.update({
        'figure.dpi': 80,
        'font.family': 'Malgun Gothic',
        'axes.unicode_minus': False
    })

def create_empty_plot(message, figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=16)
    ax.axis('off')
    return fig

# =========================
# CSS 및 JavaScript
# =========================
CUSTOM_CSS_JS = """
<style>
body { font-family: -apple-system, sans-serif; background-color: #f5f7fa; }
.floating-panel { 
    position: fixed; 
    bottom: 20px; 
    left: 20px; 
    width: 350px; 
    background: white; 
    border-radius: 16px; 
    box-shadow: 0 4px 16px rgba(0,0,0,0.15); 
    z-index: 1000; 
    max-height: 80vh; 
    overflow: hidden; 
}
.floating-panel-header { 
    background: #2A2D30; 
    color: white; 
    padding: 20px 24px; 
    border-radius: 16px 16px 0 0; 
    font-weight: 600; 
    font-size: 16px; 
    cursor: move; 
    user-select: none; 
}
.floating-panel-content { 
    padding: 20px 24px; 
    background: white; 
    border-radius: 0 0 16px 16px; 
    max-height: calc(80vh - 60px); 
    overflow-y: auto; 
}
.floating-panel-content p { font-size: 0.85em; color: #666; margin-bottom: 15px; }
.floating-panel-content hr { margin: 15px 0; border: none; border-top: 1px solid #eee; }
#toggle-button { 
    position: fixed; 
    bottom: 20px; 
    left: 390px; 
    width: 50px; 
    height: 50px; 
    background: #2A2D30; 
    border-radius: 50%; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    cursor: pointer; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.2); 
    z-index: 1000; 
    transition: all 0.2s; 
}
#toggle-button:hover { transform: scale(1.1); background: #1f2428; }
#toggle-button svg { transition: transform 0.3s ease; }
#toggle-button.panel-hidden svg { transform: rotate(180deg); }
.main-content { padding: 20px; }
.hidden { display: none !important; }

/* 아코디언 스타일 */
.accordion-section { 
    background: white; 
    border-radius: 16px; 
    margin-bottom: 20px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
    overflow: hidden;
}
.accordion-header { 
    background: #2A2D30; 
    color: white; 
    padding: 20px 28px; 
    cursor: pointer; 
    display: flex; 
    justify-content: space-between; 
    border: none; 
    width: 100%; 
    text-align: left; 
    font-size: 16px; 
    font-weight: 600; 
    border-radius: 16px 16px 0 0;
}
.accordion-header:hover { 
    background-color: #1f2428;
}
.accordion-content { 
    padding: 24px 28px; 
    background: #ffffff; 
    border-radius: 0 0 16px 16px;
}
</style>
<script>
(function() {
    'use strict';
    
    var dragState = {
        isDragging: false,
        startX: 0,
        startY: 0,
        offsetX: 0,
        offsetY: 0
    };
    
    function toggleAccordion(id) {
        var content = document.getElementById(id);
        if (content) {
            if (content.style.display === "none") {
                content.style.display = "block";
            } else {
                content.style.display = "none";
            }
        }
    }
    
    // 전역으로 노출
    window.toggleAccordion = toggleAccordion;
    
    function initDragAndToggle() {
        var panel = document.querySelector('.floating-panel');
        var header = panel ? panel.querySelector('.floating-panel-header') : null;
        var toggleBtn = document.getElementById('toggle-button');
        
        if (header && !header.dataset.dragInit) {
            header.dataset.dragInit = 'true';
            
            header.onmousedown = function(e) {
                dragState.isDragging = true;
                dragState.startX = e.clientX - dragState.offsetX;
                dragState.startY = e.clientY - dragState.offsetY;
                e.preventDefault();
            };
        }
        
        if (!document.body.dataset.mouseMoveInit) {
            document.body.dataset.mouseMoveInit = 'true';
            
            document.onmousemove = function(e) {
                if (dragState.isDragging && panel) {
                    dragState.offsetX = e.clientX - dragState.startX;
                    dragState.offsetY = e.clientY - dragState.startY;
                    panel.style.transform = 'translate(' + dragState.offsetX + 'px, ' + dragState.offsetY + 'px)';
                }
            };
            
            document.onmouseup = function() {
                dragState.isDragging = false;
            };
        }
        
        if (toggleBtn && !toggleBtn.dataset.toggleInit) {
            toggleBtn.dataset.toggleInit = 'true';
            
            toggleBtn.onclick = function() {
                if (panel) {
                    var isHidden = panel.classList.contains('hidden');
                    if (isHidden) {
                        panel.classList.remove('hidden');
                        toggleBtn.classList.remove('panel-hidden');
                    } else {
                        panel.classList.add('hidden');
                        toggleBtn.classList.add('panel-hidden');
                    }
                }
            };
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDragAndToggle);
    } else {
        initDragAndToggle();
    }
    
    setTimeout(initDragAndToggle, 500);
    setTimeout(initDragAndToggle, 1000);
    setTimeout(initDragAndToggle, 2000);
})();
</script>
"""

# =========================
# UI 정의
# =========================
def panel():
    return ui.nav_panel(
        "EDA 분석",
        ui.HTML(CUSTOM_CSS_JS),
        ui.HTML('''
            <div id="toggle-button">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" 
                     stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M15 18l-6-6 6-6"/>
                </svg>
            </div>
        '''),
        ui.div(
            ui.div("변수 선택", class_="floating-panel-header"),
            ui.div(
                ui.p("두 변수가 같으면 히스토그램, 다르면 산점도/박스플롯/히트맵이 표시됩니다."),
                ui.input_selectize("var1", "변수 1", choices=ALL_VARS, selected=list(ALL_VARS.keys())[0]),
                ui.input_selectize("var2", "변수 2", choices=ALL_VARS, selected=list(ALL_VARS.keys())[0]),
                ui.hr(),
                ui.input_checkbox("timeseries_mode", "시계열 모드"),
                ui.p("시계열 모드: x축=시간, y축=변수 1", 
                     style="font-size: 0.8em; color: #666; margin-top: -10px;"),
                ui.output_ui("date_selector_ui"),
                ui.hr(),
                ui.output_text("selection_info"),
                class_="floating-panel-content"
            ),
            class_="floating-panel"
        ),
        # 메인 콘텐츠 영역
        ui.div(
            # 검색창 스타일 제목
            ui.div(
                ui.HTML('<i class="fa-solid fa-magnifying-glass" style="color: #9ca3af; margin-right: 14px; font-size: 16px;"></i>'),
                ui.span("EDA 요약", style="color: #6b7280; font-size: 15px; font-weight: 400;"),
                style="""
                    background: white;
                    border: 1px solid #d1d5db;
                    border-radius: 24px;
                    padding: 14px 20px;
                    display: flex;
                    align-items: center;
                    flex: 1;
                    max-width: 400px;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                    margin-left: 20px;
                    margin-top: 20px;
                    margin-bottom: 20px;
                """
            ),
            
            # 아코디언 섹션
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("탐색적 데이터 분석 (EDA)", style="font-size: 16px;"),
                        ui.span("▼", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAccordion('eda_content')",
                    class_="accordion-header"
                ),
                ui.div(
                    ui.layout_columns(
                        ui.card(ui.card_header("데이터셋 정보 및 통계"),
                               ui.output_text("data_info"), ui.hr(), ui.output_data_frame("stats")),
                        ui.card(ui.card_header("시각화 결과"),
                               ui.output_plot("plots", width="100%", height="800px")),
                        col_widths=[4, 8]
                    ),
                    id="eda_content",
                    class_="accordion-content",
                    style="display: block;"
                ),
                class_="accordion-section",
                style="max-width: 1400px; margin-left: 20px; margin-right: 20px;"
            ),
            
            class_="main-content"
        )
    )

# =========================
# 서버 로직
# =========================
def server(input, output, session):
    
    @output
    @render.ui
    def date_selector_ui():
        if input.timeseries_mode():
            return ui.div(
                ui.input_selectize("selected_date", "날짜 선택", 
                                  choices=DATE_CHOICES, selected=list(DATE_CHOICES.keys())[0]),
                ui.p("선택한 날짜의 데이터만 표시됩니다", 
                     style="font-size: 0.8em; color: #666; margin-top: -10px;"),
                style="margin-top: 10px;"
            )
        return ui.div()
    
    @reactive.Calc
    def selected_vars():
        v1, v2 = input.var1(), input.var2()
        if input.timeseries_mode():
            return [v1] if v1 else []
        if not v1 and not v2:
            return []
        elif v1 == v2:
            return [v1]
        elif v1 and v2:
            return [v1, v2]
        return [v1] if v1 else [v2]
    
    # 시계열 모드용 필터링된 데이터 (캐싱)
    @reactive.Calc
    def get_timeseries_data():
        if not input.timeseries_mode():
            return None
        
        v1 = input.var1()
        if not v1:
            return None
            
        try:
            selected_date = pd.to_datetime(input.selected_date()).date()
        except:
            return None
        
        # 필요한 컬럼만 선택
        cols_needed = ['registration_time', v1]
        if HAS_PASSORFAIL:
            cols_needed.append('passorfail')
        
        # 날짜 필터링
        filtered = df[df['date'] == selected_date][cols_needed].copy()
        filtered = filtered.dropna(subset=[v1])
        
        # 데이터가 많으면 미리 샘플링
        if len(filtered) > 2000:
            filtered = filtered.sample(2000, random_state=42)
        
        return filtered.sort_values('registration_time')
    
    @output
    @render.text
    def selection_info():
        v1, v2, ts = input.var1(), input.var2(), input.timeseries_mode()
        if ts:
            if not v1:
                return "시계열 모드\n변수 1을 선택해주세요"
            viz_type = "산점도 (시간 단위)" if is_numeric_var(v1) else "막대그래프 (시간별)"
            return f"시계열 모드\nx축: 시간, y축: 변수 1\n{viz_type} 표시"
        if not v1 or not v2:
            return "두 변수를 모두 선택해주세요"
        elif v1 == v2:
            return "동일 변수 선택\n히스토그램 표시"
        n1, n2 = is_numeric_var(v1), is_numeric_var(v2)
        viz_type = "산점도" if n1 and n2 else ("박스플롯" if n1 or n2 else "히트맵")
        return f"두 변수 선택\n{viz_type} 표시"
    
    @output
    @render.text
    def data_info():
        info = f"전체 데이터 수: {len(df):,}개"
        if input.timeseries_mode():
            ts_data = get_timeseries_data()
            if ts_data is not None:
                info += f"\n선택된 날짜 데이터: {len(ts_data):,}개"
        
        selected = selected_vars()
        if selected:
            info += "\n\n결측치 수:"
            for col in selected:
                missing = df[col].isnull().sum()
                info += f"\n• {get_korean_name(col)}: {missing:,}개 ({missing/len(df)*100:.1f}%)"
        return info
    
    @output
    @render.data_frame
    def stats():
        cols = selected_vars()
        if not cols:
            return pd.DataFrame()
        
        # 시계열 모드에서는 필터링된 데이터 사용
        if input.timeseries_mode():
            ts_data = get_timeseries_data()
            if ts_data is None or len(ts_data) == 0:
                return pd.DataFrame()
            data_source = ts_data
        else:
            data_source = df
        
        data = data_source[cols].copy()
        data.columns = [get_korean_name(c) for c in cols]
        
        num_data = data.select_dtypes(include=[np.number])
        if not num_data.empty:
            stats_df = num_data.describe().round(3).reset_index().rename(columns={'index': '통계량'})
            mapping = {
                'count': 'count (개수)', 'mean': 'mean (평균)', 'std': 'std (표준편차)',
                'min': 'min (최솟값)', '25%': '25% (1사분위)', '50%': '50% (중앙값)',
                '75%': '75% (3사분위)', 'max': 'max (최댓값)'
            }
            stats_df['통계량'] = stats_df['통계량'].map(mapping)
            return stats_df
        
        cat_data = data.select_dtypes(include=['object'])
        if not cat_data.empty:
            return pd.DataFrame([{
                '변수': col, '고유값 개수': cat_data[col].nunique(),
                '최빈값': cat_data[col].mode()[0] if len(cat_data[col].mode()) > 0 else 'N/A',
                '최빈값 빈도': cat_data[col].value_counts().iloc[0] if len(cat_data[col]) > 0 else 0
            } for col in cat_data.columns])
        
        return pd.DataFrame()
    
    @output
    @render.plot
    def plots():
        setup_matplotlib()
        v1, v2, ts = input.var1(), input.var2(), input.timeseries_mode()
        
        # 시계열 모드
        if ts:
            if not v1:
                return create_empty_plot('변수 1을 선택해주세요')
            
            # 캐싱된 데이터 사용
            plot_df = get_timeseries_data()
            
            if plot_df is None:
                return create_empty_plot('날짜를 선택해주세요')
            
            if len(plot_df) == 0:
                return create_empty_plot('선택한 날짜에 데이터가 없습니다')
            
            try:
                date_label = str(input.selected_date())
            except:
                date_label = "전체"
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            if is_numeric_var(v1):
                title = f"{get_korean_name(v1)} - {date_label} ({len(plot_df):,}개)"
                
                if HAS_PASSORFAIL and 'passorfail' in plot_df.columns:
                    pass_data = plot_df[plot_df['passorfail'] == 0]
                    fail_data = plot_df[plot_df['passorfail'] == 1]
                    
                    if not pass_data.empty:
                        ax.scatter(pass_data['registration_time'], pass_data[v1], 
                                 alpha=0.6, s=10, color='#28a745', label='양품', rasterized=True)
                    if not fail_data.empty:
                        ax.scatter(fail_data['registration_time'], fail_data[v1], 
                                 alpha=0.6, s=10, color='#dc3545', label='불량', rasterized=True)
                    if not pass_data.empty or not fail_data.empty:
                        ax.legend(loc='upper right', fontsize=9)
                else:
                    ax.scatter(plot_df['registration_time'], plot_df[v1], 
                             alpha=0.6, s=10, color='steelblue', rasterized=True)
                
                ax.set_ylabel(get_korean_name(v1), fontsize=11)
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.xticks(rotation=45, ha='right', fontsize=9)
                ax.set_title(title, fontsize=12, pad=15)
            
            else:
                plot_df['hour'] = plot_df['registration_time'].dt.hour
                counts = plot_df.groupby(['hour', v1]).size().unstack(fill_value=0)
                counts.plot(kind='bar', stacked=True, ax=ax, alpha=0.8, width=0.85)
                title = f"{get_korean_name(v1)} - {date_label} (시간별 집계)"
                ax.set_title(title, fontsize=12, pad=15)
                ax.set_ylabel('빈도', fontsize=11)
                ax.set_xlabel('시간', fontsize=11)
                ax.legend(title=get_korean_name(v1), bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.set_xticklabels([f"{int(h)}시" for h in counts.index], rotation=45, ha='right', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            return fig
        
        # 일반 모드
        if not v1 or not v2:
            return create_empty_plot('두 변수를 모두 선택해주세요', figsize=(6, 3.5))
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        
        if v1 == v2:
            if is_numeric_var(v1):
                if HAS_PASSORFAIL:
                    pass_data = df[df['passorfail'] == 0][v1].dropna()
                    fail_data = df[df['passorfail'] == 1][v1].dropna()
                    
                    if not pass_data.empty or not fail_data.empty:
                        all_data = pd.concat([pass_data, fail_data])
                        bins = np.histogram_bin_edges(all_data, bins=30)
                        
                        data_list = []
                        colors_list = []
                        labels_list = []
                        
                        if not pass_data.empty:
                            data_list.append(pass_data)
                            colors_list.append('#28a745')
                            labels_list.append('양품')
                        if not fail_data.empty:
                            data_list.append(fail_data)
                            colors_list.append('#dc3545')
                            labels_list.append('불량')
                        
                        ax.hist(data_list, bins=bins, alpha=0.7, 
                               edgecolor='black', color=colors_list,
                               label=labels_list, stacked=True)
                        ax.legend(loc='upper right')
                else:
                    data_clean = df[v1].dropna()
                    ax.hist(data_clean, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
                
                ax.set_xlabel(get_korean_name(v1), fontsize=10)
                ax.set_ylabel('빈도', fontsize=10)
            else:
                counts = df[v1].fillna('NaN').value_counts().head(15)
                bars = ax.bar(range(len(counts)), counts.values, alpha=0.7, color='coral')
                ax.set_xticks(range(len(counts)))
                ax.set_xticklabels(counts.index, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('빈도', fontsize=10)
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height):,}', ha='center', va='bottom', fontsize=8)
            ax.set_title(f"{get_korean_name(v1)} 분포", fontsize=11, pad=15)
            ax.grid(axis='y', alpha=0.3)
        
        else:
            n1, n2 = is_numeric_var(v1), is_numeric_var(v2)
            
            if n1 and n2:
                if HAS_PASSORFAIL:
                    plot_df = df[[v1, v2, 'passorfail']].dropna()
                else:
                    plot_df = df[[v1, v2]].dropna()
                
                if len(plot_df) > 10000:
                    plot_df = plot_df.sample(10000)
                
                if HAS_PASSORFAIL:
                    pass_data = plot_df[plot_df['passorfail'] == 0]
                    fail_data = plot_df[plot_df['passorfail'] == 1]
                    
                    if not pass_data.empty:
                        ax.scatter(pass_data[v1], pass_data[v2], 
                                 alpha=0.5, s=10, color='#28a745', label='양품')
                    if not fail_data.empty:
                        ax.scatter(fail_data[v1], fail_data[v2], 
                                 alpha=0.5, s=10, color='#dc3545', label='불량')
                    if not pass_data.empty or not fail_data.empty:
                        ax.legend(loc='upper right')
                    
                    corr = plot_df[v1].corr(plot_df[v2])
                else:
                    ax.scatter(plot_df[v1], plot_df[v2], alpha=0.5, s=10, color='steelblue')
                    corr = plot_df[v1].corr(plot_df[v2])
                
                ax.text(0.05, 0.95, f'상관계수: {corr:.3f}', transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel(get_korean_name(v1), fontsize=10)
                ax.set_ylabel(get_korean_name(v2), fontsize=10)
                ax.set_title(f"{get_korean_name(v1)} vs {get_korean_name(v2)}", fontsize=11, pad=15)
            
            elif n1 or n2:
                num_col, cat_col = (v1, v2) if n1 else (v2, v1)
                num_name, cat_name = (get_korean_name(v1), get_korean_name(v2)) if n1 else (get_korean_name(v2), get_korean_name(v1))
                plot_df = df[[cat_col, num_col]].dropna()
                categories = plot_df[cat_col].unique()[:10]
                data_to_plot = [plot_df[plot_df[cat_col] == c][num_col].values for c in categories]
                bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                    patch.set_alpha(0.7)
                ax.set_xlabel(cat_name, fontsize=10)
                ax.set_ylabel(num_name, fontsize=10)
                ax.set_title(f"{cat_name}별 {num_name} 분포", fontsize=11, pad=15)
                plt.xticks(rotation=45, ha='right', fontsize=8)
            
            else:
                plot_df = df[[v1, v2]].dropna()
                for col in [v1, v2]:
                    if plot_df[col].nunique() > 10:
                        top_cats = plot_df[col].value_counts().head(10).index
                        plot_df = plot_df[plot_df[col].isin(top_cats)]
                crosstab = pd.crosstab(plot_df[v1], plot_df[v2])
                sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax,
                           cbar_kws={'label': '빈도'}, annot_kws={'size': 8})
                ax.set_xlabel(get_korean_name(v2), fontsize=10)
                ax.set_ylabel(get_korean_name(v1), fontsize=10)
                ax.set_title(f"{get_korean_name(v1)} vs {get_korean_name(v2)}", fontsize=11, pad=15)
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.yticks(rotation=0, fontsize=8)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig