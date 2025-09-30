# modules/tab_preprocessing.py
from shiny import ui, render, reactive
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# --- 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parents[2]
PDF_FILE  = BASE_DIR / "reports" / "preprocessing_report.pdf"
DATA_FILE = BASE_DIR / "data" / "raw" / "train.csv"

# ===== Matplotlib 한글 폰트 =====
plt.rcParams["font.family"] = [
    "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans KR",
    "NanumGothic", "DejaVu Sans", "Arial Unicode MS", "sans-serif"
]
plt.rcParams["axes.unicode_minus"] = False

# --- 상세 컨텐츠(텍스트) ---
DETAILS = {
    "drop_heating_furnace": (
        "### 단일 칼럼 제거 — heating_furnace 열\n"
        "- A: 16,413개, B: 16,318개, 결측치: 40,880개\n"
        "- → 결측치가 A와 B의 **약 2배** 수준\n"
        "- 기본 모델에서 **변수 중요도 낮음** → **학습에서 제거**"
    ),
    "drop_molten_volume": (
        "### 단일 칼럼 제거 — molten_volume 열\n"
        "- 결측치 총 34,992개으로 **데이터의 절반 가량이 결측치**\n"
        "- 기본 모델에서 **변수 중요도 낮음** → **학습에서 제거**"
    ),
    "drop_mold_temp3": (
        "### 단일 칼럼 제거 — upper/lower_mold_temp3 열\n"
        "- 이상치가 대량 발생하며 값이 **1449.0 고정** 패턴\n"
        "- 1449.0을 센서 오류 코드로 가정 → **두 칼럼 모두 제거**"
    ),
    "drop_etc": (
        "### 단일 칼럼 제거 — 기타\n"
        "- **registration_time**: `time`+`date` 결합 정보(중복 의미) → 제거"
    ),
    "impute_molten_temp": (
        "### 결측치 처리 — molten_temp 열\n"
        "- 같은 **mold_code** 내에서 **인접 count(앞·뒤)** 평균으로 대치"
    ),
    "impute_etc": (
        "### 결측치 처리 — 기타\n"
        "- 무작위 결측엔 **중앙값** 우선, 파이프라인은 **스플릿 이후** 적용"
    ),
    "outlier_etc": (
        "### 이상치 처리 — 기타\n"
        "- IQR/z-score + 공정 지식 병행, **윈저라이징/로부스트 스케일링**"
    ),
    "row_emergency_stop": (
        "### 행 제거 — emergency_stop\n"
        "- 대량 결측 동반 → 학습 데이터에서 **행 제거**"
    ),
    "row_count_dup": (
        "### 행 제거 — count 중복\n"
        "- 동일 (mold_code, date, count) 완전 중복은 **첫 레코드만 유지**"
    ),
}

# --- 카드 하단 설명(HTML) ---  ※ 그대로 유지
EVIDENCE_DESC = {
    "drop_heating_furnace": (
        "<h4>heating_furnace 열 제외 근거</h4>"
        "<ul>"
        "<li>결측이 아닌 구간: <code>mold_code</code> 일정, <code>date</code>/<code>count</code> 연속 → 동일 furnace 연속 생산으로 해석</li>"
        "<li>결측 구간(예: index 73407, 73408): <code>mold_code</code> 8917/8722로 상이, "
        "<code>molten_volume</code> 61.0→84.0, <code>count</code> 222/219로 불연속 → 서로 다른 furnace로 보임</li>"
        "<li>비교① 73407↔73410: 모두 <code>mold_code</code> 8917, <code>molten_volume</code> 61.0→60.0, "
        "<code>count</code> 222→223으로 연속성 확인</li>"
        "<li>비교② 73408↔73411: 모두 <code>mold_code</code> 8722, <code>molten_volume</code> 84 유지, "
        "<code>count</code> 219→220으로 연속성 확인</li>"
        "<li>결론: 동일 <code>mold_code</code>이면서 <code>molten_volume</code>/<code>count</code>가 이어지면 "
        "하나의 furnace에서 연속 생산. 반대로 <b>결측(Nan) 구간은 최소 2개 이상의 상이한 집단</b>이나</li>"
        "정확히 어떤 furnace를 사용하는지, molten_volume을 다시 채울 때마다 furnace를 바꾸는지 유지하는 지 알 수 없어 구분 불가</li>"
        "<li>기본 모델에서 <b>변수 중요도도 높지 않아</b> 최종적으로 <b>heating_furnace 열 제외</b></li>"
        "</ul>"
    ),
    "drop_molten_volume": (
        "<h4>molten_volume 열 제외 근거</h4>"
        "<ul>"
        "<li>mold_code별로 차이를 보임 (mold_code 8412는 총 18468개 데이터 중 14319개가 결측, </li>"
        "<li>8573은 9596개 데이터 전부 결측, 8917은 총 22922개 데이터 중 11077개가 결측, "
        "8600과 8722는 결측치 없음)<br></li>"
        "<li>mold_code별로 나눠서 count에 따라 molten_volume 그래프를 그렸을 때 "
        "count에 따라 molten_volume이 채워지고 다시 줄어드는 양상이 보임<br></li>"
        "<li>그러나 결측치가 너무 많아서 정확한 값을 예측하기 어렵고 "
        "기본 모델에서 변수 중요도도 높지 않아 최종적으로 <b>heating_furnace 열 제외</b></li>"
        "</ul>"
    ),
    "drop_mold_temp3": (
        "<h4>upper/lower_mold_temp3 열 제외 근거</h4>"
        "<ul>"
        "<li> upper/lower_mold_temp3 열에 대해 box plot을 그리면 <b>이상치가 1449.0으로 고정</b>되어 나오고, "
        "전체 데이터 중 각각 64356개, 71651개가 이상치로 <b>정상적인 데이터가 거의 존재하지 않음</b></li>"
        "<li> 최종적으로 <b>mold_temp3 열 제외</b></li>"
        "</ul>"
    ),
    "drop_etc": (
        "<h4>registration 열 제외 근거</h4>"
        "<ul>"
        "<li><code>registration</code> 형식: 연-월-일 시:분:초</li>"
        "<li><code>time</code> 형식: 연-월-일, <code>date</code> 형식: 시:분:초</li>"
        "<li><code>time</code>열과 <code>date</code>열이 결합한 것이 <code>registration</code>열이므로 데이터가 중복되지 않도록 <b>registration 열 제외</b></li>"
        "</ul>"
    ),
    "impute_molten_temp": "<h4>설명</h4><p>설명 추가하기</p>",
    "impute_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "outlier_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "row_emergency_stop": "<h4>설명</h4><p>설명 추가하기</p>",
    "row_count_dup": "<h4>설명</h4><p>설명 추가하기</p>",
}

def panel():
    # 사이드바 버튼 가운데 정렬/한 줄 유지 + 기본 폰트
    css = ui.head_content(
        ui.tags.style("""
            .sidebar .btn {
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                white-space: nowrap;
                font-size: 13px;
                text-align: center;
                box-sizing: border-box;
            }
            body, .card, .nav-link, .btn, .form-control, .table {
                font-family: "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",
                             "Nanum Gothic","Segoe UI",Tahoma,Arial,sans-serif;
            }
        """)
    )

    return ui.nav_panel(
        "데이터 전처리 요약",
        css,
        ui.page_sidebar(
            ui.sidebar(
                ui.h4("전처리 항목"),
                ui.p("• 단일 칼럼 제거"),
                ui.input_action_button("btn_drop_heating_furnace", "heating_furnace 열"),
                ui.input_action_button("btn_drop_molten_volume", "molten_volume 열"),
                ui.input_action_button("btn_drop_mold_temp3", "upper/lower_mold_temp3 열"),
                ui.input_action_button("btn_drop_etc", "기타"),
                ui.hr(),

                ui.p("• 결측치 처리"),
                ui.input_action_button("btn_impute_molten_temp", "molten_temp 열"),
                ui.input_action_button("btn_impute_etc", "기타"),
                ui.hr(),

                ui.p("• 이상치 처리"),
                ui.input_action_button("btn_outlier_etc", "기타"),
                ui.hr(),

                ui.p("• 행 제거"),
                ui.input_action_button("btn_row_emergency_stop", "emergency_stop"),
                ui.input_action_button("btn_row_count_dup", "count 중복"),
            ),

            ui.card(
                ui.h3("내용 요약"),
                ui.output_ui("detail_panel"),
            ),
            ui.output_ui("evidence_wrap"),
            ui.card(
                ui.h3("전처리 PDF"),
                ui.p("아래 버튼을 눌러 전처리 상세 문서를 PDF로 다운로드하세요."),
                ui.download_button("download_pdf", "PDF 다운로드"),
                ui.div(ui.output_text("pdf_status"), class_="text-muted mt-2"),
            ),
        ),
    )

def server(input, output, session):
    selected = reactive.Value(None)

    # --- 버튼 이벤트 ---
    @reactive.effect
    @reactive.event(input.btn_drop_heating_furnace)
    def _sel1(): selected.set("drop_heating_furnace")

    @reactive.effect
    @reactive.event(input.btn_drop_molten_volume)
    def _sel2(): selected.set("drop_molten_volume")

    @reactive.effect
    @reactive.event(input.btn_drop_mold_temp3)
    def _sel3(): selected.set("drop_mold_temp3")

    @reactive.effect
    @reactive.event(input.btn_drop_etc)
    def _sel4(): selected.set("drop_etc")

    @reactive.effect
    @reactive.event(input.btn_impute_molten_temp)
    def _sel5(): selected.set("impute_molten_temp")

    @reactive.effect
    @reactive.event(input.btn_impute_etc)
    def _sel6(): selected.set("impute_etc")

    @reactive.effect
    @reactive.event(input.btn_outlier_etc)
    def _sel7(): selected.set("outlier_etc")

    @reactive.effect
    @reactive.event(input.btn_row_emergency_stop)
    def _sel8(): selected.set("row_emergency_stop")

    @reactive.effect
    @reactive.event(input.btn_row_count_dup)
    def _sel9(): selected.set("row_count_dup")

    # --- 데이터 로드(원본 문자열 보존) ---
    @reactive.calc
    def _raw_df() -> pd.DataFrame:
        return pd.read_csv(DATA_FILE, dtype=str, low_memory=False)

    # --- 그래프용 DF (molten_volume) ---
    @reactive.calc
    def _mv_df() -> pd.DataFrame:
        df = pd.read_csv(DATA_FILE, low_memory=False)
        need = {"mold_code", "molten_volume", "count", "passorfail"}
        if not need.issubset(df.columns):
            return pd.DataFrame(columns=list(need))
        df["molten_volume"] = pd.to_numeric(df["molten_volume"], errors="coerce")
        df["count"] = pd.to_numeric(df["count"], errors="coerce")
        df["passorfail"] = df["passorfail"].astype(str)  # '0.0' / '1.0'
        return df.dropna(subset=["molten_volume", "count"])

    # --- mold_temp3용 DF (숫자형, wide 형태) ---
    @reactive.calc
    def _mt3_df() -> pd.DataFrame:
        df = pd.read_csv(DATA_FILE, low_memory=False)
        cols = ["upper_mold_temp3", "lower_mold_temp3"]
        if not set(cols).issubset(df.columns):
            return pd.DataFrame(columns=cols)
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df[cols]

    # --- 상세 설명 ---
    @output
    @render.ui
    def detail_panel():
        key = selected.get()
        if not key:
            return ui.markdown("왼쪽 사이드바에서 항목을 선택하세요.")
        return ui.markdown(DETAILS.get(key, "내용이 없습니다."))

    # --- 데이터 카드(필요할 때만 표시) ---
    @output
    @render.ui
    def evidence_wrap():
        key = selected.get()
        SHOW_CARD = {"drop_heating_furnace", "drop_etc", "drop_molten_volume", "drop_mold_temp3"}
        if key in SHOW_CARD:
            return ui.card(
                ui.h3("데이터 확인하기"),
                ui.output_ui("evidence_panel"),
                class_="shadow"
            )
        return None

    # --- 카드 내부 구성 ---
    @output
    @render.ui
    def evidence_panel():
        key = selected.get()
        if not key:
            return ui.markdown("왼쪽에서 항목을 선택하면 데이터와 설명이 표시됩니다.")

        blocks = []
        if key == "drop_heating_furnace":
            blocks += [
                ui.h4("특정 인덱스 구간 (73406–73418)"),
                ui.output_data_frame("hf_slice_df"),
                ui.hr(),
            ]
        if key == "drop_etc":
            blocks += [
                ui.h4("상위 5행 미리보기 (registration, time, date)"),
                ui.output_data_frame("etc_head_df"),
                ui.hr(),
            ]
        if key == "drop_molten_volume":
            blocks += [
                ui.h4("mold_code별 molten_volume 산점도 (빈도 상위 4개, 각 300 샘플)"),
                ui.output_plot("mv_plot", width="100%", height="1100px"),
                ui.hr(),
            ]
        if key == "drop_mold_temp3":
            blocks += [
                ui.h4("upper/lower_mold_temp3 Histogram"),
                ui.output_plot("mt3_hist", width="100%", height="700px"),
                ui.hr(),
            ]

        blocks += [
            ui.h5("설명"),
            ui.HTML(EVIDENCE_DESC.get(key, "<p>설명 추가하기</p>")),
        ]
        return ui.TagList(*blocks)

    # --- heating_furnace 표 ---
    @output
    @render.data_frame
    def hf_slice_df():
        if selected.get() != "drop_heating_furnace":
            return pd.DataFrame()

        df = _raw_df()
        required = ["heating_furnace", "mold_code", "time", "date", "molten_volume", "count"]
        if not set(required).issubset(df.columns):
            return pd.DataFrame()

        start, end = 73406, 73418
        subset = df.loc[start:end, required].copy() if df.index.max() >= end else df.loc[:, required].head(12).copy()

        display = pd.DataFrame({
            "index":           subset.index,
            "heating_furnace": subset["heating_furnace"].astype(str),
            "mold_code":       subset["mold_code"].astype(str),
            "time":            subset["time"].astype(str),
            "date":            subset["date"].astype(str),
            "molten_volume":   subset["molten_volume"].astype(str),
            "count":           subset["count"].astype(str),
        })
        display = display.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return display

    # --- drop_etc: registration/time/date 5행만 ---
    @output
    @render.data_frame
    def etc_head_df():
        if selected.get() != "drop_etc":
            return pd.DataFrame()

        df = _raw_df()
        reg_col = "registration" if "registration" in df.columns else (
            "registration_time" if "registration_time" in df.columns else None
        )

        cols = []
        if reg_col: cols.append(reg_col)
        if "time" in df.columns: cols.append("time")
        if "date" in df.columns: cols.append("date")
        if not cols:
            return pd.DataFrame()

        display = df[cols].head(5).copy()
        if reg_col and reg_col != "registration":
            display = display.rename(columns={reg_col: "registration"})
        display = display.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return display

    # --- molten_volume: mold_code별 산점도 (이상치 제거 + 큰 그림 + 간격 조정) ---
    @output
    @render.plot
    def mv_plot():
        if selected.get() != "drop_molten_volume":
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "molten_volume을 선택하세요", ha="center", va="center")
            plt.axis("off")
            return fig

        df = _mv_df()
        if df.empty:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "필수 칼럼이 없거나 데이터가 비어있습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        # 이상치 제거: molten_volume > 2500 제외
        df = df[df["molten_volume"] <= 2500]

        top_molds = df["mold_code"].value_counts().index.tolist()[:4]
        n = len(top_molds)

        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(18, 4.8 * n))
        fig.subplots_adjust(hspace=0.65, top=0.95, bottom=0.08)

        if n == 1:
            axes = [axes]

        hue_order = ["0.0", "1.0"]
        palette_map = {"0.0": "blue", "1.0": "red"}

        for i, mold in enumerate(top_molds):
            ax = axes[i]
            mold_df = df[df["mold_code"] == mold].head(300)
            sns.scatterplot(
                data=mold_df,
                x="count", y="molten_volume",
                hue="passorfail", hue_order=hue_order,
                palette=palette_map, alpha=0.7, s=24, ax=ax, legend=False
            )
            ax.set_title(f"Mold Code: {mold}", pad=6, fontsize=15)
            ax.set_xlabel("Count")
            ax.set_ylabel("Molten Volume")
            ax.margins(x=0.04, y=0.12)
            ax.grid(True, alpha=0.25)

        custom_legend = [
            Line2D([0], [0], marker='o', color='blue', linestyle='', label='양품 (0.0)', markersize=8),
            Line2D([0], [0], marker='o', color='red',  linestyle='', label='불량 (1.0)', markersize=8),
        ]
        axes[-1].legend(handles=custom_legend, title="품질", loc="upper right")
        return fig

    # --- mold_temp3: Histogram ---
    # --- mold_temp3: Histogram (upper/lower 모두 0~1500 구간으로 통일) ---
    @output
    @render.plot
    def mt3_hist():
        if selected.get() != "drop_mold_temp3":
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "upper/lower_mold_temp3를 선택하세요", ha="center", va="center")
            plt.axis("off")
            return fig

        df = _mt3_df()  # 숫자형으로 변환된 upper/lower_mold_temp3를 반환하는 기존 헬퍼
        if df.empty:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "필수 칼럼이 없거나 데이터가 비어있습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
        fig.subplots_adjust(hspace=0.45, top=0.92, bottom=0.1, left=0.07, right=0.98)

        cols = ["upper_mold_temp3", "lower_mold_temp3"]
        name_map = {"upper_mold_temp3": "Upper Mold Temp3", "lower_mold_temp3": "Lower Mold Temp3"}

        for i, col in enumerate(cols):
            ax = axes[i]

            # 전체(원본)과, 시각화용(0~1500) 시리즈 분리
            s_all  = df[col].dropna()
            s_plot = s_all[(s_all >= 0) & (s_all <= 1500)]

            # 히스토그램
            ax.hist(s_plot, bins=120, color="#4c78a8", alpha=0.85, edgecolor="white")

            # 1449.0 표시 및 개수 주석(개수는 전체 기준)
            ax.axvline(1449.0, color="red", linestyle="--", linewidth=1.8)
            cnt_1449 = int((s_all == 1449.0).sum())
            ax.set_title(
                f"{name_map[col]} — n={len(s_plot):,} (x≤1500), 1449.0 개수={cnt_1449:,}",
                pad=8
            )
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Count")
            ax.set_xlim(0, 1500)
            ax.grid(True, axis="y", alpha=0.25)
            ymax = ax.get_ylim()[1]
            ax.text(1449.0, ymax * 0.9, f"{cnt_1449:,}", color="red",
                    ha="left", va="center", fontsize=10, rotation=90)

        return fig



    # --- PDF 상태 안내 ---
    @output
    @render.text
    def pdf_status():
        return f"파일 위치: {PDF_FILE}" if PDF_FILE.exists() \
            else "주의: PDF 파일이 아직 없습니다. reports/preprocessing_report.pdf 경로에 파일을 생성해 주세요."

    # --- PDF 다운로드 ---
    @render.download(filename="preprocessing_report.pdf")
    def download_pdf():
        if PDF_FILE.exists():
            with open(PDF_FILE, "rb") as f:
                chunk = f.read(8192)
                while chunk:
                    yield chunk
                    chunk = f.read(8192)
        else:
            minimal_pdf = (
                b"%PDF-1.4\n"
                b"1 0 obj<<>>endobj\n"
                b"2 0 obj<< /Type /Catalog /Pages 3 0 R >>endobj\n"
                b"3 0 obj<< /Type /Pages /Kids [4 0 R] /Count 1 >>endobj\n"
                b"4 0 obj<< /Type /Page /Parent 3 0 R /MediaBox [0 0 300 144] /Contents 5 0 R >>endobj\n"
                b"5 0 obj<< /Length 62 >>stream\n"
                b"BT /F1 12 Tf 24 100 Td (Preprocessing report missing) Tj ET\n"
                b"endstream endobj\n"
                b"6 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n"
                b"xref\n0 7\n0000000000 65535 f \n0000000010 00000 n \n0000000050 00000 n \n0000000102 00000 n \n0000000162 00000 n \n0000000258 00000 n \n0000000358 00000 n \n"
                b"trailer<< /Size 7 /Root 2 0 R >>\nstartxref\n408\n%%EOF\n"
            )
            yield minimal_pdf
