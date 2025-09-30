from shiny import ui, render, reactive
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# Shiny 버전에 따라 ui.nav 또는 ui.nav_panel을 사용
NAV = getattr(ui, "nav", ui.nav_panel)

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

# --- 카드 하단 설명(HTML) ---
EVIDENCE_DESC = {
    "drop_heating_furnace": """
<h4>heating_furnace 열 제외 근거</h4>
<ul>
  <li>결측이 아닌 구간: <code>mold_code</code> 일정, <code>date</code>/<code>count</code> 연속 → 동일 furnace 연속 생산으로 해석</li>
  <li>결측 구간(예: index 73407, 73408): <code>mold_code</code> 8917/8722로 상이,
      <code>molten_volume</code> 61.0→84.0, <code>count</code> 222/219로 불연속 → 서로 다른 furnace로 보임</li>
  <li>비교① 73407↔73410: 모두 <code>mold_code</code> 8917, <code>molten_volume</code> 61.0→60.0,
      <code>count</code> 222→223으로 연속성 확인</li>
  <li>비교② 73408↔73411: 모두 <code>mold_code</code> 8722, <code>molten_volume</code> 84 유지,
      <code>count</code> 219→220으로 연속성 확인</li>
  <li>결론: 동일 <code>mold_code</code>이면서 <code>molten_volume</code>/<code>count</code>가 이어지면 하나의 furnace에서 연속 생산.
      반대로 <b>결측(NaN) 구간</b>은 최소 2개 이상의 상이한 집단일 가능성이 큼. 또한 어떤 furnace를 사용하는지,
      molten_volume을 다시 채울 때마다 furnace를 바꾸는지 등은 구분이 어려움</li>
  <li>모델 관점: 변수 중요도도 높지 않아 최종적으로 <b>heating_furnace 열 제외</b></li>
</ul>
""",
    "drop_molten_volume":   ("<h4>…생략…</h4>"),
    "drop_mold_temp3":      ("<h4>…생략…</h4>"),
    "drop_etc":             ("<h4>…생략…</h4>"),
    "impute_molten_temp": "<h4>설명</h4><p>설명 추가하기</p>",
    "impute_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "outlier_etc": "<h4>설명</h4><p>설명 추가하기</p>",
    "row_emergency_stop": "<h4>설명</h4><p>설명 추가하기</p>",
    "row_count_dup": "<h4>설명</h4><p>설명 추가하기</p>",
}


# =========================
# UI
# =========================
def panel():
    css = ui.head_content(
        ui.tags.style("""
            .card .nav { margin-bottom: 0.75rem; }
            .left-col { padding-right: 1rem; border-right: 1px solid #eee; }
            .right-col { padding-left: 1rem; }
            .muted { color: #6c757d; }
            /* 접기 박스 */
            details.details-box { margin-top: .75rem; }
            details.details-box > summary { cursor: pointer; font-weight: 600; }
        """)
    )

    # 공통 레이아웃: 각 카드마다 탭 + (왼쪽 텍스트 / 오른쪽 시각화)
    def two_col(left, right):
        return ui.row(
            ui.column(6, left, class_="left-col"),
            ui.column(6, right, class_="right-col"),
        )

    return ui.nav_panel(
        "데이터 전처리 요약",
        css,

        # 1) 단일 칼럼 제거
        ui.card(
            ui.card_header("단일 칼럼 제거"),
            ui.navset_tab(
                NAV(
                    "heating_furnace 열",
                    two_col(
                        ui.markdown(DETAILS["drop_heating_furnace"]),
                        ui.output_plot("hf_hist", width="100%", height="380px"),
                    ),
                    ui.tags.details(
                        ui.tags.summary("상세 설명 열기/닫기"),
                        ui.div(
                            ui.h5("특정 인덱스 구간 (73406–73418)"),
                            ui.output_data_frame("hf_slice_df"),
                            style="margin: .5rem 0 1rem 0;"
                        ),
                        ui.HTML(EVIDENCE_DESC["drop_heating_furnace"]),
                        class_="details-box"
                    ),
                ),
                NAV(
                    "molten_volume 열",
                    two_col(
                        ui.markdown(DETAILS["drop_molten_volume"]),
                        ui.output_plot("mv_pie", width="100%", height="380px"),
                    ),
                ),
                NAV(
                    "upper/lower_mold_temp3 열",
                    two_col(
                        ui.markdown(DETAILS["drop_mold_temp3"]),
                        ui.output_plot("mt3_hist", width="100%", height="420px"),
                    ),
                ),
                NAV(
                    "registration_time 열",
                    two_col(
                        ui.markdown(DETAILS["drop_etc"]),
                        ui.output_data_frame("reg_head_df"),
                    ),
                ),
            ),
        ),

        # 2) 결측치 처리
        ui.card(
            ui.card_header("결측치 처리"),
            ui.navset_tab(
                NAV(
                    "molten_temp 열",
                    two_col(
                        ui.markdown(DETAILS["impute_molten_temp"]),
                        ui.p("그래프/표 없음", class_="muted"),
                    ),
                ),
                NAV(
                    "기타",
                    two_col(
                        ui.markdown(DETAILS["impute_etc"]),
                        ui.p("그래프/표 없음", class_="muted"),
                    ),
                ),
            ),
        ),

        # 3) 이상치 처리
        ui.card(
            ui.card_header("이상치 처리"),
            ui.navset_tab(
                NAV(
                    "기타",
                    two_col(
                        ui.markdown(DETAILS["outlier_etc"]),
                        ui.p("그래프/표 없음", class_="muted"),
                    ),
                ),
            ),
        ),

        # 4) 행 제거
        ui.card(
            ui.card_header("행 제거"),
            ui.navset_tab(
                NAV(
                    "emergency_stop",
                    two_col(
                        ui.markdown(DETAILS["row_emergency_stop"]),
                        ui.p("그래프/표 없음", class_="muted"),
                    ),
                ),
                NAV(
                    "count 중복",
                    two_col(
                        ui.markdown(DETAILS["row_count_dup"]),
                        ui.p("그래프/표 없음", class_="muted"),
                    ),
                ),
            ),
        ),

        # PDF
        ui.card(
            ui.card_header("전처리 PDF"),
            ui.p("아래 버튼을 눌러 전처리 상세 문서를 PDF로 다운로드하세요."),
            ui.download_button("download_pdf", "PDF 다운로드"),
            ui.div(ui.output_text("pdf_status"), class_="text-muted mt-2"),
        ),
    )

# =========================
# SERVER
# =========================
def server(input, output, session):

    # ── DataFrames ────────────────────────────────────────────────────────────
    @reactive.calc
    def _raw_df() -> pd.DataFrame:
        # 문자열 보존용(등록/시간 열 미리보기 등에 사용)
        return pd.read_csv(DATA_FILE, dtype=str, low_memory=False)

    @reactive.calc
    def _num_df() -> pd.DataFrame:
        # 수치 시각화용
        df = pd.read_csv(DATA_FILE, low_memory=False)
        return df

    # ── (단일 칼럼) heating_furnace: A/B/NaN 개수 바 차트 ─────────────────────
    @output
    @render.plot
    def hf_hist():
        df = _raw_df()
        if "heating_furnace" not in df.columns:
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "heating_furnace 열이 없습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        s = df["heating_furnace"]
        # NaN을 문자열 "NaN"으로 치환 후 개수 집계
        s = s.where(~s.isna(), "NaN")
        vc = s.value_counts(dropna=False)  # A/B/NaN
        labels = vc.index.tolist()
        values = vc.values

        fig, ax = plt.subplots(figsize=(9, 3.6))
        bars = ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#bcbddc"])
        ax.set_ylabel("Count")
        ax.set_title("heating_furnace 분포 (A/B/NaN)")

        for rect, val in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                    f"{int(val):,}", ha="center", va="bottom", fontsize=10)
        return fig
    
    # --- heating_furnace: 특정 인덱스 구간 미리보기 ---
    @output
    @render.data_frame
    def hf_slice_df():
        df = _raw_df()
        required = ["heating_furnace", "mold_code", "time", "date", "molten_volume", "count"]
        if not set(required).issubset(df.columns):
            return pd.DataFrame()

        start, end = 73406, 73418
        subset = df.loc[start:end, required].copy() if df.index.max() >= end \
                 else df.loc[:, required].head(12).copy()

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


    # ── (단일 칼럼) molten_volume: 결측 vs 비결측 도넛 차트 ─────────────────────
    # ── (단일 칼럼) molten_volume: 결측 vs 비결측 도넛 차트 ─────────────────────
    @output
    @render.plot
    def mv_pie():
        df = _raw_df()
        if "molten_volume" not in df.columns or df.empty:
            fig = plt.figure(figsize=(3, 2))
            plt.text(0.5, 0.5, "molten_volume 칼럼이 없거나 데이터가 비어있습니다.",
                     ha="center", va="center")
            plt.axis("off")
            return fig
    
        # 문자열로 통일 후 공백 제거
        s = df["molten_volume"].astype("string").str.strip()
    
        # 결측 판단: NaN + 빈문자 + 'nan'/'none' (대소문자 무시)
        miss_mask = s.isna() | s.eq("") | s.str.lower().isin(["nan", "none"])
        n_miss = int(miss_mask.sum())
        n_ok   = int((~miss_mask).sum())
    
        labels = ["결측", "비결측"]
        sizes  = [n_miss, n_ok]
        colors = ["#e15759", "#4c78a8"]
    
        fig, ax = plt.subplots(figsize=(9, 4.2))
        # 반환값 언팩하지 않으면 버전 차이에도 안전
        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.38, edgecolor="white"),  # 도넛 형태
            pctdistance=0.80,
            textprops={"fontsize": 12},
            autopct=lambda p: f"{p:.1f}%\n({int(round(p/100*sum(sizes))):,})",
        )
        ax.set_title("molten_volume 결측/비결측 비율", pad=8)
        ax.axis("equal")
        return fig


    # ── (단일 칼럼) upper/lower_mold_temp3: 0~1500 히스토그램 + 1449 라인 ─────
    @output
    @render.plot
    def mt3_hist():
        df = _num_df()
        cols = ["upper_mold_temp3", "lower_mold_temp3"]
        if not set(cols).issubset(df.columns):
            fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "upper/lower_mold_temp3 열이 없습니다.", ha="center", va="center")
            plt.axis("off")
            return fig

        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
        fig.subplots_adjust(hspace=0.4, top=0.92, bottom=0.12, left=0.08, right=0.98)

        name_map = {"upper_mold_temp3": "Upper Mold Temp3", "lower_mold_temp3": "Lower Mold Temp3"}
        for i, c in enumerate(cols):
            ax = axes[i]
            s_all  = df[c].dropna()
            s_plot = s_all[(s_all >= 0) & (s_all <= 1500)]
            ax.hist(s_plot, bins=120, color="#4c78a8", alpha=0.85, edgecolor="white")
            ax.axvline(1449.0, color="red", linestyle="--", linewidth=1.8)
            cnt_1449 = int((s_all == 1449.0).sum())
            ax.set_xlim(0, 1500)
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Count")
            ax.set_title(f"{name_map[c]} — n={len(s_plot):,} (x≤1500), 1449.0 개수={cnt_1449:,}")
            ymax = ax.get_ylim()[1]
            ax.text(1449.0, ymax * 0.9, f"{cnt_1449:,}", color="red",
                    ha="left", va="center", fontsize=10, rotation=90)

        return fig

    # ── (단일 칼럼) registration/time/date 5행 미리보기 ────────────────────────
    @output
    @render.data_frame
    def reg_head_df():
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

        view = df[cols].head(5).copy()
        if reg_col and reg_col != "registration":
            view = view.rename(columns={reg_col: "registration"})
        view = view.replace({pd.NA: "Nan", None: "Nan", "": "Nan"}).fillna("Nan")
        return view

    # ── PDF 상태/다운로드 ──────────────────────────────────────────────────────
    @output
    @render.text
    def pdf_status():
        return f"파일 위치: {PDF_FILE}" if PDF_FILE.exists() \
            else "주의: PDF 파일이 아직 없습니다. reports/preprocessing_report.pdf 경로에 파일을 생성해 주세요."

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
