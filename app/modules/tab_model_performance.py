# modules/tab_process_explanation.py
from pathlib import Path

from shiny import ui, reactive, render
import plotly.express as px
import plotly.io as pio
import pandas as pd
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
SCORE_V0_PATH = APP_DIR / "data" / "models" / "v0"
SCORE_V1_PATH = APP_DIR / "data" / "models" / "v1"
SCORE_V2_PATH = APP_DIR / "data" / "models" / "v2"

SCORE_V0_LGBM = SCORE_V0_PATH / "LightGBM_v0_scores.csv"
SCORE_V1_LGBM = SCORE_V1_PATH / "LightGBM_v1_scores.csv"
SCORE_V2_LGBM = SCORE_V2_PATH / "LightGBM_v2_scores.csv"

SCORE_V0_RF = SCORE_V0_PATH / "RandomForest_v0_scores.csv"
SCORE_V1_RF = SCORE_V1_PATH / "RandomForest_v1_scores.csv"
SCORE_V2_RF = SCORE_V2_PATH / "RandomForest_v2_scores.csv"

SCORE_V0_XGB = SCORE_V0_PATH / "XGBoost_v0_scores.csv"
SCORE_V1_XGB = SCORE_V1_PATH / "XGBoost_v1_scores.csv"
SCORE_V2_XGB = SCORE_V2_PATH / "XGBoost_v2_scores.csv"

MODELS = ["LightGBM", "RandomForest", "XGBoost"]
VERSIONS = ["v0", "v1", "v2"]

DISPLAY_METRIC_MAP = {
    "ROC-AUC": "roc_auc",
    "F1-Score": "f1_score",
    "Recall": "recall",
    "Precision": "precision",
}

DETAIL_METRIC_FIELDS = [
    ("PR-AUC", "pr_auc"),
    ("ROC-AUC", "roc_auc"),
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1-Score", "f1_score"),
    ("Operating Threshold", "operating_threshold"),
]

APP_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "data" / "models"

MODEL_METRICS: dict[str, dict[str, dict[str, float]]] = {model: {} for model in MODELS}
METRIC_VALUES: dict[str, dict[str, dict[str, float]]] = {}
HIGHLIGHT_MAP: dict[str, tuple[str, str] | None] = {}


def _load_model_metrics() -> None:
    """Load model score summaries from data/models directory."""

    global MODEL_METRICS, METRIC_VALUES, HIGHLIGHT_MAP

    # Reset containers
    MODEL_METRICS = {model: {} for model in MODELS}
    METRIC_VALUES = {metric: {model: {} for model in MODELS} for metric in DISPLAY_METRIC_MAP.keys()}
    HIGHLIGHT_MAP = {}

    for version in VERSIONS:
        version_dir = MODELS_DIR / version
        if not version_dir.exists():
            continue

        for model in MODELS:
            file_name = f"{model}_{version}_scores.csv"
            file_path = version_dir / file_name
            if not file_path.exists():
                continue

            try:
                df = pd.read_csv(file_path)
            except Exception:
                continue

            if df.empty:
                continue

            row = df.iloc[0].to_dict()
            MODEL_METRICS[model][version] = row

            for display_label, column_name in DISPLAY_METRIC_MAP.items():
                value = row.get(column_name)
                if pd.notna(value):
                    METRIC_VALUES[display_label][model][version] = float(value)

    for display_label, column_name in DISPLAY_METRIC_MAP.items():
        best_value = None
        best_pair: tuple[str, str] | None = None

        for model in MODELS:
            for version in VERSIONS:
                value = MODEL_METRICS.get(model, {}).get(version, {}).get(column_name)
                if value is None or pd.isna(value):
                    continue
                if best_value is None or value > best_value:
                    best_value = value
                    best_pair = (model, version)

        if best_pair:
            HIGHLIGHT_MAP[display_label] = best_pair
        else:
            HIGHLIGHT_MAP[display_label] = None


_load_model_metrics()

custom_css = """
<style>
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    border: 1px solid #dee2e6;
    border-radius: 8px;
    overflow: hidden;
}
.metrics-grid > div {
    aspect-ratio: 1 / 1;
    display: flex;
    align-items: center;
    justify-content: center;
    border-right: 1px solid #dee2e6;
    border-bottom: 1px solid #dee2e6;
    font-size: 1.025rem;
    font-weight: 500;
}
.metrics-grid > div:nth-child(4n) {
    border-right: none;
}
.metrics-grid > div:nth-last-child(-n + 4) {
    border-bottom: none;
}
.metrics-grid__header {
    background-color: #f8f9fa;
    font-weight: 600;
}
.metrics-grid__row-header {
    background-color: #fdfdfd;
    font-weight: 600;
}
.metrics-grid__cell--highlight {
    background-color: #2A2D30;
    color: #ffffff;
    font-weight: 700;
}
.metrics-btn.active {
    background-color: #2A2D30 !important;
    color: #ffffff !important;
}
.metrics-btn {
    transition: all 0.2s ease;
}
.metric-detail-card {
    background-color: #ffffff;
    border: 1px solid #e3e6eb;
    border-radius: 16px;
    padding: 20px 24px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 12px;
}
.metric-detail-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #212529;
}
.metric-detail-subtitle {
    font-size: 0.95rem;
    font-weight: 500;
    color: #5c636a;
}
.metric-detail-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}
.metric-detail-table th {
    width: 50%;
    padding: 10px 12px;
    text-align: left;
    font-weight: 600;
    color: #495057;
    background-color: #f8f9fa;
    border-top: 1px solid #e3e6eb;
    border-left: 1px solid #e3e6eb;
}
.metric-detail-table td {
    padding: 10px 12px;
    text-align: right;
    font-weight: 600;
    color: #212529;
    border-top: 1px solid #e3e6eb;
    border-right: 1px solid #e3e6eb;
}
.metric-detail-table tr:first-child th,
.metric-detail-table tr:first-child td {
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}
.metric-detail-table tr:last-child th {
    border-bottom-left-radius: 12px;
    border-bottom: 1px solid #e3e6eb;
}
.metric-detail-table tr:last-child td {
    border-bottom-right-radius: 12px;
    border-bottom: 1px solid #e3e6eb;
}
.importance-card {
    margin-top: 16px;
    padding: 16px 20px;
    background-color: #ffffff;
    border: 1px solid #e3e6eb;
    border-radius: 16px;
}
.importance-card h4 {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 12px;
    color: #212529;
}
</style>
"""


def _build_metrics_grid(metric_name: str) -> str:
    metric_data = METRIC_VALUES.get(metric_name, {})
    if not metric_data:
        metric_data = {}

    highlight_pair = HIGHLIGHT_MAP.get(metric_name)

    cells = ["<div class='metrics-grid mt-3'>"]

    # Header row
    cells.append("<div class='metrics-grid__header'>모델</div>")
    for version in VERSIONS:
        cells.append(f"<div class='metrics-grid__header'>{version}</div>")

    # Body rows
    for model in MODELS:
        cells.append("<div class='metrics-grid__row-header'>{}</div>".format(model))
        for version in VERSIONS:
            value = metric_data.get(model, {}).get(version)
            cell_classes = ["metrics-grid__cell"]
            if highlight_pair and (model, version) == highlight_pair:
                cell_classes.append("metrics-grid__cell--highlight")
            display_value = "-" if value is None else f"{value:.3f}"
            cells.append(
                f"<div class='{' '.join(cell_classes)}'>{display_value}</div>"
            )

    cells.append("</div>")
    return "".join(cells)


def _build_metric_button(button_id: str, label: str, active: bool) -> ui.Tag:
    classes = ["flex-fill", "metrics-btn"]
    if active:
        classes.append("active")
    return ui.input_action_button(
        button_id,
        label,
        class_=" ".join(classes),
        style="font-size:0.85rem; white-space:nowrap; width:100%;",
    )


def _build_selection_details(metric_name: str) -> ui.Tag:
    highlight = HIGHLIGHT_MAP.get(metric_name)
    if not highlight:
        return ui.div("선택된 모델 정보가 없습니다.", class_="text-muted")

    model_name, version = highlight
    metric_rows = []
    metrics_dict = MODEL_METRICS.get(model_name, {}).get(version, {})

    for display_label, column_name in DETAIL_METRIC_FIELDS:
        value = metrics_dict.get(column_name)
        if value is None or pd.isna(value):
            display_value = "-"
        else:
            display_value = f"{float(value):.3f}"
        metric_rows.append(
            ui.tags.tr(
                ui.tags.th(display_label),
                ui.tags.td(display_value),
            )
        )

    return ui.div(
        ui.div(f"{model_name}", class_="metric-detail-header"),
        ui.div(f"버전: {version}", class_="metric-detail-subtitle"),
        ui.tags.table(
            {"class": "metric-detail-table"},
            *metric_rows,
        ),
        class_="metric-detail-card",
    )


def _generate_mock_importance(model_name: str) -> pd.DataFrame:
    mock_data = {
        "LightGBM": {
            "features": [
                "high_section_speed",
                "cast_pressure",
                "upper_mold_temp1",
                "physical_strength",
                "coolant_temperature",
            ],
            "importance": [0.24, 0.19, 0.17, 0.14, 0.11],
        },
        "RandomForest": {
            "features": [
                "molten_temp",
                "production_cycletime",
                "low_section_speed",
                "upper_mold_temp2",
                "cast_pressure",
            ],
            "importance": [0.21, 0.18, 0.16, 0.13, 0.12],
        },
        "XGBoost": {
            "features": [
                "working",
                "EMS_operation_time",
                "lower_mold_temp1",
                "sleeve_temperature",
                "biscuit_thickness",
            ],
            "importance": [0.23, 0.20, 0.17, 0.15, 0.09],
        },
    }

    entry = mock_data.get(model_name, mock_data["LightGBM"])
    return pd.DataFrame(entry)

def panel():
    return ui.nav_panel(
        "모델 성능 평가",
        ui.HTML(custom_css),
        ui.div(
            ui.div(
                ui.card(
                    ui.card_header("모델 선택"),
                    ui.card_body(
                        ui.output_ui("metric_button_row"),
                        ui.output_ui("metrics_grid"),
                        class_="w-100",
                    ),
                    full_height=True,
                    class_="h-100 d-flex flex-column",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            ui.div(
                ui.card(
                    ui.card_header("상세 정보"),
                    ui.card_body(
                        ui.output_ui("selection_details"),
                        ui.output_ui("importance_chart"),
                        class_="w-100 h-100 d-flex flex-column gap-3",
                    ),
                    full_height=True,
                    class_="h-100 d-flex flex-column",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            class_="d-flex flex-row align-stretch vh-100 w-100",
            style="gap: 1rem; width: 100%;",
        ),
    )

def server(input, output, session):
    active_metric = reactive.Value("ROC-AUC")

    @reactive.effect
    @reactive.event(input.btn_metric_roc_auc)
    def _set_metric_roc_auc():
        active_metric.set("ROC-AUC")

    @reactive.effect
    @reactive.event(input.btn_metric_f1)
    def _set_metric_f1():
        active_metric.set("F1-Score")

    @reactive.effect
    @reactive.event(input.btn_metric_recall)
    def _set_metric_recall():
        active_metric.set("Recall")

    @reactive.effect
    @reactive.event(input.btn_metric_precision)
    def _set_metric_precision():
        active_metric.set("Precision")

    @render.ui
    def metrics_grid():
        metric_name = active_metric.get()
        return ui.HTML(_build_metrics_grid(metric_name))

    @render.ui
    def metric_button_row():
        metric_name = active_metric.get()
        return ui.div(
            _build_metric_button("btn_metric_roc_auc", "ROC-AUC", metric_name == "ROC-AUC"),
            _build_metric_button("btn_metric_f1", "F1-Score", metric_name == "F1-Score"),
            _build_metric_button("btn_metric_recall", "Recall", metric_name == "Recall"),
            _build_metric_button("btn_metric_precision", "Precision", metric_name == "Precision"),
            class_="d-flex w-100 gap-2",
        )

    @render.ui
    def selection_details():
        metric_name = active_metric.get()
        return _build_selection_details(metric_name)

    @render.ui
    def importance_chart():
        metric_name = active_metric.get()
        highlight = HIGHLIGHT_MAP.get(metric_name)
        if not highlight:
            return ui.div("변수 중요도 정보가 없습니다.", class_="text-muted")

        model_name, _ = highlight
        mock_df = _generate_mock_importance(model_name)
        fig = px.bar(
            mock_df,
            x="importance",
            y="features",
            orientation="h",
            title=None,
            labels={"features": "변수", "importance": "중요도"},
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
        )
        fig.update_traces(marker_color="#2A2D30")

        return ui.div(
            ui.h4("변수 중요도", class_="mb-3"),
            ui.HTML(
                pio.to_html(
                    fig,
                    include_plotlyjs="cdn",
                    full_html=False,
                    config={"displayModeBar": False},
                )
            ),
            class_="importance-card",
        )

