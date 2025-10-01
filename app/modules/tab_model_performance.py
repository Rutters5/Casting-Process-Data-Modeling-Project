# modules/tab_process_explanation.py
from pathlib import Path

from shiny import ui, reactive, render
import plotly.express as px
import plotly.io as pio
import pandas as pd

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
    ("ROC-AUC", "roc_auc"),
    ("F1-Score", "f1_score"),
    ("Precision", "precision"),
    ("Recall", "recall"),
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
    height: auto;
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
.insight-tabset .nav-tabs {
    border-bottom: 1px solid #e3e6eb;
    margin-bottom: 0.75rem;
}
.insight-tabset .nav-link {
    font-weight: 600;
    color: #5c636a;
}
.insight-tabset .nav-link.active {
    color: #2A2D30;
    border-color: #e3e6eb #e3e6eb transparent;
}
.insight-tabset {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 520px;
}
.insight-tabset .tab-content {
    flex-grow: 1;
    display: flex;
}
.insight-tabset .tab-content > .tab-pane {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
    min-height: 360px;
}
.insight-tabset .tab-content > .tab-pane:not(.active) {
    display: none;
}
.card-no-gap {
    margin-bottom: 0 !important;
}
.panel-equal-row {
    align-items: stretch !important;
}
.card-equal {
    height: 100%;
    min-height: 520px;
    display: flex;
    flex-direction: column;
}
.card-equal .card-body {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
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
    cells.append("<div class='metrics-grid__header'>버전</div>")
    for model in MODELS:
        cells.append(f"<div class='metrics-grid__header'>{model}</div>")

    # Body rows
    for version in VERSIONS:
        cells.append("<div class='metrics-grid__row-header'>{}</div>".format(version))
        for model in MODELS:
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


def _generate_mock_best_params(model_name: str, version: str) -> dict[str, str]:
    mock_params = {
        "LightGBM": {
            "num_leaves": "64",
            "learning_rate": "0.045",
            "feature_fraction": "0.72",
            "bagging_fraction": "0.80",
            "min_child_samples": "28",
        },
        "RandomForest": {
            "n_estimators": "280",
            "max_depth": "18",
            "min_samples_split": "4",
            "min_samples_leaf": "2",
            "max_features": "sqrt",
        },
        "XGBoost": {
            "max_depth": "6",
            "eta": "0.11",
            "subsample": "0.78",
            "colsample_bytree": "0.74",
            "gamma": "0.8",
        },
    }

    params = mock_params.get(model_name, mock_params["LightGBM"]).copy()
    params["model_version"] = version
    return params


def _generate_mock_shap_summary(model_name: str) -> pd.DataFrame:
    mock_shap = {
        "LightGBM": {
            "feature": [
                "high_section_speed",
                "cast_pressure",
                "upper_mold_temp1",
                "coolant_temperature",
                "physical_strength",
            ],
            "mean_abs_shap": [0.128, 0.104, 0.091, 0.075, 0.062],
            "impact": ["양(+)", "양(+)", "음(-)", "양(+)", "음(-)"],
        },
        "RandomForest": {
            "feature": [
                "molten_temp",
                "production_cycletime",
                "low_section_speed",
                "upper_mold_temp2",
                "cast_pressure",
            ],
            "mean_abs_shap": [0.117, 0.101, 0.087, 0.073, 0.066],
            "impact": ["양(+)", "음(-)", "양(+)", "음(-)", "양(+)"]
        },
        "XGBoost": {
            "feature": [
                "working",
                "EMS_operation_time",
                "lower_mold_temp1",
                "sleeve_temperature",
                "biscuit_thickness",
            ],
            "mean_abs_shap": [0.133, 0.118, 0.095, 0.082, 0.058],
            "impact": ["음(-)", "양(+)", "양(+)", "음(-)", "양(+)"]
        },
    }

    entry = mock_shap.get(model_name, mock_shap["LightGBM"])
    return pd.DataFrame(entry)


def _build_importance_tab(model_name: str) -> ui.Tag:
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


def _build_best_params_tab(model_name: str, version: str) -> ui.Tag:
    params = _generate_mock_best_params(model_name, version)
    rows = []
    for key, value in params.items():
        display_key = key.replace("_", " ").title()
        rows.append(
            ui.tags.tr(
                ui.tags.th(display_key),
                ui.tags.td(value),
            )
        )

    return ui.div(
        ui.tags.table(
            {"class": "metric-detail-table"},
            *rows,
        ),
        class_="importance-card",
    )


def _build_shap_tab(model_name: str) -> ui.Tag:
    shap_df = _generate_mock_shap_summary(model_name)
    fig = px.bar(
        shap_df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="impact",
        labels={"feature": "변수", "mean_abs_shap": "|SHAP| 평균", "impact": "영향 방향"},
        color_discrete_map={"양(+)": "#2A2D30", "음(-)": "#8A9098"},
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 0.16]),
        plot_bgcolor="white",
        legend_title_text="영향",
    )

    return ui.div(
        ui.HTML(
            pio.to_html(
                fig,
                include_plotlyjs="cdn",
                full_html=False,
                config={"displayModeBar": False},
            )
        ),
        ui.p("상위 중요 변수의 SHAP 값을 임의 데이터로 표시했습니다."),
        class_="importance-card",
    )


def _build_insight_tabs(model_name: str, version: str) -> ui.Tag:
    navset = ui.navset_tab(
        ui.nav_panel("변수 중요도", _build_importance_tab(model_name)),
        ui.nav_panel("베스트 파라미터", _build_best_params_tab(model_name, version)),
        ui.nav_panel("SHAP 설명", _build_shap_tab(model_name)),
    )
    return ui.div(navset, class_="insight-tabset")

def panel_body():
    return ui.TagList(
        ui.HTML(custom_css),
        ui.div(
            ui.div(
                ui.card(
                    ui.card_header("모델 선택"),
                    ui.card_body(
                        ui.output_ui("metric_button_row"),
                        ui.output_ui("metrics_grid"),
                        class_="w-100 d-flex flex-column gap-3 flex-grow-1",
                    ),
                    class_="card-no-gap card-equal",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            ui.div(
                ui.card(
                    ui.card_header("모델 정보"),
                    ui.card_body(
                        ui.output_ui("selection_details"),
                        ui.output_ui("insight_tabs"),
                        class_="w-100 d-flex flex-column gap-3 flex-grow-1",
                    ),
                    class_="card-no-gap card-equal",
                ),
                class_="d-flex flex-column",
                style="flex: 1 1 50%; max-width: 50%;",
            ),
            class_="d-flex flex-row align-items-stretch panel-equal-row w-100",
            style="gap: 1rem; width: 100%;",
        ),
    )


def panel():
    return ui.nav_panel("모델 성능 평가", panel_body())
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
    def insight_tabs():
        metric_name = active_metric.get()
        highlight = HIGHLIGHT_MAP.get(metric_name)
        if not highlight:
            return ui.div("세부 인사이트 정보가 없습니다.", class_="text-muted")

        model_name, version = highlight
        return _build_insight_tabs(model_name, version)

