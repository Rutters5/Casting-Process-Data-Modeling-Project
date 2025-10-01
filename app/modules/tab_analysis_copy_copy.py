from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# --- 경로 및 자원 로드 ---
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_FILE = BASE_DIR / "data" / "processed" / "test_v1.csv"
MODEL_FILE = BASE_DIR / "data" / "models" / "v1" / "LightGBM_v1.pkl"

df = pd.read_csv(DATA_FILE, encoding="utf-8", low_memory=False)
artifact = joblib.load(MODEL_FILE)

model = artifact["model"]
scaler = artifact.get("scaler")
ordinal_encoder = artifact.get("ordinal_encoder")
onehot_encoder = artifact.get("onehot_encoder")
operating_threshold = float(artifact.get("operating_threshold", 0.5))

numeric_cols_model = list(getattr(scaler, "feature_names_in_", [])) if scaler is not None else []
categorical_cols_model = list(getattr(ordinal_encoder, "feature_names_in_", [])) if ordinal_encoder is not None else []
all_model_cols = numeric_cols_model + categorical_cols_model

missing_columns = [col for col in all_model_cols if col not in df.columns]
for col in missing_columns:
    df[col] = np.nan

numeric_cols = numeric_cols_model
categorical_cols = categorical_cols_model
all_cols = all_model_cols

COLUMN_NAMES_KR = {
    "registration_time": "등록 일시",
    "count": "생산 순번",
    "working": "가동 여부",
    "emergency_stop": "비상 정지",
    "facility_operation_cycleTime": "설비 운영 사이클타임",
    "production_cycletime": "제품 생산 사이클타임",
    "low_section_speed": "저속 구간 속도",
    "high_section_speed": "고속 구간 속도",
    "cast_pressure": "주조 압력",
    "biscuit_thickness": "비스킷 두께",
    "upper_mold_temp1": "상부 금형 온도1",
    "upper_mold_temp2": "상부 금형 온도2",
    "lower_mold_temp1": "하부 금형 온도1",
    "lower_mold_temp2": "하부 금형 온도2",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "전자교반 가동시간",
    "mold_code": "금형 코드",
    "tryshot_signal": "트라이샷 신호",
    "molten_temp": "용탕 온도",
    "uniformity": "균일도",
    "mold_temp_udiff": "금형 온도차(상/하)",
    "P_diff": "압력 차이",
    "Cycle_diff": "사이클 시간 차이"
}
pass_reference = df[df.get("passorfail", 0) == 0].copy()
if pass_reference.empty:
    pass_reference = df.copy()


NUMERIC_FEATURE_RANGES = {}
for col in numeric_cols:
    series = pass_reference[col].dropna()
    if series.empty:
        series = df[col].dropna()
    if not series.empty:
        NUMERIC_FEATURE_RANGES[col] = (float(series.min()), float(series.max()))

explainer = shap.TreeExplainer(model)

def _resolve_expected_value(expected):
    if isinstance(expected, (list, tuple, np.ndarray)):
        if len(expected) > 1:
            return float(expected[1])
        return float(expected[0])
    return float(expected)

SHAP_EXPECTED_VALUE = _resolve_expected_value(explainer.expected_value)

numeric_index_map = {feat: idx for idx, feat in enumerate(numeric_cols)}

ohe_feature_slices = {}
ohe_value_labels = {}
start_idx = len(numeric_cols)
if categorical_cols and onehot_encoder is not None and ordinal_encoder is not None:
    for feat, ohe_cats, ord_cats in zip(categorical_cols, onehot_encoder.categories_, ordinal_encoder.categories_):
        length = len(ohe_cats)
        ohe_feature_slices[feat] = (start_idx, start_idx + length)
        labels = []
        for code in ohe_cats:
            code_int = int(code)
            if 0 <= code_int < len(ord_cats):
                labels.append(str(ord_cats[code_int]))
            else:
                labels.append("unknown")
        ohe_value_labels[feat] = labels
        start_idx += length
else:
    for feat in categorical_cols:
        ohe_feature_slices[feat] = (len(numeric_cols), len(numeric_cols))
        ohe_value_labels[feat] = []

def build_input_dataframe(row_dict):
    data = {col: row_dict.get(col) for col in all_cols}
    input_df = pd.DataFrame([data], columns=all_cols)
    if numeric_cols:
        input_df[numeric_cols] = input_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if categorical_cols:
        input_df[categorical_cols] = input_df[categorical_cols].fillna("unknown").astype(str)
    return input_df

def prepare_feature_matrix(input_df):
    arrays = []
    if numeric_cols:
        if scaler is not None:
            num_part = scaler.transform(input_df[numeric_cols].astype(float))
        else:
            num_part = input_df[numeric_cols].to_numpy(dtype=float)
        arrays.append(num_part)
    if categorical_cols:
        if ordinal_encoder is not None and onehot_encoder is not None:
            cat_values = input_df[categorical_cols]
            cat_ord = ordinal_encoder.transform(cat_values).astype(int)
            cat_ohe = onehot_encoder.transform(cat_ord)
            if hasattr(cat_ohe, "toarray"):
                cat_ohe = cat_ohe.toarray()
        else:
            cat_ohe = np.zeros((len(input_df), 0))
        arrays.append(cat_ohe)
    if arrays:
        return np.hstack(arrays).astype(np.float32)
    return np.zeros((len(input_df), 0), dtype=np.float32)

def compute_shap_contributions(feature_matrix):
    shap_values = explainer.shap_values(feature_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    shap_vector = shap_values[0]
    contributions = {}
    for feat, idx in numeric_index_map.items():
        contributions[feat] = float(shap_vector[idx])
    for feat, (start, end) in ohe_feature_slices.items():
        if end > start:
            contributions[feat] = float(np.sum(shap_vector[start:end]))
        else:
            contributions[feat] = 0.0
    return contributions, shap_vector

def predict_with_model(row_dict, compute_shap=False):
    input_df = build_input_dataframe(row_dict)
    feature_matrix = prepare_feature_matrix(input_df)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_matrix)
        probability = float(proba[0, 1] if proba.ndim == 2 else proba[0])
    else:
        raw_pred = model.predict(feature_matrix)
        probability = float(raw_pred[0] if raw_pred.ndim else raw_pred)
    prediction = 1 if probability >= operating_threshold else 0
    forced_fail = False
    tryshot = row_dict.get("tryshot_signal")
    if tryshot is not None and str(tryshot).upper() == "D":
        if probability < operating_threshold:
            forced_fail = True
        prediction = 1
    result = {
        "probability": probability,
        "prediction": prediction,
        "forced_fail": forced_fail,
        "input_df": input_df,
        "features": feature_matrix
    }
    if compute_shap:
        contributions, shap_vector = compute_shap_contributions(feature_matrix)
        result["shap_aggregated"] = contributions
        result["shap_vector"] = shap_vector
    return result

def _extract_feature_values(row_dict):
    values = []
    for col in all_cols:
        val = row_dict.get(col, np.nan)
        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else np.nan
        if isinstance(val, (pd.Timestamp, pd.Timedelta)):
            val = str(val)
        if pd.isna(val):
            values.append(np.nan)
        else:
            values.append(val)
    return values


def build_shap_explanation(contributions, input_row):
    if not contributions:
        return None
    shap_values = np.array([float(contributions.get(col, 0.0)) for col in all_cols], dtype=float)
    feature_values = np.array(_extract_feature_values(input_row), dtype=object)
    feature_names = [COLUMN_NAMES_KR.get(col, col) for col in all_cols]
    try:
        return shap.Explanation(values=shap_values, base_values=SHAP_EXPECTED_VALUE, data=feature_values, feature_names=feature_names)
    except Exception:
        return None


def evaluate_prediction(row_dict):
    result = predict_with_model(row_dict, compute_shap=False)
    return result["prediction"], result["probability"]


def find_normal_range_binary_fixed(base_row, feature, bounds, threshold=operating_threshold, tol_ratio=0.01, max_iter=20, n_check=5):
    if not bounds:
        return None
    f_min, f_max = bounds
    if pd.isna(f_min) or pd.isna(f_max):
        return None
    low, high = float(f_min), float(f_max)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return None
    tol = max((high - low) * tol_ratio, 1e-3)
    best_details = None
    for _ in range(max_iter):
        samples = np.linspace(low, high, n_check)
        normal_samples = []
        for val in samples:
            trial = base_row.copy()
            trial[feature] = float(val)
            pred, prob = evaluate_prediction(trial)
            if pred == 0:
                normal_samples.append((float(val), float(prob)))
        if not normal_samples:
            break
        normal_samples.sort(key=lambda item: item[1])
        low = min(v for v, _ in normal_samples)
        high = max(v for v, _ in normal_samples)
        top_val, top_prob = normal_samples[0]
        examples = [top_val]
        if low not in examples:
            examples.append(low)
        if high not in examples:
            examples.append(high)
        if best_details is None or top_prob < best_details[3]:
            best_details = (low, high, examples[:3], top_prob)
        if (high - low) <= tol:
            break
    if best_details is None:
        return None
    low, high, examples, best_prob = best_details
    return {
        "min": float(low),
        "max": float(high),
        "examples": [float(v) for v in examples],
        "best_prob": float(best_prob)
    }


def binary_search_normal_ranges(base_row, features, feature_ranges, threshold=operating_threshold, max_iter=10, tol_ratio=0.01):
    usable = {}
    for feat in features:
        bounds = feature_ranges.get(feat)
        if not bounds:
            continue
        f_min, f_max = bounds
        if pd.isna(f_min) or pd.isna(f_max) or not np.isfinite(f_min) or not np.isfinite(f_max):
            continue
        if f_min >= f_max:
            continue
        usable[feat] = [float(f_min), float(f_max)]
    if not usable:
        return None, {}, None
    best_solution = None
    best_prob = None
    for _ in range(max_iter):
        trial = base_row.copy()
        mids = {}
        for feat, (low, high) in usable.items():
            mid = (low + high) / 2.0
            mids[feat] = mid
            trial[feat] = mid
        pred, prob = evaluate_prediction(trial)
        is_normal = pred == 0
        if is_normal and (best_prob is None or prob < best_prob):
            best_prob = float(prob)
            best_solution = {feat: float(val) for feat, val in mids.items()}
        updated = False
        for feat, (low, high) in list(usable.items()):
            mid = mids[feat]
            left = mid - low
            right = high - mid
            if left <= 0 and right <= 0:
                continue
            base_range = feature_ranges[feat]
            tol = max((base_range[1] - base_range[0]) * tol_ratio, 1e-3)
            if is_normal:
                new_range = [low, mid] if left >= right else [mid, high]
            else:
                new_range = [mid, high] if left >= right else [low, mid]
            if abs(new_range[1] - new_range[0]) < tol:
                new_range = [float(new_range[0]), float(new_range[1])]
            if new_range != usable[feat]:
                usable[feat] = new_range
                updated = True
        if not updated:
            break
    return best_solution, {feat: tuple(bounds) for feat, bounds in usable.items()}, best_prob


def evaluate_categorical_candidates(base_row, feature, choices, top_k=3):
    candidates = []
    for value in choices:
        trial = base_row.copy()
        trial[feature] = value
        pred, prob = evaluate_prediction(trial)
        if pred == 0:
            candidates.append((value, float(prob)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[1])
    values = [val for val, _ in candidates[:top_k]]
    return {
        "values": values,
        "best_prob": float(candidates[0][1])
    }


def recommend_ranges(base_row, focus_features):
    if not focus_features:
        return {}
    recommendations = {}
    best_prob = None

    numeric_targets = [feat for feat in focus_features if feat in numeric_cols]
    categorical_targets = [feat for feat in focus_features if feat in categorical_cols]

    numeric_ranges = {feat: NUMERIC_FEATURE_RANGES.get(feat) for feat in numeric_targets if NUMERIC_FEATURE_RANGES.get(feat)}

    if len(numeric_ranges) >= 2:
        solution, final_ranges, prob_multi = binary_search_normal_ranges(base_row, list(numeric_ranges.keys()), numeric_ranges, threshold=operating_threshold)
        if solution:
            for feat, mid in solution.items():
                bounds = final_ranges.get(feat, numeric_ranges.get(feat))
                if not bounds:
                    continue
                record = recommendations.get(feat, {"type": "numeric"})
                record["min"] = float(bounds[0])
                record["max"] = float(bounds[1])
                examples = record.get("examples", [])
                if mid not in examples:
                    examples.append(float(mid))
                record["examples"] = examples[:3]
                record["method"] = "binary_multi"
                recommendations[feat] = record
            if prob_multi is not None:
                best_prob = prob_multi if best_prob is None else min(best_prob, prob_multi)

    for feat, bounds in numeric_ranges.items():
        details = find_normal_range_binary_fixed(base_row, feat, bounds, threshold=operating_threshold)
        if not details:
            continue
        record = recommendations.get(feat, {"type": "numeric"})
        record["min"] = details["min"]
        record["max"] = details["max"]
        examples = record.get("examples", [])
        for val in details.get("examples", []):
            if val not in examples:
                examples.append(val)
        record["examples"] = examples[:3]
        record["method"] = record.get("method", "binary_search")
        recommendations[feat] = record
        prob_val = details.get("best_prob")
        if prob_val is not None:
            best_prob = prob_val if best_prob is None else min(best_prob, prob_val)

    for feat in categorical_targets:
        meta = input_metadata.get(feat)
        if not meta:
            continue
        choices = meta.get("choices", [])
        result = evaluate_categorical_candidates(base_row, feat, choices, top_k=3)
        if not result:
            continue
        recommendations[feat] = {
            "type": "categorical",
            "values": result["values"]
        }
        best_prob = result["best_prob"] if best_prob is None else min(best_prob, result["best_prob"])

    if best_prob is not None:
        recommendations["best_probability"] = float(best_prob)
    return recommendations

def format_value(value):
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)

custom_css = """
<style>
body {
    font-family: -apple-system, sans-serif;
    background-color: #f5f7fa;
}
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
.accordion-header:hover { background-color: #3B3E42; }
.accordion-content {
    padding: 24px 28px;
    background: #ffffff;
    border-radius: 0 0 16px 16px;
}
.input-item { margin-bottom: 20px; }
.irs--shiny .irs-bar { background: #2A2D30; }
.irs--shiny .irs-handle { border: 2px solid #142D4A; background: white; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single { background: #2A2D30; }
#predict:hover { background: #b91f1f !important; transform: translateY(-1px); }
#load_defect_sample:hover { background: #a8a6a6 !important; transform: translateY(-1px); box-shadow: 0 4px 8px rgba(194, 192, 192, 0.4) !important; }
.hidden { display: none !important; }
#draggable-prediction {
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
}
#draggable-prediction .card-header {
    border-radius: 16px 16px 0 0 !important;
}
.shap-plot-card {
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    padding: 16px;
    margin-bottom: 16px;
}
.shap-plot-card h4 {
    margin: 0 0 12px 0;
    font-size: 16px;
    font-weight: 600;
    color: #142D4A;
}
.shap-plot-card img {
    width: 100%;
    height: auto;
    border-radius: 12px;
}
.shap-plot-empty {
    text-align: center;
    color: #6c757d;
    font-size: 13px;
    padding: 24px 12px;
}</style>
<script>
function toggleAnalysisAccordion(evt, id) {
    if (evt) {
        evt.preventDefault();
        evt.stopPropagation();
    }
    const panel = document.getElementById(id);
    if (!panel) {
        return;
    }
    const isHidden = window.getComputedStyle(panel).display === 'none';
    panel.style.display = isHidden ? 'block' : 'none';
}
function togglePredictionCard() { document.getElementById('draggable-prediction').classList.toggle('hidden'); }
let isDragging = false, currentX, currentY, initialX, initialY, xOffset = 0, yOffset = 0;
document.addEventListener('DOMContentLoaded', function() {
    const drag = document.getElementById('draggable-prediction');
    if (drag) {
        const header = drag.querySelector('.card-header');
        if (header) {
            header.addEventListener('mousedown', e => { initialX = e.clientX - xOffset; initialY = e.clientY - yOffset; isDragging = true; });
            document.addEventListener('mousemove', e => { if (isDragging) { e.preventDefault(); currentX = e.clientX - initialX; currentY = e.clientY - initialY; xOffset = currentX; yOffset = currentY; drag.style.transform = `translate(${currentX}px, ${currentY}px)`; }});
            document.addEventListener('mouseup', () => { initialX = currentX; initialY = currentY; isDragging = false; });
        }
    }
    const btn = document.getElementById('settings-button');
    if (btn) {
        btn.addEventListener('click', togglePredictionCard);
        btn.addEventListener('mouseenter', function() { this.style.transform = 'rotate(90deg)'; });
        btn.addEventListener('mouseleave', function() { this.style.transform = 'rotate(0deg)'; });
    }
});
</script>
"""

def create_input_metadata():
    metadata = {}
    for col in categorical_cols:
        values = sorted([str(v) for v in df[col].dropna().unique()])
        if values:
            metadata[col] = {"type": "categorical", "choices": values, "default": values[0]}
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        vmin, vmax, vdef = float(s.min()), float(s.max()), float(s.median())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        step = max(1, round((vmax - vmin) / 200.0)) if (s.round() == s).all() else (vmax - vmin) / 200.0
        metadata[col] = {"type": "numeric", "min": vmin, "max": vmax, "value": vdef, "step": step}
    return metadata

input_metadata = create_input_metadata()

def create_widgets(cols, is_categorical=False):
    widgets = []
    for col in cols:
        if col not in input_metadata:
            continue
        meta = input_metadata[col]
        label = COLUMN_NAMES_KR.get(col, col)
        if is_categorical:
            widgets.append(ui.div(ui.input_select(col, label, choices=meta["choices"], selected=meta["default"]), class_="input-item"))
        else:
            widgets.append(ui.div(ui.input_slider(col, label, min=meta["min"], max=meta["max"], value=meta["value"], step=meta["step"]), class_="input-item"))
    return widgets

def panel_body():
    cat_widgets = create_widgets(categorical_cols, True)
    num_widgets = create_widgets(numeric_cols, False)

    cat_rows = [ui.layout_columns(*cat_widgets[i:i+4], col_widths=[3, 3, 3, 3]) for i in range(0, len(cat_widgets), 4)]
    num_rows = [ui.layout_columns(*num_widgets[i:i+4], col_widths=[3, 3, 3, 3]) for i in range(0, len(num_widgets), 4)]

    return ui.page_fluid(
        ui.HTML(custom_css),
        ui.div(
            ui.div(
                ui.div("예측 결과", class_="card-header", style="background: #2A2D30; color: white; padding: 16px 20px; border-radius: 16px 16px 0 0; font-weight: 600; cursor: move;"),
                ui.div(
                    ui.div(ui.output_ui("prediction_result"), style="margin-bottom: 12px; text-align: center; font-size: 24px; font-weight: 700;"),
                    ui.div(ui.output_ui("probability_text"), style="margin-bottom: 12px; text-align: center; font-size: 14px; color: #495057;"),
                    ui.div(ui.output_ui("shap_top_features"), style="margin-bottom: 12px; font-size: 13px; color: #495057;"),
                    ui.div(ui.output_ui("recommendation_text"), style="margin-bottom: 16px; font-size: 13px; color: #495057;"),
                    ui.input_action_button("predict", "불량 여부 예측", style="width: 100%; background: #dc3545; color: white; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600; margin-bottom: 10px;"),
                    ui.input_action_button("load_defect_sample", "불량 샘플 랜덤 추출", style="width: 100%; background: #C2C0C0; color: #2c3e50; border: none; border-radius: 10px; padding: 12px 24px; font-weight: 600;"),
                    style="background: white; padding: 20px; border-radius: 0 0 16px 16px;"
                )
            ),
            id="draggable-prediction",
            style="position: fixed; bottom: 20px; right: 100px; width: 320px; z-index: 1000;"
        ),
        ui.HTML('<div id="settings-button" style="position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px; background: #2A2D30; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; box-shadow: 0 4px 12px rgba(0,0,0,0.2); z-index: 1000; transition: transform 0.2s;"><svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg></div>'),
        ui.div(
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("변수 설정", style="font-size: 16px;"),
                        ui.span("접기", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAnalysisAccordion(event, 'variables_content')",
                    class_="accordion-header",
                    type="button"
                ),
                ui.div(*cat_rows, *num_rows, id="variables_content", class_="accordion-content", style="display: block;"),
                class_="accordion-section",
                style="margin-bottom: 16px;"
            ),
            ui.div(
                ui.tags.button(
                    ui.div(
                        ui.span("불량 샘플", style="font-size: 16px;"),
                        ui.span("접기", style="font-size: 12px;"),
                        style="display: flex; justify-content: space-between; width: 100%;"
                    ),
                    onclick="toggleAnalysisAccordion(event, 'defect_sample_content')",
                    class_="accordion-header",
                    type="button"
                ),
                ui.div(
                    ui.output_ui("defect_sample_table"),
                    ui.layout_columns(
                        ui.div(ui.output_plot("shap_force_plot"), class_="shap-plot-card"),
                        ui.div(ui.output_plot("shap_waterfall_plot"), class_="shap-plot-card"),
                        col_widths=[6, 6]
                    ),
                    id="defect_sample_content",
                    class_="accordion-content",
                    style="display: block;"
                ),
                class_="accordion-section"
            ),
            style="padding: 24px; max-width: 1400px; margin: 0 auto;"
        )
    )

def panel():
    return ui.nav_panel("예측 분석", panel_body())
def server(input, output, session):
    prediction_state = reactive.Value(None)
    active_sample = reactive.Value(None)
    show_defect_samples = reactive.Value(False)

    def _current_shap_explanation():
        details = prediction_state.get()
        if not details or details.get("error"):
            return None
        contributions = details.get("shap_aggregated")
        input_row = details.get("input_row")
        if not contributions or input_row is None:
            return None
        return build_shap_explanation(contributions, input_row)

    def _shap_placeholder(message="예측 결과를 먼저 계산하세요.", figsize=(6, 2.5)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#6c757d")
        return fig

    @reactive.effect
    @reactive.event(input.load_defect_sample)
    def _load_samples():
        try:
            if "passorfail" not in df.columns:
                prediction_state.set({"error": "passorfail 컬럼이 없어 불량 샘플을 불러올 수 없습니다."})
                return
            defect = df[df["passorfail"] == 1].copy()
            if defect.empty:
                prediction_state.set({"error": "불량 샘플을 찾을 수 없습니다."})
                show_defect_samples.set(False)
                return
            sample_row = defect.sample(n=1).iloc[0]
            active_sample.set(sample_row)
            show_defect_samples.set(True)
            for col in all_cols:
                if col not in input_metadata:
                    continue
                val = sample_row.get(col)
                if pd.isna(val):
                    continue
                meta = input_metadata[col]
                if meta["type"] == "categorical":
                    value_str = str(val)
                    if value_str not in meta["choices"]:
                        continue
                    session.send_input_message(col, {"value": value_str})
                else:
                    numeric_val = float(val)
                    numeric_val = max(meta["min"], min(meta["max"], numeric_val))
                    session.send_input_message(col, {"value": numeric_val})
            prediction_state.set(None)
        except Exception as exc:
            prediction_state.set({"error": f"불량 샘플 적용 중 오류: {exc}"})

    @output
    @render.ui
    def defect_sample_table():
        if not show_defect_samples.get():
            return ui.div("불량 샘플 랜덤 추출 버튼을 눌러주세요.", style="text-align: center; color: #6c757d; padding: 20px;")
        sample_row = active_sample.get()
        if sample_row is None:
            return ui.div("불량 샘플 정보가 로드되지 않았습니다.", style="text-align: center; color: #6c757d; padding: 20px;")
        columns_to_display = [col for col in all_cols + ["passorfail"] if col in df.columns]
        row_df = pd.DataFrame([sample_row])[columns_to_display]
        html = '<div style="overflow-x: auto;"><table style="width: 100%; border-collapse: collapse; font-size: 12px;"><thead style="background: #2A2D30; color: white;"><tr>'
        for col in row_df.columns:
            html += f'<th style="padding: 10px; text-align: left; border: 1px solid #dee2e6; white-space: nowrap;">{col}</th>'
        html += '</tr></thead><tbody><tr style="background: white;">'
        for col in row_df.columns:
            value = row_df.iloc[0][col]
            display_value = '-' if pd.isna(value) else str(value)
            html += f'<td style="padding: 8px; border: 1px solid #dee2e6; white-space: nowrap;">{display_value}</td>'
        html += '</tr>'
        html += '</tbody></table></div>'
        return ui.HTML(html)

    @reactive.effect
    @reactive.event(input.predict)
    def _run_prediction():
        try:
            input_row = {}
            for col in all_cols:
                meta = input_metadata.get(col)
                if not meta:
                    continue
                value = getattr(input, col)()
                if value is None:
                    continue
                if meta["type"] == "categorical":
                    input_row[col] = str(value)
                else:
                    input_row[col] = float(value)
            result = predict_with_model(input_row, compute_shap=True)
            aggregated = result.get("shap_aggregated", {})
            positive_items = [(feat, contrib) for feat, contrib in aggregated.items() if contrib > 0]
            if positive_items:
                positive_items.sort(key=lambda item: item[1], reverse=True)
                top_items = positive_items[:5]
            else:
                top_items = sorted(aggregated.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
            top_features = []
            for feat, contrib in top_items:
                top_features.append({
                    "name": feat,
                    "label": COLUMN_NAMES_KR.get(feat, feat),
                    "value": input_row.get(feat, "-"),
                    "contribution": contrib
                })
            recommendations = recommend_ranges(input_row, [item["name"] for item in top_features])
            result["top_features"] = top_features
            result["recommendations"] = recommendations
            result["input_row"] = input_row
            prediction_state.set(result)
        except Exception as exc:
            prediction_state.set({"error": str(exc)})

    @output
    @render.ui
    def prediction_result():
        details = prediction_state.get()
        if not details:
            return ui.div("불량 여부 예측 버튼을 눌러 결과를 확인하세요.", style="font-size: 14px; font-weight: 500; color: #6c757d;")
        if "error" in details:
            return ui.div(f"오류 발생: {details['error']}", style="font-size: 16px; font-weight: 600; color: #dc3545;")
        label = "불량" if details["prediction"] == 1 else "정상"
        color = "#dc3545" if label == "불량" else "#28a745"
        note = " (tryshot_signal 규칙 적용)" if details.get("forced_fail") else ""
        return ui.div(f"{label}{note}", style=f"color: {color};")

    @output
    @render.ui
    def probability_text():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div("불량 확률은 예측 후 확인할 수 있습니다.")
        prob = details.get("probability", 0.0)
        text = f"불량 확률: {prob:.4f} (임계값 {operating_threshold:.4f})"
        if details.get("forced_fail"):
            text += " - 규칙에 의해 불량으로 판정됨"
        return ui.div(text)

    @output
    @render.ui
    def shap_top_features():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div()
        top_features = details.get("top_features", [])
        if not top_features:
            return ui.div("SHAP 상위 기여 요인을 확인하려면 예측을 실행하세요.")
        header_text = "불량 영향 상위 변수 (최대 5개)"
        items = []
        for item in top_features:
            value_text = format_value(item.get("value", "-"))
            contrib = item.get("contribution", 0.0)
            direction = "위험 증가" if contrib > 0 else "위험 감소"
            items.append(f"<li><strong>{item['label']}</strong> (현재 {value_text}) - SHAP {contrib:+.4f} ({direction})</li>")
        html = f"<div><span style='font-weight:600;'>{header_text}</span><ul style='padding-left:18px; margin:8px 0 0 0;'>{''.join(items)}</ul></div>"
        return ui.HTML(html)

    @output
    @render.plot
    def shap_force_plot():
        explanation = _current_shap_explanation()
        if explanation is None:
            return _shap_placeholder()
        try:
            order = np.argsort(np.abs(explanation.values))[::-1]
            if len(order) > 20:
                order = order[:20]
            focus_exp = explanation[order]
            plt.close("all")
            shap.force_plot(
                focus_exp.base_values,
                focus_exp.values,
                focus_exp.data,
                feature_names=focus_exp.feature_names,
                matplotlib=True,
                show=False,
            )
            fig = plt.gcf()
            fig.set_size_inches(6, 2.8)
            fig.tight_layout()
            return fig
        except Exception:
            return _shap_placeholder("SHAP force plot을 생성하는 중 문제가 발생했습니다.")

    @output
    @render.plot
    def shap_waterfall_plot():
        explanation = _current_shap_explanation()
        if explanation is None:
            return _shap_placeholder(figsize=(6, 4.0))
        try:
            plt.close("all")
            shap.plots.waterfall(explanation, max_display=15, show=False)
            fig = plt.gcf()
            fig.set_size_inches(6, 4.5)
            fig.tight_layout()
            return fig
        except Exception:
            return _shap_placeholder("SHAP waterfall plot을 생성하는 중 문제가 발생했습니다.", figsize=(6, 4.0))

    @output
    @render.ui
    def recommendation_text():
        details = prediction_state.get()
        if not details or "error" in details:
            return ui.div()
        recommendations = details.get("recommendations", {})
        if not recommendations:
            return ui.div("추천 정상 구간을 찾지 못했습니다. 다른 변수 조합을 시도해보세요.")
        best_prob = recommendations.get("best_probability")
        lines = []
        for feat, info in recommendations.items():
            if feat == "best_probability":
                continue
            label = COLUMN_NAMES_KR.get(feat, feat)
            if info.get("type") == "numeric":
                min_val = format_value(info.get("min"))
                max_val = format_value(info.get("max"))
                examples = ", ".join(format_value(v) for v in info.get("examples", []))
                lines.append(f"<li><strong>{label}</strong>: 추천 정상 범위 {min_val} ~ {max_val} (예시: {examples})</li>")
            elif info.get("type") == "categorical":
                values = ", ".join(info.get("values", []))
                lines.append(f"<li><strong>{label}</strong>: 추천 정상 값 {values}</li>")
        summary = "<ul style='padding-left:18px; margin:8px 0 0 0;'>" + "".join(lines) + "</ul>"
        if best_prob is not None:
            summary += f"<div style='margin-top:8px;'>추천 조합 적용 시 예상 불량 확률 ~ {best_prob:.4f}</div>"
        return ui.HTML("<div><span style='font-weight:600;'>정상 전환 추천 구간</span>" + summary + "</div>")


app = App(panel, server)



