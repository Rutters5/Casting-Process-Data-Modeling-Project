from __future__ import annotations

import math
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path
from typing import NamedTuple

try:
	import joblib
except ImportError as exc:
	raise SystemExit("joblib 패키지가 필요합니다. 'pip install joblib' 명령으로 설치 후 다시 실행해주세요.") from exc

import numpy as np
import pandas as pd

try:
	import matplotlib
	matplotlib.use("Agg")
	import matplotlib.pyplot as plt
	from matplotlib import font_manager
except ImportError as exc:
	raise SystemExit("matplotlib 패키지가 필요합니다. 'pip install matplotlib' 명령으로 설치 후 다시 실행해주세요.") from exc

# Configure Korean font for plots (fallback to default if unavailable)
try:
	plt.rcParams["font.family"] = "Malgun Gothic"
	plt.rcParams["axes.unicode_minus"] = False
except Exception:
	available_fonts = [f.name for f in font_manager.fontManager.ttflist]
	for candidate in ["Malgun Gothic", "NanumGothic", "Nanum Gothic", "AppleGothic"]:
		if candidate in available_fonts:
			plt.rcParams["font.family"] = candidate
			plt.rcParams["axes.unicode_minus"] = False
			break

from sklearn.inspection import PartialDependenceDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
MODELS_DIR = APP_DIR / "data" / "models"
PNG_DIR = APP_DIR / "data" / "png"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = "passorfail"
DROP_COLUMNS = ["date", "time", "Unnamed: 0"]
MAX_SAMPLE_SIZE = 1000
GRID_RESOLUTION = 40


class PreprocessInfo(NamedTuple):
	scaler: object | None
	numeric_cols: list[str]


def _list_version_dirs() -> list[Path]:
	if not MODELS_DIR.exists():
		raise FileNotFoundError(f"모델 디렉터리를 찾을 수 없습니다: {MODELS_DIR}")
	return sorted([path for path in MODELS_DIR.iterdir() if path.is_dir()])


def _load_train_features(version: str) -> pd.DataFrame:
	train_path = DATA_DIR / f"train_{version}.csv"
	if not train_path.exists():
		raise FileNotFoundError(f"학습 데이터가 존재하지 않습니다: {train_path}")

	df = pd.read_csv(train_path)
	df = df.drop(columns=DROP_COLUMNS, errors="ignore")

	if TARGET_COLUMN in df.columns:
		df = df.drop(columns=[TARGET_COLUMN])

	df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
	if df.empty:
		raise ValueError(f"결측치 제거 후 남은 데이터가 없습니다: {train_path}")

	if len(df) > MAX_SAMPLE_SIZE:
		df = df.sample(n=MAX_SAMPLE_SIZE, random_state=42)

	return df.reset_index(drop=True)


def _load_model_artifact(model_path: Path) -> dict | object:
	try:
		artifact = joblib.load(model_path)
	except Exception as exc:
		raise RuntimeError(f"모델을 불러오는 중 오류가 발생했습니다: {model_path.name}") from exc

	if isinstance(artifact, dict):
		model_aliases = [
			"model",
			"model_clf",
			"model_reg",
			"estimator",
			"classifier",
			"regressor",
			"clf",
		]
		if artifact.get("model") is None:
			for key in model_aliases:
				if key in artifact and artifact[key] is not None:
					artifact["model"] = artifact[key]
					break
		return artifact
	return {"model": artifact}


def _get_feature_names(artifact: dict | None) -> list[str]:
	if not artifact:
		return []

	names: list[str] = []

	scaler = artifact.get("scaler") if isinstance(artifact, dict) else None
	if scaler is not None and hasattr(scaler, "feature_names_in_"):
		names.extend(list(scaler.feature_names_in_))

	ohe = artifact.get("onehot_encoder") if isinstance(artifact, dict) else None
	if ohe is not None:
		if hasattr(ohe, "get_feature_names_out"):
			try:
				names.extend(list(ohe.get_feature_names_out()))
			except Exception:
				pass
		if not names:
			try:
				input_features = getattr(ohe, "feature_names_in_", None)
				categories = getattr(ohe, "categories_", None)
				if input_features is not None and categories is not None:
					for base_name, cats in zip(input_features, categories):
						for cat in cats:
							names.append(f"{base_name}={cat}")
			except Exception:
				pass

	if not names:
		model = artifact.get("model") if isinstance(artifact, dict) else artifact
		if hasattr(model, "feature_name"):
			try:
				names = list(model.feature_name())
			except Exception:
				names = []

	return names


def _extract_importance_values(model) -> np.ndarray | None:
	if model is None:
		return None

	if hasattr(model, "feature_importance"):
		try:
			return np.array(model.feature_importance(importance_type="gain"))
		except TypeError:
			return np.array(model.feature_importance())
	if hasattr(model, "feature_importances_"):
		return np.array(model.feature_importances_)
	if hasattr(model, "coef_"):
		coefs = model.coef_
		if isinstance(coefs, np.ndarray):
			return np.mean(np.abs(coefs), axis=0).ravel()
	return None


def _compute_importances(model, primary_names: list[str], fallback_names: list[str] | None = None) -> pd.DataFrame:
	importances = _extract_importance_values(model)
	if importances is None or importances.size == 0:
		return pd.DataFrame()

	feature_names = primary_names or []
	if not feature_names or len(feature_names) != importances.size:
		if fallback_names is not None and len(fallback_names) == importances.size:
			feature_names = list(fallback_names)
		else:
			feature_names = [f"feature_{i}" for i in range(importances.size)]

	df = pd.DataFrame({"feature": feature_names, "importance": importances})
	df = df.sort_values("importance", ascending=False, ignore_index=True)
	total = df["importance"].sum()
	df["normalized"] = df["importance"] / total if total > 0 else df["importance"]
	return df


def _get_ohe_feature_names(encoder, base_names: list[str]) -> list[str]:
	if encoder is None:
		return []
	if hasattr(encoder, "get_feature_names_out"):
		return list(encoder.get_feature_names_out(base_names))
	if hasattr(encoder, "get_feature_names"):
		return list(encoder.get_feature_names(base_names))

	categories = getattr(encoder, "categories_", None)
	feature_names: list[str] = []
	if categories is not None:
		for base_name, cats in zip(base_names, categories):
			for cat in cats:
				feature_names.append(f"{base_name}={cat}")
	return feature_names


def _prepare_features(df: pd.DataFrame, artifact: dict) -> tuple[pd.DataFrame, list[str], PreprocessInfo]:
	scaler = artifact.get("scaler")
	ordinal = artifact.get("ordinal_encoder")
	ohe = artifact.get("onehot_encoder")

	if scaler is not None and hasattr(scaler, "feature_names_in_"):
		numeric_cols = list(scaler.feature_names_in_)
	else:
		numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

	if ordinal is not None and hasattr(ordinal, "feature_names_in_"):
		cat_cols = list(ordinal.feature_names_in_)
	else:
		cat_cols = [col for col in df.columns if col not in numeric_cols]

	missing_numeric = [col for col in numeric_cols if col not in df.columns]
	missing_categorical = [col for col in cat_cols if col not in df.columns]

	for col in missing_numeric:
		df[col] = 0.0

	for col in missing_categorical:
		df[col] = "missing"

	matrices = []
	feature_names: list[str] = []

	if numeric_cols:
		numeric_data = df[numeric_cols]
		if scaler is not None:
			numeric_array = scaler.transform(numeric_data)
		else:
			numeric_array = numeric_data.to_numpy(dtype=float)
		matrices.append(np.asarray(numeric_array, dtype=float))
		feature_names.extend(numeric_cols)

	if cat_cols and ordinal is not None and ohe is not None:
		cat_data = df[cat_cols].astype(object)
		cat_ord = ordinal.transform(cat_data)
		cat_ohe = ohe.transform(cat_ord)
		matrices.append(np.asarray(cat_ohe, dtype=float))
		feature_names.extend(_get_ohe_feature_names(ohe, cat_cols))

	if not matrices:
		raise ValueError("전처리 후 사용할 수 있는 특성이 없습니다.")

	combined = matrices[0] if len(matrices) == 1 else np.hstack(matrices)
	feature_df = pd.DataFrame(combined, columns=feature_names)
	preprocess_info = PreprocessInfo(scaler=scaler, numeric_cols=list(numeric_cols))
	return feature_df, feature_names, preprocess_info


class LightGBMBoosterWrapper:
	def __init__(self, booster, n_features: int):
		self.booster = booster
		self.n_features_in_ = n_features
		self.classes_ = np.array([0, 1])
		self._is_fitted = True
		self._estimator_type = "classifier"

	def fit(self, X, y=None):  # type: ignore[override]
		self._is_fitted = True
		return self

	def __sklearn_is_fitted__(self):
		return self._is_fitted

	def predict(self, X):
		proba = self.predict_proba(X)[:, 1]
		return (proba >= 0.5).astype(int)

	def predict_proba(self, X):
		preds = self.booster.predict(X, num_iteration=getattr(self.booster, "best_iteration", None))
		preds = np.asarray(preds)
		if preds.ndim == 1:
			preds = np.column_stack([1 - preds, preds])
		return preds


def _build_estimator(artifact: dict, n_features: int):
	model = artifact.get("model")
	if model is None:
		raise ValueError("모델 객체가 아티팩트에 없습니다.")

	if model.__class__.__module__.startswith("lightgbm") and not hasattr(model, "predict_proba"):
		return LightGBMBoosterWrapper(model, n_features)
	return model


def _resolve_top_features(artifact: dict, processed_feature_names: list[str], top_k: int = 5) -> list[str]:
	feature_names = _get_feature_names(artifact)
	model = artifact.get("model")
	importance_df = _compute_importances(model, feature_names, processed_feature_names)
	if importance_df.empty:
		return []
	return importance_df.head(top_k)["feature"].tolist()


def _filter_available_features(features: list[str], available: list[str]) -> list[str]:
	available_set = set(available)
	filtered: list[str] = []
	for name in features:
		if name in available_set:
			filtered.append(name)
	return filtered


def _inverse_scale_axis_ticks(ax, feature: str, preprocess_info: PreprocessInfo) -> None:
	scaler = preprocess_info.scaler
	numeric_cols = preprocess_info.numeric_cols
	if scaler is None or not numeric_cols or feature not in numeric_cols:
		return

	try:
		feature_idx = numeric_cols.index(feature)
	except ValueError:
		return

	ticks = np.asarray(ax.get_xticks(), dtype=float)
	if ticks.size == 0:
		return

	base = np.zeros((ticks.size, len(numeric_cols)))
	base[:, feature_idx] = ticks
	try:
		original_values = scaler.inverse_transform(base)[:, feature_idx]
	except Exception:
		return

	formatted = [f"{val:.3g}" for val in original_values]
	ax.set_xticks(ticks)
	ax.set_xticklabels(formatted)


def _apply_inverse_scaling(display: PartialDependenceDisplay, features: list[str], preprocess_info: PreprocessInfo) -> None:
	if preprocess_info.scaler is None or not preprocess_info.numeric_cols:
		return

	for ax, feature in zip(display.axes_.ravel(), features):
		_inverse_scale_axis_ticks(ax, feature, preprocess_info)


def _style_display(
	display: PartialDependenceDisplay,
	x_label_fontsize: int = 22,
	tick_label_fontsize: int = 16,
) -> None:
	for ax in display.axes_.ravel():
		ax.set_ylabel("")
		xlabel = ax.get_xlabel()
		if xlabel:
			ax.set_xlabel(xlabel, fontsize=x_label_fontsize)
		for line in ax.get_lines():
			line.set_color("red")
			line.set_linewidth(2.0)
		ax.tick_params(axis="both", labelsize=tick_label_fontsize)


def _generate_pdp(artifact: dict, version: str, model_name: str, features: list[str], feature_df: pd.DataFrame, preprocess_info: PreprocessInfo, output_path: Path) -> None:
	if not features:
		print(f"    [경고] 사용 가능한 상위 특성이 없어 PDP를 건너뜁니다.")
		return

	estimator = _build_estimator(artifact, n_features=feature_df.shape[1])

	n_features = len(features)
	n_cols = 2 if n_features > 1 else 1
	n_rows = math.ceil(n_features / n_cols)
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
	axes_flat = axes.flatten()

	for extra_ax in axes_flat[n_features:]:
		extra_ax.remove()

	target_axes = axes_flat[:n_features]

	try:
		display = PartialDependenceDisplay.from_estimator(
			estimator,
			feature_df,
			features=features,
			kind="average",
			grid_resolution=GRID_RESOLUTION,
			ax=target_axes,
			feature_names=feature_df.columns,
		)
		_apply_inverse_scaling(display, features, preprocess_info)
		_style_display(display)
	except Exception as exc:
		plt.close(fig)
		print(f"    [오류] PDP 생성에 실패했습니다: {exc}")
		return

	fig.tight_layout()

	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=150)
	plt.close(fig)
	print(f"    [완료] {output_path.relative_to(PROJECT_ROOT)} 저장")


def main() -> None:
	print(f"프로젝트 루트: {PROJECT_ROOT}")
	version_dirs = _list_version_dirs()

	for version_dir in version_dirs:
		version = version_dir.name
		print(f"\n[버전 처리 중] {version}")

		try:
			raw_df = _load_train_features(version)
		except Exception as exc:
			print(f"  [경고] 학습 데이터 로드 실패: {exc}")
			continue

		for model_path in sorted(version_dir.glob("*.pkl")):
			model_name = model_path.stem.replace(f"_{version}", "")
			print(f"  - 모델: {model_name}")

			try:
				artifact = _load_model_artifact(model_path)
			except Exception as exc:
				print(f"    [오류] 모델 로드 실패: {exc}")
				continue

			try:
				processed_df, processed_names, preprocess_info = _prepare_features(raw_df, artifact)
			except Exception as exc:
				print(f"    [오류] 전처리 실패: {exc}")
				continue

			top_features = _resolve_top_features(artifact, processed_names, top_k=5)
			filtered_features = _filter_available_features(top_features, processed_names)

			if not filtered_features:
				print("    [경고] 사용 가능한 상위 특성이 없어 건너뜀")
				continue

			output_dir = PNG_DIR / version
			output_path = output_dir / f"{model_name}_{version}_PDP.png"
			_generate_pdp(artifact, version, model_name, filtered_features, processed_df, preprocess_info, output_path)


if __name__ == "__main__":
	main()
