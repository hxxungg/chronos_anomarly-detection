"""Saved Chronos test predictions versus a one-hour persistence baseline.

Naive prediction: y_hat(t, v) = y(t-1, v)
Every Chronos/Naive pair is scored on exactly the same valid test timestamps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd


DEFAULT_PREDICTION_FILES = {
    "Zero-shot · 공통 원본 입력": Path(
        "checkpoints/review_v2_process_pred1/all_process_full_pred1/"
        "test_predictions_zeroshot.csv"
    ),
    "Full FT · Tukey IQR 미적용": Path(
        "checkpoints/review_v2_process_pred1/all_process_full_pred1/"
        "test_predictions_full.csv"
    ),
    "Full FT · Tukey IQR 적용": Path(
        "checkpoints/review_v2_process_pred1/all_process_full_pred1_tukey_iqr_trainfit/"
        "test_predictions_full_iqr_common_actual.csv"
    ),
}


def configure_korean_font() -> str | None:
    """Use an installed Korean font so exported figures keep readable labels."""
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Nanum Gothic"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)
    if selected is not None:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


configure_korean_font()


def _point_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = actual - prediction
    abs_error = np.abs(error)
    nonzero = np.abs(actual) > 1e-6
    ss_total = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "r2": 1.0 - float(np.sum(error**2)) / ss_total if ss_total > 0 else np.nan,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(abs_error)),
        "mape": (
            float(np.mean(abs_error[nonzero] / np.abs(actual[nonzero])) * 100)
            if nonzero.any()
            else np.nan
        ),
    }


def compare_prediction_file(path: str | Path, model_name: str) -> pd.DataFrame:
    """Compare one saved Chronos result with lag-1 Naive on common valid rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"예측 결과 CSV가 없습니다: {path}")

    columns = [
        "equipment_id", "variable", "timestamp", "actual", "pred_median",
        "excluded_reason",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["variable"] = (
        frame["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    keys = ["equipment_id", "variable"]
    frame = frame.sort_values([*keys, "timestamp"]).reset_index(drop=True)
    grouped = frame.groupby(keys, sort=False)
    frame["naive_prediction"] = grouped["actual"].shift(1)
    frame["previous_timestamp"] = grouped["timestamp"].shift(1)

    rows = []
    for (equipment_id, variable), group in frame.groupby(keys, sort=False):
        actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
        chronos = pd.to_numeric(group["pred_median"], errors="coerce").to_numpy(dtype=float)
        naive = pd.to_numeric(group["naive_prediction"], errors="coerce").to_numpy(dtype=float)
        consecutive = group["timestamp"].sub(group["previous_timestamp"]).eq(pd.Timedelta(hours=1))
        valid = (
            group["excluded_reason"].isna().to_numpy()
            & consecutive.to_numpy()
            & np.isfinite(actual)
            & np.isfinite(chronos)
            & np.isfinite(naive)
        )
        if valid.sum() < 2:
            continue

        chronos_metrics = _point_metrics(actual[valid], chronos[valid])
        naive_metrics = _point_metrics(actual[valid], naive[valid])
        row = {
            "model": model_name,
            "equipment_id": equipment_id,
            "variable": variable,
            "n_scored": int(valid.sum()),
        }
        for metric in ["r2", "rmse", "mae", "mape"]:
            row[f"chronos_{metric}"] = chronos_metrics[metric]
            row[f"naive_{metric}"] = naive_metrics[metric]
        row["r2_gain_vs_naive"] = chronos_metrics["r2"] - naive_metrics["r2"]
        row["rmse_ratio_vs_naive"] = chronos_metrics["rmse"] / naive_metrics["rmse"]
        row["mae_ratio_vs_naive"] = chronos_metrics["mae"] / naive_metrics["mae"]
        row["chronos_better_r2"] = chronos_metrics["r2"] > naive_metrics["r2"]
        row["chronos_better_rmse"] = chronos_metrics["rmse"] < naive_metrics["rmse"]
        row["chronos_better_mae"] = chronos_metrics["mae"] < naive_metrics["mae"]
        row["chronos_better_mape"] = chronos_metrics["mape"] < naive_metrics["mape"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_comparison(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, group in detail.groupby("model", sort=False):
        rows.append({
            "model": model_name,
            "variables": len(group),
            "chronos_r2_mean": group["chronos_r2"].mean(),
            "naive_r2_mean": group["naive_r2"].mean(),
            "chronos_r2_median": group["chronos_r2"].median(),
            "naive_r2_median": group["naive_r2"].median(),
            "chronos_rmse_mean": group["chronos_rmse"].mean(),
            "naive_rmse_mean": group["naive_rmse"].mean(),
            "chronos_mae_mean": group["chronos_mae"].mean(),
            "naive_mae_mean": group["naive_mae"].mean(),
            "chronos_mape_mean": group["chronos_mape"].mean(),
            "naive_mape_mean": group["naive_mape"].mean(),
            "chronos_better_r2": int(group["chronos_better_r2"].sum()),
            "chronos_better_rmse": int(group["chronos_better_rmse"].sum()),
            "chronos_better_mae": int(group["chronos_better_mae"].sum()),
            "chronos_better_mape": int(group["chronos_better_mape"].sum()),
        })
    return pd.DataFrame(rows)


def summarize_by_equipment(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model_name, equipment_id), group in detail.groupby(
        ["model", "equipment_id"], sort=False
    ):
        rows.append({
            "model": model_name,
            "equipment_id": equipment_id,
            "variables": len(group),
            "chronos_r2_mean": group["chronos_r2"].mean(),
            "naive_r2_mean": group["naive_r2"].mean(),
            "chronos_r2_median": group["chronos_r2"].median(),
            "naive_r2_median": group["naive_r2"].median(),
            "chronos_better_r2": int(group["chronos_better_r2"].sum()),
            "chronos_better_rmse": int(group["chronos_better_rmse"].sum()),
            "chronos_better_mae": int(group["chronos_better_mae"].sum()),
            "chronos_better_mape": int(group["chronos_better_mape"].sum()),
        })
    return pd.DataFrame(rows)


def plot_win_counts(summary: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    metrics = ["R²", "RMSE", "MAE", "MAPE"]
    columns = [
        "chronos_better_r2", "chronos_better_rmse",
        "chronos_better_mae", "chronos_better_mape",
    ]
    x = np.arange(len(summary))
    width = 0.19
    fig, ax = plt.subplots(figsize=(14, 6))
    for index, (metric, column) in enumerate(zip(metrics, columns)):
        offset = (index - 1.5) * width
        bars = ax.bar(x + offset, summary[column], width=width, label=metric)
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.axhline(summary["variables"].max() / 2, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["model"], rotation=12, ha="right")
    ax.set_ylabel("Chronos가 Naive보다 우수한 변수 수")
    ax.set_title("동일 Test 시점에서 Chronos와 1시간 지속 Naive 비교")
    ax.legend(ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig, ax


def plot_r2_by_variable(
    detail: pd.DataFrame,
    equipment_id: str = "2CM",
    model_names: tuple[str, ...] = (
        "Full FT · Tukey IQR 미적용",
        "Full FT · Tukey IQR 적용",
    ),
    display_min: float = -2.0,
    display_max: float = 1.05,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot R² on a readable display range without changing stored values."""
    selected = detail.loc[
        detail["equipment_id"].eq(equipment_id) & detail["model"].isin(model_names)
    ].copy()
    if selected.empty:
        raise ValueError(f"{equipment_id}: 표시할 비교 결과가 없습니다.")

    reference = selected.loc[selected["model"].eq(model_names[0])]
    variable_order = reference.sort_values(
        "chronos_r2", ascending=False, na_position="last"
    )["variable"].tolist()
    x = np.arange(len(variable_order))
    fig, axes = plt.subplots(len(model_names), 1, figsize=(max(13, len(x) * 0.42), 9), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, model_name in zip(axes, model_names):
        shown = (
            selected.loc[selected["model"].eq(model_name)]
            .set_index("variable")
            .reindex(variable_order)
        )
        chronos = shown["chronos_r2"].to_numpy(dtype=float)
        naive = shown["naive_r2"].to_numpy(dtype=float)
        ax.plot(x, np.clip(chronos, display_min, display_max), marker="o", markersize=3,
                linewidth=1.2, label="Chronos")
        ax.plot(x, np.clip(naive, display_min, display_max), marker="o", markersize=3,
                linewidth=1.2, label="Naive: 1시간 전 실제값")
        for values, color in [(chronos, "tab:blue"), (naive, "tab:orange")]:
            below = np.isfinite(values) & (values < display_min)
            if below.any():
                ax.scatter(x[below], np.full(below.sum(), display_min), marker="v", s=34,
                           color=color, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(display_min, display_max)
        ax.set_ylabel("R² (▼: 실제값 < -2)")
        ax.set_title(f"{equipment_id} · {model_name} · Chronos vs Naive")
        ax.grid(alpha=0.2)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(variable_order, rotation=90, fontsize=7)
    fig.tight_layout()
    return fig, axes


def run_comparison(
    prediction_files: Mapping[str, str | Path] = DEFAULT_PREDICTION_FILES,
    output_dir: str | Path = "model_comparison",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts = [
        compare_prediction_file(path, model_name)
        for model_name, path in prediction_files.items()
    ]
    detail = pd.concat(parts, ignore_index=True)
    summary = summarize_comparison(detail)
    by_equipment = summarize_by_equipment(detail)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output_dir / "chronos_vs_naive_by_variable.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output_dir / "chronos_vs_naive_summary.csv", index=False, encoding="utf-8-sig")
    by_equipment.to_csv(
        output_dir / "chronos_vs_naive_by_equipment.csv", index=False, encoding="utf-8-sig"
    )
    return detail, summary, by_equipment


if __name__ == "__main__":
    comparison_detail, comparison_summary, comparison_by_equipment = run_comparison()
    print("\n=== Chronos vs Naive 전체 변수 비교 ===")
    print(comparison_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== 설비별 비교 ===")
    print(comparison_by_equipment.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    plot_win_counts(comparison_summary)
    plt.show()
