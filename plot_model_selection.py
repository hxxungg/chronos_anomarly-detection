"""Plots used to select Tukey-IQR preprocessing and present final model metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_SPECS = [
    ("r2", "R²", "tab:green"),
    ("rmse", "RMSE", "tab:red"),
    ("accuracy", "Accuracy (%)", "tab:blue"),
    ("mape", "MAPE (%)", "tab:orange"),
    ("coverage", "Coverage (%)", "tab:purple"),
]


def configure_korean_font() -> str | None:
    candidates = ["AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Nanum Gothic"]
    available = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in available), None)
    if selected is not None:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def _prepare(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["variable"] = (
        frame["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    if "label" not in frame:
        frame["label"] = frame["variable"]
    frame["label"] = frame["label"].astype(str).str.replace("\n", " ", regex=False)
    return frame.drop_duplicates(subset=["variable"], keep="first")


def _variable_order(reference: pd.DataFrame) -> list[str]:
    return (
        reference.sort_values("r2", ascending=False, na_position="last")["variable"]
        .dropna()
        .tolist()
    )


def _metric_limits(frames: Iterable[pd.DataFrame]) -> dict[str, tuple[float, float]]:
    frames = list(frames)
    limits: dict[str, tuple[float, float]] = {}
    for column, _, _ in METRIC_SPECS:
        if column == "r2":
            # R² is unbounded below. Values below -2 are shown at the lower display edge.
            limits[column] = (-2.0, 1.05)
            continue
        values = pd.concat(
            [pd.to_numeric(frame[column], errors="coerce") for frame in frames],
            ignore_index=True,
        )
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        value_min, value_max = float(values.min()), float(values.max())
        span = value_max - value_min
        padding = max(0.10, span * 0.08, max(abs(value_min), abs(value_max), 1.0) * 0.03)
        lower, upper = value_min - padding, value_max + padding
        if column in {"rmse", "mape"}:
            lower = 0.0
        elif value_min >= 0 and value_max <= 100 and column in {"accuracy", "coverage"}:
            lower = max(0.0, lower)
            upper = min(100.5, max(upper, value_max + 0.5))
        limits[column] = (lower, upper)
    return limits


def _plot_values(
    ax: plt.Axes,
    x: np.ndarray,
    values: np.ndarray,
    column: str,
    color: str,
) -> None:
    if column != "r2":
        ax.plot(x, values, marker="o", markersize=3, linewidth=1.2, color=color)
        return
    display_min, display_max = -2.0, 1.05
    ax.plot(
        x, np.clip(values, display_min, display_max),
        marker="o", markersize=3, linewidth=1.2, color=color,
    )
    below = np.isfinite(values) & (values < display_min)
    if below.any():
        ax.scatter(
            x[below], np.full(below.sum(), display_min),
            marker="v", s=34, color=color, zorder=4,
        )
    ax.set_ylabel("R² (▼: 실제값 < -2)")


def plot_preprocessing_comparison(
    no_percentile: pd.DataFrame,
    percentile: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    metric_columns: tuple[str, ...] | None = None,
) -> list[plt.Figure]:
    """Create same-scale, two-panel plots: no masking versus masking."""
    configure_korean_font()
    no_frame, percentile_frame = _prepare(no_percentile), _prepare(percentile)
    order = _variable_order(no_frame)
    no_frame = no_frame.set_index("variable").reindex(order).reset_index()
    percentile_frame = percentile_frame.set_index("variable").reindex(order).reset_index()
    limits = _metric_limits([no_frame, percentile_frame])
    labels = no_frame["label"].fillna(no_frame["variable"])
    x = np.arange(len(order))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []

    selected_specs = [
        spec for spec in METRIC_SPECS
        if metric_columns is None or spec[0] in metric_columns
    ]
    for column, metric_label, color in selected_specs:
        fig, axes = plt.subplots(
            2, 1, figsize=(max(13, len(order) * 0.42), 9), sharex=True, sharey=True,
        )
        for ax, frame, preprocessing_label in [
            (axes[0], no_frame, "Tukey IQR 미적용"),
            (axes[1], percentile_frame, "Tukey 1.5×IQR 적용"),
        ]:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            _plot_values(ax, x, values, column, color)
            ax.set_ylim(*limits[column])
            ax.set_title(f"파인튜닝 후 · {preprocessing_label} · {equipment_id} · {metric_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=7)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.2)
            if column == "coverage":
                ax.axhline(98.0, color="black", linestyle="--", linewidth=1)
        fig.tight_layout(pad=2.0, h_pad=2.0)
        fig.savefig(
            output_dir / f"preprocessing_no_vs_percentile_{equipment_id}_{column}.png",
            dpi=180, bbox_inches="tight",
        )
        figures.append(fig)
    return figures


def plot_final_no_percentile_metric_files(
    summary: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    metric_columns: tuple[str, ...] = ("r2", "rmse", "mape", "coverage"),
) -> list[plt.Figure]:
    """Save one presentation-ready figure per final-model metric."""
    configure_korean_font()
    frame = _prepare(summary)
    order = _variable_order(frame)
    frame = frame.set_index("variable").reindex(order).reset_index()
    labels = frame["label"].fillna(frame["variable"])
    limits = _metric_limits([frame])
    x = np.arange(len(frame))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_specs = [spec for spec in METRIC_SPECS if spec[0] in metric_columns]
    figures = []

    for column, metric_label, color in selected_specs:
        fig, ax = plt.subplots(figsize=(max(13, len(frame) * 0.42), 5.5))
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        _plot_values(ax, x, values, column, color)
        ax.set_ylim(*limits[column])
        ax.set_title(
            f"파인튜닝 후 · Tukey IQR 미적용 · {equipment_id} · {metric_label}"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(alpha=0.2)
        if column == "coverage":
            ax.axhline(98.0, color="black", linestyle="--", linewidth=1)
        fig.tight_layout(pad=2.0)
        fig.savefig(
            output_dir / f"final_no_percentile_{equipment_id}_{column}.png",
            dpi=180, bbox_inches="tight",
        )
        figures.append(fig)
    return figures


def plot_final_no_percentile_metrics(
    summary: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
) -> tuple[plt.Figure, np.ndarray, dict[str, tuple[float, float]]]:
    """Plot the selected no-percentile model with y-scales fitted only to it."""
    configure_korean_font()
    frame = _prepare(summary)
    order = _variable_order(frame)
    frame = frame.set_index("variable").reindex(order).reset_index()
    labels = frame["label"].fillna(frame["variable"])
    limits = _metric_limits([frame])
    x = np.arange(len(frame))
    fig, axes = plt.subplots(5, 1, figsize=(max(13, len(frame) * 0.42), 19), sharex=True)

    for ax, (column, metric_label, color) in zip(axes, METRIC_SPECS):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        _plot_values(ax, x, values, column, color)
        ax.set_ylim(*limits[column])
        ax.set_title(f"최종 선택 · Full 파인튜닝 · Tukey IQR 미적용 · {equipment_id} · {metric_label}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(alpha=0.2)
    axes[4].axhline(98.0, color="black", linestyle="--", linewidth=1)
    fig.tight_layout(pad=2.0, h_pad=2.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"final_no_percentile_metrics_{equipment_id}.png",
        dpi=180, bbox_inches="tight",
    )
    return fig, axes, limits


def plot_process_metric_scorecard(
    metrics_summary: pd.DataFrame,
    naive_comparison: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    model_name: str = "Full FT · Tukey IQR 미적용",
    coverage_target: float = 98.0,
    coverage_tolerance: float = 2.0,
    mape_limit: float = 10.0,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """각 변수의 예측 성능을 한 장의 스코어카드로 저장한다.

    색상은 변수 간 단위 차이를 줄이기 위해 RMSE 절대값 대신
    Naive 대비 RMSE 개선률을 사용한다. 셀 안의 RMSE 숫자는
    원래 단위의 Chronos RMSE이며, 괄호는 Naive 대비 감소율이다.
    """
    configure_korean_font()
    metrics = _prepare(metrics_summary)
    metrics = metrics.loc[metrics["equipment_id"].astype(str).eq(str(equipment_id))].copy()
    if metrics.empty:
        raise ValueError(f"{equipment_id}: 최종 평가 지표가 없습니다.")

    comparison = naive_comparison.copy()
    comparison["variable"] = (
        comparison["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    comparison = comparison.loc[
        comparison["equipment_id"].astype(str).eq(str(equipment_id))
        & comparison["model"].astype(str).eq(model_name)
    ].copy()
    if comparison.empty:
        raise ValueError(f"{equipment_id}: '{model_name}' Naive 비교 결과가 없습니다.")

    metric_columns = ["variable", "label", "coverage"]
    frame = comparison.merge(
        metrics[metric_columns], on="variable", how="inner", validate="one_to_one",
    )
    if frame.empty:
        raise ValueError(f"{equipment_id}: 평가 결과와 Naive 비교 변수가 일치하지 않습니다.")

    for column in [
        "chronos_r2", "naive_r2", "chronos_rmse", "naive_rmse",
        "chronos_mape", "coverage",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["r2_gain"] = frame["chronos_r2"] - frame["naive_r2"]
    frame["rmse_improvement_pct"] = np.where(
        frame["naive_rmse"].abs() > 1e-12,
        (frame["naive_rmse"] - frame["chronos_rmse"]) / frame["naive_rmse"] * 100.0,
        np.nan,
    )
    frame["coverage_error"] = (frame["coverage"] - coverage_target).abs()

    def classify(row: pd.Series) -> str:
        r2 = row["chronos_r2"]
        gain = row["r2_gain"]
        if not np.isfinite(r2) or not np.isfinite(gain):
            return "판정 불가"
        if gain <= 0:
            return "Naive 충분"
        auxiliary_warning = (
            (np.isfinite(row["chronos_mape"]) and row["chronos_mape"] > mape_limit)
            or (
                np.isfinite(row["coverage_error"])
                and row["coverage_error"] > coverage_tolerance
            )
        )
        if r2 >= 0.7:
            return "조건부 우수" if auxiliary_warning else "우수"
        if r2 >= 0.3:
            return "조건부 활용" if auxiliary_warning else "활용 가능"
        if r2 >= 0:
            return "제한적"
        return "예측 부족"

    frame["판정"] = frame.apply(classify, axis=1)
    judgement_order = {
        "우수": 0,
        "조건부 우수": 1,
        "Naive 충분": 2,
        "활용 가능": 3,
        "조건부 활용": 4,
        "제한적": 5,
        "예측 부족": 6,
        "판정 불가": 7,
    }
    frame["_judgement_order"] = frame["판정"].map(judgement_order)
    frame = frame.sort_values(
        ["_judgement_order", "chronos_r2"], ascending=[True, False], na_position="last"
    ).reset_index(drop=True)

    def scale(values: pd.Series, xp: list[float], fp: list[float]) -> np.ndarray:
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        return np.where(np.isfinite(numeric), np.interp(numeric, xp, fp), 0.5)

    judgement_score = frame["판정"].map(
        {
            "우수": 1.0,
            "조건부 우수": 0.80,
            "Naive 충분": 0.68,
            "활용 가능": 0.76,
            "조건부 활용": 0.58,
            "제한적": 0.45,
            "예측 부족": 0.0,
            "판정 불가": 0.5,
        }
    ).to_numpy(dtype=float)
    scores = np.column_stack(
        [
            scale(frame["chronos_r2"], [-1.0, 0.0, 0.3, 0.7, 1.0], [0.0, 0.2, 0.48, 0.78, 1.0]),
            scale(frame["r2_gain"], [-0.2, 0.0, 0.05, 0.3], [0.0, 0.42, 0.72, 1.0]),
            scale(frame["rmse_improvement_pct"], [-25.0, 0.0, 10.0, 40.0], [0.0, 0.42, 0.72, 1.0]),
            scale(frame["chronos_mape"], [0.0, 5.0, 10.0, 20.0, 50.0], [1.0, 0.9, 0.72, 0.42, 0.0]),
            scale(frame["coverage_error"], [0.0, 1.0, 3.0, 8.0], [1.0, 0.9, 0.55, 0.0]),
            judgement_score,
        ]
    )

    def fmt(value: float, pattern: str = ".2f") -> str:
        return format(value, pattern) if np.isfinite(value) else "-"

    annotations: list[list[str]] = []
    for _, row in frame.iterrows():
        improvement = row["rmse_improvement_pct"]
        rmse_text = fmt(row["chronos_rmse"])
        if np.isfinite(improvement):
            rmse_text += f"\n({improvement:+.1f}%)"
        annotations.append(
            [
                fmt(row["chronos_r2"]),
                fmt(row["r2_gain"], "+.2f"),
                rmse_text,
                f"{fmt(row['chronos_mape'], '.1f')}%",
                f"{fmt(row['coverage'], '.1f')}%",
                str(row["판정"]),
            ]
        )

    n_rows = len(frame)
    fig_height = max(10.0, n_rows * 0.38 + 3.0)
    fig, ax = plt.subplots(figsize=(14.5, fig_height))
    ax.imshow(scores, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    column_labels = [
        "Chronos R²\n(↑)",
        "ΔR² vs Naive\n(↑)",
        "RMSE\n(Naive 대비 %)",
        "MAPE (%)\n(↓)",
        f"Coverage (%)\n(목표 {coverage_target:g})",
        "종합 판정",
    ]
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, fontsize=9)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(frame["label"].fillna(frame["variable"]), fontsize=8)

    for row_index in range(n_rows):
        for column_index in range(len(column_labels)):
            score = scores[row_index, column_index]
            text_color = "white" if score < 0.18 or score > 0.88 else "black"
            ax.text(
                column_index, row_index, annotations[row_index][column_index],
                ha="center", va="center", fontsize=7.5, color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, len(column_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        f"{equipment_id} · Full 파인튜닝 · Tukey IQR 미적용 · 변수별 예측 성능 스코어카드",
        fontsize=14, pad=48,
    )
    fig.text(
        0.5, 0.012,
        "초록=양호, 빨강=주의  ·  RMSE 괄호=동일 테스트의 Naive 대비 감소율  ·  "
        "MAPE는 실제값이 0에 가까운 변수에서 과대평가될 수 있음",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.98))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"process_metric_scorecard_{equipment_id}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    frame.drop(columns=["_judgement_order"]).to_csv(
        output_dir / f"process_metric_scorecard_{equipment_id}.csv",
        index=False, encoding="utf-8-sig",
    )
    return fig, ax, frame.drop(columns=["_judgement_order"])


def plot_process_metric_lines(
    metrics_summary: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    coverage_target: float = 98.0,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """R², RMSE, MAPE, Coverage를 한 장의 2×2 꺽은선 그래프로 표시한다."""
    configure_korean_font()
    frame = _prepare(metrics_summary)
    frame = frame.loc[
        frame["equipment_id"].astype(str).eq(str(equipment_id))
    ].copy()
    if frame.empty:
        raise ValueError(f"{equipment_id}: 최종 평가 지표가 없습니다.")

    order = _variable_order(frame)
    frame = frame.set_index("variable").reindex(order).reset_index()
    labels = frame["label"].fillna(frame["variable"])
    x = np.arange(len(frame))
    limits = _metric_limits([frame])
    plot_specs = [
        ("r2", "R² (↑ 높을수록 좋음)", "tab:green"),
        ("rmse", "RMSE (↓ 낮을수록 좋음)", "tab:red"),
        ("mape", "MAPE (%) (↓ 낮을수록 좋음)", "tab:orange"),
        ("coverage", f"Coverage (%) (목표 {coverage_target:g}%)", "tab:purple"),
    ]

    # PPT 한 장에 넣기 쉬운 16:10 비율로 고정한다. 저장 해상도는 아래에서
    # 300 DPI로 높여, 그림 크기를 줄여 배치해도 글자와 선이 선명하게 보이게 한다.
    figure_width = 16.0
    fig, axes = plt.subplots(
        2, 2, figsize=(figure_width, 10.0), sharex=True,
        gridspec_kw={"hspace": 0.88, "wspace": 0.18},
    )
    axes_flat = axes.ravel()
    for index, (ax, (column, title, color)) in enumerate(zip(axes_flat, plot_specs)):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        _plot_values(ax, x, values, column, color)
        ax.set_ylim(*limits[column])
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", alpha=0.08)
        if column == "coverage":
            ax.axhline(
                coverage_target, color="black", linestyle="--", linewidth=1.2,
                label=f"목표 {coverage_target:g}%",
            )
            ax.legend(loc="best", fontsize=7)
        # 네 그래프에 동일한 변수 순서와 변수명을 모두 표시한다.
        ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
        ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        f"{equipment_id} · Full 파인튜닝 · Tukey IQR 미적용 · 변수별 예측 성능",
        fontsize=13, y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "변수는 R² 높은 순으로 정렬 · R²·RMSE·MAPE·Coverage는 서로 단위가 달라 각 패널의 축을 따로 사용",
        ha="center", fontsize=7,
    )
    fig.subplots_adjust(bottom=0.17, top=0.94, left=0.055, right=0.99)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = output_dir / f"process_metric_lines_{equipment_id}"
    fig.savefig(
        output_stem.with_suffix(".png"),
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    # PowerPoint에서는 SVG를 사용하면 확대·축소해도 선과 글자가 깨지지 않는다.
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches="tight", facecolor="white",
    )
    return fig, axes, frame


def plot_process_metric_overlay(
    metrics_summary: pd.DataFrame,
    naive_comparison: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    model_name: str = "Full FT · Tukey IQR 미적용",
    coverage_target: float = 98.0,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """네 지표를 공통 0~100 표시점수로 변환해 한 축에 겹쳐 그린다.

    점수는 PPT용 비교 표시값이며 새로운 평가지표가 아니다.
    - R²: 0 이하=0점, 1=100점
    - RMSE: Naive와 같으면 50점, 50% 감소=100점, 50% 증가=0점
    - MAPE: 0%=100점, 50% 이상=0점
    - Coverage: 98%와 같으면 100점, 8%p 이상 차이=0점
    """
    configure_korean_font()
    metrics = _prepare(metrics_summary)
    metrics = metrics.loc[
        metrics["equipment_id"].astype(str).eq(str(equipment_id))
    ].copy()
    if metrics.empty:
        raise ValueError(f"{equipment_id}: 최종 평가 지표가 없습니다.")

    comparison = naive_comparison.copy()
    comparison["variable"] = (
        comparison["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    comparison = comparison.loc[
        comparison["equipment_id"].astype(str).eq(str(equipment_id))
        & comparison["model"].astype(str).eq(model_name)
    ].copy()
    if comparison.empty:
        raise ValueError(f"{equipment_id}: '{model_name}' Naive 비교 결과가 없습니다.")

    frame = comparison.merge(
        metrics[["variable", "label", "coverage"]],
        on="variable", how="inner", validate="one_to_one",
    )
    for column in [
        "chronos_r2", "chronos_rmse", "naive_rmse", "chronos_mape", "coverage",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("chronos_r2", ascending=False, na_position="last").reset_index(drop=True)

    frame["r2_score"] = np.clip(frame["chronos_r2"], 0.0, 1.0) * 100.0
    rmse_ratio = np.where(
        frame["naive_rmse"].abs() > 1e-12,
        frame["chronos_rmse"] / frame["naive_rmse"],
        np.nan,
    )
    frame["rmse_score"] = np.clip(50.0 + 100.0 * (1.0 - rmse_ratio), 0.0, 100.0)
    frame["mape_score"] = np.clip(
        100.0 * (1.0 - frame["chronos_mape"] / 50.0), 0.0, 100.0
    )
    frame["coverage_score"] = np.clip(
        100.0 * (1.0 - (frame["coverage"] - coverage_target).abs() / 8.0),
        0.0, 100.0,
    )

    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(max(18.0, len(frame) * 0.5), 8.5))
    series = [
        ("r2_score", "R²", "tab:green", "o"),
        ("rmse_score", "RMSE (Naive 대비)", "tab:red", "s"),
        ("mape_score", "MAPE", "tab:orange", "^"),
        ("coverage_score", f"Coverage (목표 {coverage_target:g}%)", "tab:purple", "D"),
    ]
    for column, label, color, marker in series:
        ax.plot(
            x, frame[column], color=color, marker=marker, markersize=4,
            linewidth=1.5, label=label,
        )

    ax.set_ylim(-3, 103)
    ax.set_ylabel("표시용 상대 성능점수 (0~100, 높을수록 양호)")
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"].fillna(frame["variable"]), rotation=90, fontsize=7)
    ax.set_title(
        f"{equipment_id} · Full 파인튜닝 · Tukey IQR 미적용 · 4개 지표 통합 비교",
        fontsize=15,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", alpha=0.08)
    ax.legend(loc="lower left", ncol=4, fontsize=9)
    fig.text(
        0.5, 0.01,
        "한 축에 겹치기 위한 표시점수: R² 원값·Naive 대비 RMSE·MAPE·Coverage 98% 근접도를 0~100으로 변환",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.04, 1.0, 0.98))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"process_metric_overlay_{equipment_id}.png",
        dpi=200, bbox_inches="tight",
    )
    frame.to_csv(
        output_dir / f"process_metric_overlay_{equipment_id}.csv",
        index=False, encoding="utf-8-sig",
    )
    return fig, ax, frame


def plot_final_r2_vs_lag1_acf(
    prediction_csv: str | Path,
    metrics_summary: pd.DataFrame,
    equipment_id: str,
    output_dir: str | Path = "model_comparison",
    display_min: float = -2.0,
    display_max: float = 1.05,
    show_relationship_panel: bool = True,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame, dict[str, float]]:
    """Compare final-model R² with lag-1 ACF on the same no-percentile test period."""
    configure_korean_font()
    prediction_csv = Path(prediction_csv)
    predictions = pd.read_csv(
        prediction_csv,
        usecols=[
            "equipment_id", "variable", "timestamp", "actual", "excluded_reason",
        ],
    )
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions["variable"] = (
        predictions["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    predictions = predictions.loc[predictions["equipment_id"].eq(equipment_id)].copy()
    predictions = predictions.sort_values(["variable", "timestamp"]).reset_index(drop=True)
    grouped = predictions.groupby("variable", sort=False)
    predictions["previous_actual"] = grouped["actual"].shift(1)
    predictions["previous_timestamp"] = grouped["timestamp"].shift(1)

    acf_rows = []
    for variable, group in predictions.groupby("variable", sort=False):
        current = pd.to_numeric(group["actual"], errors="coerce")
        previous = pd.to_numeric(group["previous_actual"], errors="coerce")
        valid = (
            group["excluded_reason"].isna()
            & current.notna()
            & previous.notna()
            & group["timestamp"].sub(group["previous_timestamp"]).eq(pd.Timedelta(hours=1))
        )
        n_pairs = int(valid.sum())
        acf = float(current[valid].corr(previous[valid])) if n_pairs >= 2 else np.nan
        acf_rows.append({"variable": variable, "acf_lag1": acf, "acf_pairs": n_pairs})

    metrics = _prepare(metrics_summary)
    comparison = metrics[["variable", "label", "r2"]].merge(
        pd.DataFrame(acf_rows), on="variable", how="inner", validate="one_to_one",
    )
    comparison = comparison.dropna(subset=["r2", "acf_lag1"]).sort_values(
        "acf_lag1", ascending=False
    ).reset_index(drop=True)
    if comparison.empty:
        raise ValueError(f"{equipment_id}: R²와 자기상관을 함께 계산할 변수가 없습니다.")

    correlations = {
        "pearson_raw": float(comparison["acf_lag1"].corr(comparison["r2"])),
        "rank_correlation": float(
            comparison["acf_lag1"].rank().corr(comparison["r2"].rank())
        ),
    }
    x = np.arange(len(comparison))
    r2_raw = comparison["r2"].to_numpy(dtype=float)
    r2_display = np.clip(r2_raw, display_min, display_max)
    acf = comparison["acf_lag1"].to_numpy(dtype=float)

    if show_relationship_panel:
        fig, axes = plt.subplots(
            2, 1, figsize=(max(13, len(comparison) * 0.42), 11)
        )
    else:
        fig, line_ax = plt.subplots(
            1, 1, figsize=(max(13, len(comparison) * 0.42), 6)
        )
        axes = np.atleast_1d(line_ax)
    axes[0].plot(x, acf, marker="o", markersize=4, linewidth=1.3,
                 color="tab:gray", label="자기상관 ACF(1)")
    axes[0].plot(x, r2_display, marker="o", markersize=4, linewidth=1.3,
                 color="tab:blue", label="Full FT R²")
    below = r2_raw < display_min
    if below.any():
        axes[0].scatter(x[below], np.full(below.sum(), display_min), marker="v", s=38,
                        color="tab:blue", zorder=4, label="실제 R² < -2")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylim(display_min, display_max)
    axes[0].set_ylabel("계수 값")
    axes[0].set_title(
        f"{equipment_id} · Full 파인튜닝 · Tukey IQR 미적용 · R²와 lag-1 자기상관"
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(comparison["label"], rotation=90, fontsize=7)
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(alpha=0.2)

    if show_relationship_panel:
        axes[1].scatter(acf, r2_display, s=28, alpha=0.75, color="tab:blue")
        if below.any():
            axes[1].scatter(acf[below], r2_display[below], marker="v", s=50,
                            color="tab:blue", zorder=4)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_xlim(-1.05, 1.05)
        axes[1].set_ylim(display_min, display_max)
        axes[1].set_xlabel("lag-1 자기상관 ACF(1)")
        axes[1].set_ylabel("R² 표시값 (▼: 실제값 < -2)")
        axes[1].set_title(
            f"변수별 관계 · 순위상관={correlations['rank_correlation']:.3f}"
        )
        axes[1].grid(alpha=0.2)
        fig.tight_layout(pad=2.0, h_pad=2.5)
    else:
        fig.tight_layout(pad=2.0)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / f"final_no_percentile_r2_vs_acf_{equipment_id}.csv",
        index=False, encoding="utf-8-sig",
    )
    fig.savefig(
        output_dir / f"final_no_percentile_r2_vs_acf_{equipment_id}.png",
        dpi=180, bbox_inches="tight",
    )
    return fig, axes, comparison, correlations


def plot_lag1_repeat_evidence(
    prediction_csv: str | Path,
    metrics_summary: pd.DataFrame,
    equipment_id: str = "2CM",
    output_dir: str | Path = "model_comparison",
    variable: str | None = None,
    window_hours: int = 96,
    equality_atol: float = 1e-9,
) -> tuple[plt.Figure, np.ndarray, pd.DataFrame, str]:
    """Visualize how repeated hourly actuals support high lag-1 ACF and R².

    The repeat rate, ACF, and R² are calculated on the same valid test pairs.
    When ``variable`` is omitted, the function selects the variable with the
    strongest combined percentile ranks for all three measures.
    """
    configure_korean_font()
    predictions = pd.read_csv(
        Path(prediction_csv),
        usecols=[
            "equipment_id", "variable", "timestamp", "actual", "excluded_reason",
        ],
    )
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"])
    predictions["variable"] = (
        predictions["variable"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    predictions = predictions.loc[
        predictions["equipment_id"].eq(equipment_id)
    ].sort_values(["variable", "timestamp"]).reset_index(drop=True)
    if predictions.empty:
        raise ValueError(f"{equipment_id}: 테스트 실제값을 찾지 못했습니다.")

    grouped = predictions.groupby("variable", sort=False)
    predictions["previous_actual"] = grouped["actual"].shift(1)
    predictions["previous_timestamp"] = grouped["timestamp"].shift(1)
    predictions["valid_lag1_pair"] = (
        predictions["excluded_reason"].isna()
        & pd.to_numeric(predictions["actual"], errors="coerce").notna()
        & pd.to_numeric(predictions["previous_actual"], errors="coerce").notna()
        & predictions["timestamp"].sub(predictions["previous_timestamp"]).eq(
            pd.Timedelta(hours=1)
        )
    )
    predictions["same_as_previous"] = False
    valid_pairs = predictions["valid_lag1_pair"]
    predictions.loc[valid_pairs, "same_as_previous"] = np.isclose(
        pd.to_numeric(predictions.loc[valid_pairs, "actual"], errors="coerce"),
        pd.to_numeric(predictions.loc[valid_pairs, "previous_actual"], errors="coerce"),
        rtol=0.0,
        atol=equality_atol,
    )

    rows: list[dict[str, float | int | str]] = []
    for variable_name, group in predictions.groupby("variable", sort=False):
        pair_group = group.loc[group["valid_lag1_pair"]]
        current = pd.to_numeric(pair_group["actual"], errors="coerce")
        previous = pd.to_numeric(pair_group["previous_actual"], errors="coerce")
        n_pairs = len(pair_group)
        rows.append(
            {
                "variable": variable_name,
                "repeat_rate": (
                    float(pair_group["same_as_previous"].mean()) if n_pairs else np.nan
                ),
                "acf_lag1": (
                    float(current.corr(previous)) if n_pairs >= 2 else np.nan
                ),
                "n_pairs": int(n_pairs),
            }
        )

    metrics = _prepare(metrics_summary)
    evidence = metrics[["variable", "label", "r2"]].merge(
        pd.DataFrame(rows), on="variable", how="inner", validate="one_to_one",
    )
    evidence = evidence.dropna(subset=["r2", "repeat_rate", "acf_lag1"]).copy()
    if evidence.empty:
        raise ValueError(f"{equipment_id}: 반복률·자기상관·R²를 함께 계산할 변수가 없습니다.")
    evidence["repeat_percent"] = evidence["repeat_rate"] * 100.0
    evidence["evidence_score"] = evidence[
        ["repeat_rate", "acf_lag1", "r2"]
    ].rank(pct=True).mean(axis=1)
    evidence = evidence.sort_values(
        ["evidence_score", "r2"], ascending=False
    ).reset_index(drop=True)

    if variable is None:
        selected_variable = str(evidence.iloc[0]["variable"])
    else:
        selected_variable = str(variable).replace("\n", " ").strip()
        if selected_variable not in set(evidence["variable"]):
            raise ValueError(f"{equipment_id}: '{selected_variable}' 변수를 찾지 못했습니다.")
    selected = evidence.loc[evidence["variable"].eq(selected_variable)].iloc[0]
    selected_label = str(selected["label"])

    series = predictions.loc[predictions["variable"].eq(selected_variable)].copy()
    pair_series = series.loc[series["valid_lag1_pair"]].copy()
    changed = pair_series.loc[~pair_series["same_as_previous"]]
    if not changed.empty:
        center = changed.iloc[len(changed) // 2]["timestamp"]
    else:
        center = series.iloc[len(series) // 2]["timestamp"]
    half_window = pd.Timedelta(hours=max(2, window_hours // 2))
    window = series.loc[
        series["timestamp"].between(center - half_window, center + half_window)
    ].copy()
    if window.empty:
        window = series.tail(window_hours).copy()

    fig, axes = plt.subplots(
        1, 3, figsize=(17, 5.5),
        gridspec_kw={"width_ratios": [2.2, 1.45, 0.9]},
    )

    # 1) A short actual-value window makes the repeated plateaus visible.
    axes[0].plot(
        window["timestamp"], window["actual"], color="tab:blue",
        linewidth=1.5, marker="o", markersize=2.5, label="시간별 실제값",
    )
    repeated_window = window.loc[window["valid_lag1_pair"] & window["same_as_previous"]]
    changed_window = window.loc[window["valid_lag1_pair"] & ~window["same_as_previous"]]
    axes[0].scatter(
        repeated_window["timestamp"], repeated_window["actual"],
        s=20, color="tab:orange", label="직전 1시간과 동일", zorder=3,
    )
    axes[0].scatter(
        changed_window["timestamp"], changed_window["actual"],
        s=34, marker="x", color="tab:red", label="직전값에서 변화", zorder=4,
    )
    axes[0].set_title(f"실제값 반복 구간 ({window_hours}시간 예시)")
    axes[0].set_xlabel("시간")
    axes[0].set_ylabel("실제값")
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # 2) Exact repetitions lie on y_t = y_(t-1).
    scatter_data = pair_series
    if len(scatter_data) > 3000:
        scatter_data = scatter_data.sample(3000, random_state=42)
    axes[1].scatter(
        scatter_data["previous_actual"], scatter_data["actual"],
        s=12, alpha=0.35, color="tab:blue",
    )
    bounds = pd.concat(
        [scatter_data["previous_actual"], scatter_data["actual"]],
        ignore_index=True,
    )
    lower, upper = float(bounds.min()), float(bounds.max())
    padding = max((upper - lower) * 0.05, 1e-6)
    axes[1].plot(
        [lower - padding, upper + padding], [lower - padding, upper + padding],
        linestyle="--", linewidth=1.2, color="tab:red", label=r"$y_t=y_{t-1}$",
    )
    axes[1].set_xlim(lower - padding, upper + padding)
    axes[1].set_ylim(lower - padding, upper + padding)
    axes[1].set_xlabel("직전 1시간 실제값 $y_{t-1}$")
    axes[1].set_ylabel("현재 실제값 $y_t$")
    axes[1].set_title("직전값과 현재값의 관계")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    # 3) Put the three supporting quantities on one common 0-1 scale.
    bar_labels = ["동일 반복률", "ACF(1)", "R²"]
    bar_values = [
        float(selected["repeat_rate"]),
        float(selected["acf_lag1"]),
        float(selected["r2"]),
    ]
    bars = axes[2].bar(
        bar_labels, bar_values,
        color=["tab:orange", "tab:gray", "tab:blue"], width=0.65,
    )
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel("비율 또는 계수")
    axes[2].set_title("대표 변수의 근거 지표")
    axes[2].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, bar_values):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.02),
            f"{value:.3f}", ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle(
        f"{equipment_id} 대표 사례 · 직전값 반복과 높은 자기상관·예측 성능 — {selected_label}",
        fontsize=14,
    )
    fig.text(
        0.5, 0.01,
        "※ 동일한 유효 테스트 구간에서 계산한 대표 사례이며, 반복률만으로 모든 변수의 R²를 설명하지는 않습니다.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.93], w_pad=2.2)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(
        output_dir / f"lag1_repeat_evidence_{equipment_id}.csv",
        index=False, encoding="utf-8-sig",
    )
    fig.savefig(
        output_dir / f"lag1_repeat_evidence_{equipment_id}.png",
        dpi=180, bbox_inches="tight",
    )
    return fig, axes, evidence, selected_variable
