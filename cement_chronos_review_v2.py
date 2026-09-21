"""GitHub Cement 프로젝트의 review_v2 원칙을 이 프로젝트 데이터에 적용한 공통 함수.

핵심 원칙
1. 통계적 IQR 제거와 앞뒤 시점 중앙값 치환을 하지 않는다.
2. 명백한 물리/설비 한계 위반값만 NaN으로 마스킹한다.
3. 1시간 간격으로 재색인하고 실제 시간 갭은 NaN으로 남긴다.
4. 설비별 시간순 70/15/15 분할을 사용한다.
5. POLYCOM 운전시간은 예측 대상이 아니라 과거 공변량과 정지 판정에만 사용한다.
"""

from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


PROTOCOL_VERSION = "review_v2_process_pred1"

ID_COL = "ID"
DATE_COL = "근무일자"
HOUR_COL = "근무시간"
DOWNTIME_COL = "POLYCOM\n운전시간"
PRODUCT_TYPE_COL = "기타\n품종"
REMARK_COL = "기타\n비고"
QUALITY_COLS = ["기타\n품질\nBLAINE", "기타\n품질\n44㎛R"]

# GitHub Cement_code_fin/config.py의 CLEANING_RULES를 통합 데이터의 한글 열에 대응시킨 값이다.
DAMPER_COLS = [
    "POLYCOM\nSEPOL\nFANDP'",
    "POLYCOM\n160 B/F\nFANDP'",
    "MILL\n186 B/F\nDP'",
    "MILL\nSEPOL\nFANDP'",
    "MILL\n183 B/F\nFANDP'",
]
BAG_FILTER_PRESSURE_COLS = [
    "POLYCOM\n160 B/F\n압력",
    "MILL\n186 B/F\n압력",
    "MILL\n183 B/F\n압력",
]

PHYSICAL_LIMITS = {
    "min_date": "2020-01-15",
    "required_product_type": "내수",
    "damper_min_pct": 0.0,
    "damper_max_pct": 100.0,
    "bag_filter_pressure_max": 0.0,
    "mill_feed_coarse_max": 10000.0,
    "blaine_min": 1000.0,
    "blaine_max": 10000.0,
    "residue_max": 20.0,
    "mill_out_gas_temp_max": 5000.0,
    "mill_out_material_temp_max": 5000.0,
    "roller_p2_max": 20000.0,
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _mask_rule(
    df: pd.DataFrame,
    column: str,
    mask: pd.Series,
    rule: str,
    records: list[dict],
) -> None:
    """한 물리 규칙을 적용하고 마스킹한 개수와 위치를 감사 기록에 남긴다."""
    mask = mask.fillna(False)
    if not mask.any():
        return
    for idx in df.index[mask]:
        records.append(
            {
                "source_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
                "timestamp": df.at[idx, "timestamp"],
                "variable": column,
                "original": df.at[idx, column],
                "rule": rule,
            }
        )
    df.loc[mask, column] = np.nan


def apply_physical_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """GitHub의 고정 물리/설비 규칙을 적용한다.

    반환값은 (정리된 데이터, 마스킹 감사 기록, 품종 필터로 제외한 행 수)이다.
    원본 값은 중앙값이나 경계값으로 바꾸지 않고 NaN으로 마스킹한다.
    """
    df = df.copy()
    records: list[dict] = []

    for col in DAMPER_COLS:
        values = _numeric(df[col])
        _mask_rule(
            df,
            col,
            (values < PHYSICAL_LIMITS["damper_min_pct"])
            | (values > PHYSICAL_LIMITS["damper_max_pct"]),
            "damper outside [0, 100]",
            records,
        )

    for col in BAG_FILTER_PRESSURE_COLS:
        values = _numeric(df[col])
        _mask_rule(
            df,
            col,
            values > PHYSICAL_LIMITS["bag_filter_pressure_max"],
            "bag-filter pressure > 0",
            records,
        )

    rules = [
        (
            "MILL\nFEED\n조분량",
            lambda x: x > PHYSICAL_LIMITS["mill_feed_coarse_max"],
            "coarse feed > 10000",
        ),
        (
            QUALITY_COLS[0],
            lambda x: (x < PHYSICAL_LIMITS["blaine_min"])
            | (x > PHYSICAL_LIMITS["blaine_max"]),
            "BLAINE outside [1000, 10000]",
        ),
        (
            QUALITY_COLS[1],
            lambda x: x > PHYSICAL_LIMITS["residue_max"],
            "residue > 20",
        ),
        (
            "MILL\nC/M출구온도\nGAS",
            lambda x: x > PHYSICAL_LIMITS["mill_out_gas_temp_max"],
            "mill outlet gas temperature > 5000",
        ),
        (
            "MILL\nC/M출구온도\n원료",
            lambda x: x > PHYSICAL_LIMITS["mill_out_material_temp_max"],
            "mill outlet material temperature > 5000",
        ),
        (
            "POLYCOM\nROLLER\nNO2",
            lambda x: x > PHYSICAL_LIMITS["roller_p2_max"],
            "roller pressure 2 > 20000",
        ),
    ]
    for col, predicate, label in rules:
        values = _numeric(df[col])
        _mask_rule(df, col, predicate(values), label, records)

    # 품종은 품질 측정 시점에만 기록된 경우가 있으므로 먼저 시간순 전방 채움한 뒤 내수만 남긴다.
    product_type = df[PRODUCT_TYPE_COL].replace(r"^\s*$", np.nan, regex=True)
    df[PRODUCT_TYPE_COL] = product_type.ffill()
    in_scope = df[PRODUCT_TYPE_COL].eq(PHYSICAL_LIMITS["required_product_type"])
    n_product_rows_dropped = int((~in_scope).sum())
    df = df.loc[in_scope].copy()

    audit = pd.DataFrame(
        records,
        columns=["source_index", "timestamp", "variable", "original", "rule"],
    )
    return df, audit, n_product_rows_dropped


def load_preprocessed_equipment(
    path: str | Path,
    sheet_name: str,
    equipment_id: str,
    *,
    header_row: int = 0,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame, dict]:
    """설비 하나를 GitHub 방식으로 정리하고 완전한 1시간 grid로 반환한다."""
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    required = {
        ID_COL,
        DATE_COL,
        HOUR_COL,
        DOWNTIME_COL,
        PRODUCT_TYPE_COL,
        REMARK_COL,
        *QUALITY_COLS,
        *DAMPER_COLS,
        *BAG_FILTER_PRESSURE_COLS,
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"전처리에 필요한 열이 없습니다: {missing}")

    df = df.loc[df[ID_COL].eq(equipment_id)].copy()
    if df.empty:
        raise ValueError(f"{equipment_id} 데이터가 없습니다.")

    df["timestamp"] = pd.to_datetime(df[DATE_COL], errors="coerce") + pd.to_timedelta(
        pd.to_numeric(df[HOUR_COL], errors="coerce"), unit="h"
    )
    n_bad_timestamp = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    df = df.loc[df["timestamp"] >= pd.Timestamp(PHYSICAL_LIMITS["min_date"])]
    df = df.sort_values("timestamp")
    n_duplicates = int(df["timestamp"].duplicated().sum())
    df = df.drop_duplicates("timestamp", keep="first")

    df, cleaning_audit, n_product_rows_dropped = apply_physical_cleaning(df)

    drop_cols = [ID_COL, DATE_COL, HOUR_COL, PRODUCT_TYPE_COL, REMARK_COL, *QUALITY_COLS, "timestamp"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric.index = pd.DatetimeIndex(df["timestamp"])
    numeric = numeric.sort_index()

    full_index = pd.date_range(numeric.index.min(), numeric.index.max(), freq="h")
    n_observed_rows = len(numeric)
    numeric = numeric.reindex(full_index)
    numeric.index.name = "timestamp"

    target_cols = [c for c in feature_cols if c != DOWNTIME_COL]
    data = numeric.reset_index(drop=True)
    time_index = pd.Series(numeric.index, name="timestamp").reset_index(drop=True)

    stats = {
        "equipment_id": equipment_id,
        "observed_rows": n_observed_rows,
        "hourly_rows": len(data),
        "inserted_gap_rows": len(data) - n_observed_rows,
        "bad_timestamp_rows": n_bad_timestamp,
        "duplicate_timestamp_rows": n_duplicates,
        "product_rows_dropped": n_product_rows_dropped,
        "physical_values_masked": len(cleaning_audit),
        "n_targets": len(target_cols),
    }
    return data, time_index, target_cols, cleaning_audit, stats


def equipment_frame(
    equipment_id: str,
    data: pd.DataFrame,
    time_index: pd.Series,
    target_cols: Sequence[str],
) -> pd.DataFrame:
    """Chronos-2 from_data_frame에 넣을 long-format 설비 프레임을 만든다."""
    frame = data[[*target_cols, DOWNTIME_COL]].copy()
    frame.insert(0, "timestamp", pd.to_datetime(time_index).to_numpy())
    frame.insert(0, "item_id", equipment_id)
    return frame


def validate_hourly_frame(df: pd.DataFrame) -> None:
    if df.empty or df[["item_id", "timestamp"]].isna().any().any():
        raise ValueError("빈 데이터이거나 item_id/timestamp가 비어 있습니다.")
    if df.duplicated(["item_id", "timestamp"]).any():
        raise ValueError("item_id/timestamp 중복이 있습니다.")
    for item, group in df.groupby("item_id", sort=False):
        group = group.sort_values("timestamp")
        if not group["timestamp"].diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
            raise ValueError(f"{item}: 1시간 간격 데이터가 아닙니다.")


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """각 설비를 시간순 70/15/15로 나눈다. 행을 섞지 않는다."""
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1:
        raise ValueError("train_frac와 val_frac가 올바르지 않습니다.")
    validate_hourly_frame(df)
    parts = ([], [], [])
    for _, group in df.sort_values(["item_id", "timestamp"]).groupby("item_id", sort=False):
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        parts[0].append(group.iloc[:n_train])
        parts[1].append(group.iloc[n_train : n_train + n_val])
        parts[2].append(group.iloc[n_train + n_val :])
    return tuple(pd.concat(part).reset_index(drop=True) for part in parts)


def fit_iqr_thresholds(
    train_df: pd.DataFrame,
    target_cols: Sequence[str],
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """학습 구간에서 설비·변수별 Tukey IQR 경계를 계산한다.

    Q1/Q3와 경계는 반드시 시간순으로 분리한 학습 데이터에서만 계산한다. 결측치는
    분위수 계산에서 제외하고, IQR이 0인 거의 일정한 변수는 이후 마스킹 대상에서
    제외한다. 이 동작은 GitHub Cement의 ``remove_iqr_outliers``와 동일한 규칙이다.
    """
    if multiplier <= 0:
        raise ValueError("IQR multiplier는 0보다 커야 합니다.")
    source_df = train_df
    validate_hourly_frame(source_df)
    rows: list[dict] = []
    for item_id, group in source_df.groupby("item_id", sort=False):
        for column in target_cols:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                q1 = q3 = iqr = low = high = np.nan
            else:
                q1, q3 = values.quantile([0.25, 0.75]).to_numpy(dtype=float)
                iqr = float(q3 - q1)
                low = float(q1 - multiplier * iqr)
                high = float(q3 + multiplier * iqr)
            rows.append(
                {
                    "item_id": item_id,
                    "variable": column,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "multiplier": multiplier,
                    "low_value": low,
                    "high_value": high,
                    "source_observed": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def apply_iqr_thresholds(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_cols: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """미리 계산한 설비·변수별 Tukey IQR 경계 밖의 값만 NaN으로 마스킹한다.

    반환값은 (마스킹된 데이터, 설비·변수별 마스킹 개수 요약)이다. 이 함수 안에서는
    경계를 다시 계산하지 않는다. IQR이 0이거나 유효한 경계가 없는 변수는 원본을
    그대로 유지한다.
    """
    result = df.copy()
    required = {"item_id", "variable", "iqr", "low_value", "high_value"}
    missing = required.difference(thresholds.columns)
    if missing:
        raise KeyError(f"IQR threshold 열이 없습니다: {sorted(missing)}")

    threshold_map = thresholds.set_index(["item_id", "variable"])[
        ["iqr", "low_value", "high_value"]
    ]
    rows: list[dict] = []
    for item_id, indices in result.groupby("item_id", sort=False).groups.items():
        for column in target_cols:
            key = (item_id, column)
            if key not in threshold_map.index:
                raise KeyError(f"IQR threshold가 없습니다: {key}")
            iqr, low, high = threshold_map.loc[key].to_numpy(dtype=float)
            values = pd.to_numeric(result.loc[indices, column], errors="coerce")
            if np.isfinite(iqr) and iqr > 0 and np.isfinite(low) and np.isfinite(high):
                mask = values.lt(low) | values.gt(high)
            else:
                mask = pd.Series(False, index=values.index)
            result.loc[mask.index[mask], column] = np.nan
            rows.append(
                {
                    "item_id": item_id,
                    "variable": column,
                    "iqr": iqr,
                    "low_value": low,
                    "high_value": high,
                    "masked": int(mask.sum()),
                }
            )
    return result, pd.DataFrame(rows)


def validation_windows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_cols: Sequence[str],
    prediction_length: int,
    context_length: int,
    windows_per_item: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """검증 구간 전체에 고르게 퍼진 결정론적 윈도우를 만든다.

    각 윈도우의 마지막 prediction_length 행에 예측 대상 실측이 하나 이상 있어야 한다.
    """
    if min(prediction_length, context_length) < 1 or windows_per_item < 0:
        raise ValueError("prediction/context/window 설정이 올바르지 않습니다.")
    history = pd.concat([train_df, val_df], ignore_index=True)
    history = history.sort_values(["item_id", "timestamp"]).reset_index(drop=True)
    validate_hourly_frame(history)

    frames: list[pd.DataFrame] = []
    rows: list[dict] = []
    for item, group in history.groupby("item_id", sort=False):
        group = group.reset_index(drop=True)
        val_group = val_df.loc[val_df["item_id"].eq(item)]
        if val_group.empty:
            raise ValueError(f"{item}: 검증 구간이 없습니다.")
        val_start, val_end = val_group["timestamp"].min(), val_group["timestamp"].max()
        observed = np.isfinite(group[list(target_cols)].to_numpy(dtype=float)).any(axis=1)
        in_validation = group["timestamp"].between(val_start, val_end).to_numpy()

        candidates: list[int] = []
        next_free = 0
        for start in np.flatnonzero(observed & in_validation):
            end = int(start) + prediction_length
            horizon_has_label = np.isfinite(
                group.loc[start : end - 1, list(target_cols)].to_numpy(dtype=float)
            ).any()
            if (
                start < next_free
                or start == 0
                or end > len(group)
                or group["timestamp"].iloc[end - 1] > val_end
                or not horizon_has_label
            ):
                continue
            candidates.append(int(start))
            next_free = end

        if not candidates:
            raise ValueError(f"{item}: 유효한 검증 윈도우가 없습니다.")
        if windows_per_item and len(candidates) > windows_per_item:
            selected = np.linspace(0, len(candidates) - 1, windows_per_item, dtype=int)
            candidates = [candidates[j] for j in selected]

        for number, start in enumerate(candidates):
            begin = max(0, start - context_length)
            end = start + prediction_length
            window = group.iloc[begin:end].copy()
            window_id = f"{item}__validation_{number:05d}"
            window["item_id"] = window_id
            observed_labels = int(
                np.isfinite(group.iloc[start:end][list(target_cols)].to_numpy(dtype=float)).sum()
            )
            frames.append(window)
            rows.append(
                {
                    "window_id": window_id,
                    "item_id": item,
                    "context_start": group["timestamp"].iloc[begin],
                    "context_end": group["timestamp"].iloc[start - 1],
                    "forecast_start": group["timestamp"].iloc[start],
                    "forecast_end": group["timestamp"].iloc[end - 1],
                    "context_rows": start - begin,
                    "observed_labels": observed_labels,
                }
            )

    if not frames:
        raise ValueError("검증 윈도우가 만들어지지 않았습니다.")
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def to_chronos_inputs(
    df: pd.DataFrame,
    target_cols: Sequence[str],
    prediction_length: int,
):
    """모든 공정 변수는 target, 운전시간은 past covariate로 변환한다."""
    from chronos.chronos2.preprocess import from_data_frame

    keep = ["item_id", "timestamp", *target_cols, DOWNTIME_COL]
    return from_data_frame(
        df=df[keep],
        target_columns=list(target_cols),
        prediction_length=prediction_length,
        id_column="item_id",
        timestamp_column="timestamp",
        use_target_encoding=False,
    )


def _prepared_input(
    target_context: np.ndarray,
    past_covariates: np.ndarray,
    prediction_length: int,
) -> dict:
    context = np.concatenate([target_context.T, past_covariates.T], axis=0).astype(np.float32)
    return {
        "context": torch.from_numpy(context),
        "future_covariates": torch.full(
            (context.shape[0], prediction_length), float("nan"), dtype=torch.float32
        ),
        "n_targets": int(target_context.shape[1]),
        "n_covariates": int(past_covariates.shape[1]),
        "n_future_covariates": 0,
    }


def rolling_forecast_multivariate(
    pipeline,
    data: pd.DataFrame,
    target_cols: Sequence[str],
    variable_scale: Sequence[float],
    is_down: np.ndarray,
    *,
    actual_data: pd.DataFrame | None = None,
    context_length: int,
    prediction_length: int = 1,
    stride: int = 1,
    batch_windows: int = 128,
    quantile_low: float = 0.01,
    quantile_high: float = 0.99,
    severity_eps: float = 1e-3,
) -> pd.DataFrame:
    """정리된 입력으로 rolling 예측하고 별도로 지정한 실측값으로 채점한다.

    ``actual_data``를 생략하면 ``data``로 입력과 채점을 모두 수행한다. 별도 데이터를 넘기면
    모델 입력과 실제 채점값의 시간축은 유지하면서 서로 다른 마스킹 정책을 적용할 수 있다.
    """
    if prediction_length != 1:
        raise ValueError("현재 review_v2 공정 파이프라인은 prediction_length=1로 고정합니다.")
    targets = data[list(target_cols)].to_numpy(dtype=np.float32)
    if actual_data is None:
        actual_targets = targets
    else:
        if len(actual_data) != len(data):
            raise ValueError("data와 actual_data의 행 수가 다릅니다.")
        actual_targets = actual_data[list(target_cols)].to_numpy(dtype=np.float32)
    covariates = data[[DOWNTIME_COL]].to_numpy(dtype=np.float32)
    n = len(data)
    positions = list(range(context_length, n, stride))
    if not positions:
        raise ValueError(f"데이터 길이({n})가 context_length({context_length})보다 짧습니다.")

    quantiles = list(pipeline.quantiles)
    nearest = lambda q: min(range(len(quantiles)), key=lambda k: abs(quantiles[k] - q))
    low_i, mid_i, high_i = nearest(quantile_low), nearest(0.5), nearest(quantile_high)
    scales = np.asarray(variable_scale, dtype=float)

    rows: list[dict] = []
    for batch_start in range(0, len(positions), batch_windows):
        batch_pos = positions[batch_start : batch_start + batch_windows]
        inputs = [
            _prepared_input(
                targets[i - context_length : i],
                covariates[i - context_length : i],
                prediction_length,
            )
            for i in batch_pos
        ]
        forecasts = pipeline.predict(
            inputs,
            prediction_length=prediction_length,
            batch_size=len(inputs),
            context_length=context_length,
        )

        for batch_index, target_index in enumerate(batch_pos):
            forecast = forecasts[batch_index].detach().to("cpu").float().numpy()
            for variable_index, column in enumerate(target_cols):
                actual = float(actual_targets[target_index, variable_index])
                q_low = float(forecast[variable_index, low_i, 0])
                q_mid = float(forecast[variable_index, mid_i, 0])
                q_high = float(forecast[variable_index, high_i, 0])
                if np.isnan(actual):
                    reason, error, severity, is_anomaly = "gap_or_cleaned", np.nan, np.nan, False
                else:
                    reason = "down" if bool(is_down[target_index]) else None
                    error = actual - q_mid
                    severity = abs(error) / max(float(scales[variable_index]), severity_eps)
                    is_anomaly = bool(actual < q_low or actual > q_high) if reason is None else False
                rows.append(
                    {
                        "idx": target_index,
                        "variable": column,
                        "actual": actual,
                        "pred_median": q_mid,
                        "pred_low": q_low,
                        "pred_high": q_high,
                        "error": error,
                        "is_anomaly": is_anomaly,
                        "severity": severity,
                        "excluded_reason": reason,
                    }
                )
    return pd.DataFrame(rows)


def make_loss_history_callback():
    from transformers.trainer_callback import TrainerCallback

    class LossHistoryCallback(TrainerCallback):
        def __init__(self):
            self.log_history: list[dict] = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self.log_history.append({**logs, "step": state.global_step})
            return control

    return LossHistoryCallback()


def loss_history_from_callback(callback) -> pd.DataFrame:
    train = {entry["step"]: entry["loss"] for entry in callback.log_history if "loss" in entry}
    val = {entry["step"]: entry["eval_loss"] for entry in callback.log_history if "eval_loss" in entry}
    steps = sorted(set(train) | set(val))
    return pd.DataFrame(
        {
            "step": steps,
            "train_loss": [train.get(step) for step in steps],
            "val_loss": [val.get(step) for step in steps],
        }
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_versions(packages: Iterable[str] | None = None) -> dict:
    packages = packages or ("chronos-forecasting", "torch", "transformers", "pandas", "numpy")
    result = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
