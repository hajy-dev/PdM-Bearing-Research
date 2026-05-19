"""공통 유틸: 데이터 로드, 4-fold 수동 2단 배분, 평가 리포트, 메타 저장."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# === 경로/상수 ===
DATA_CSV = Path(r"C:\Users\User\Desktop\git\finalproject\AIpassBackend\training_data.csv")
RESULTS_DIR = Path(r"D:\project\예지보전_v2\증명_3피처ML학습\results")
FEATURES = ["vibration", "temperature", "motor_current"]
RANDOM_STATE = 42
N_FOLDS = 4

ANOMALY_CLASSES = [0, 1]
ANOMALY_NAMES = {0: "NORMAL", 1: "ANOMALY"}
FAULT_CLASSES_3 = ["BEARING_FAULT", "MOTOR_FAULT", "COMPOUND_FAULT"]
FAULT_DROP_LABELS = {"MONITORING", "NORMAL"}


# === 1. 로드/검증 ===
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    required = {
        "equipment_id",
        "vibration",
        "temperature",
        "motor_current",
        "risk_level",
        "fault_type",
        "is_anomaly",
    }
    missing = required - set(df.columns)
    assert not missing, f"누락 컬럼: {missing}"
    assert df[FEATURES].isnull().sum().sum() == 0, "피처 결측치 존재"
    assert df["equipment_id"].nunique() == 24, f"장비 24대 아님: {df['equipment_id'].nunique()}"

    # is_anomaly가 bool/str/string 혼재 가능성 → 정수화
    s = df["is_anomaly"]
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
        df["is_anomaly_int"] = s.astype(int)
    else:
        df["is_anomaly_int"] = (
            s.astype(str)
            .str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0, "t": 1, "f": 0})
            .astype(int)
        )
    return df


def filter_fault_data(df: pd.DataFrame) -> pd.DataFrame:
    """fault 모델 학습용 필터. NORMAL + MONITORING 제외."""
    df_f = df[df["fault_type"].isin(FAULT_CLASSES_3)].reset_index(drop=True)
    assert len(df_f) > 0
    assert set(df_f["fault_type"].unique()) == set(FAULT_CLASSES_3)
    return df_f


def verify_anomaly_label_consistency(df: pd.DataFrame) -> dict:
    rule_based = (df["risk_level"] != "LOW").astype(int)
    csv_based = df["is_anomaly_int"]
    mismatch = int((rule_based != csv_based).sum())
    return {
        "mismatch_count": mismatch,
        "total": int(len(df)),
        "mismatch_pct": round(mismatch / len(df) * 100, 4),
    }


# === 2. 4-fold 수동 2단 배분 ===
def make_4fold_split(
    df: pd.DataFrame,
    label_col: str,
    equipment_col: str = "equipment_id",
    single_class_as_normal: bool = True,
    random_state: int = RANDOM_STATE,
):
    """
    4-fold 수동 2단 배분.
    - single_class_as_normal=True: label 단일값 장비를 '정상장비'로 분리 후 균등 배분 (anomaly용)
    - single_class_as_normal=False: 전체 장비를 패턴 기반 round-robin (fault 3-class용)

    반환: list of (train_row_idx, test_row_idx), len=N_FOLDS
    """
    rng = np.random.RandomState(random_state)
    eq_classes = df.groupby(equipment_col)[label_col].apply(
        lambda s: tuple(sorted(pd.unique(s)))
    )

    if single_class_as_normal:
        normal_eqs = sorted([eq for eq, cls in eq_classes.items() if len(cls) == 1])
        fault_eqs = sorted([eq for eq, cls in eq_classes.items() if len(cls) > 1])
    else:
        normal_eqs = []
        fault_eqs = sorted(eq_classes.index.tolist())

    # 정상장비 균등 분배
    normal_folds = [[] for _ in range(N_FOLDS)]
    if len(normal_eqs) > 0:
        normal_shuffled = rng.permutation(normal_eqs)
        normal_folds = [list(x) for x in np.array_split(normal_shuffled, N_FOLDS)]

    # 고장장비 패턴 round-robin
    fault_by_pattern: dict = {}
    for eq in fault_eqs:
        fault_by_pattern.setdefault(eq_classes[eq], []).append(eq)
    fault_folds: list = [[] for _ in range(N_FOLDS)]
    for pat, eqs in fault_by_pattern.items():
        eqs_shuffled = rng.permutation(eqs)
        for i, eq in enumerate(eqs_shuffled):
            fault_folds[i % N_FOLDS].append(eq)

    splits = []
    all_eqs = set(df[equipment_col].unique())
    for k in range(N_FOLDS):
        test_eqs = set(list(normal_folds[k]) + list(fault_folds[k]))
        train_eqs = all_eqs - test_eqs
        train_idx = df[df[equipment_col].isin(train_eqs)].index.values
        test_idx = df[df[equipment_col].isin(test_eqs)].index.values
        splits.append((train_idx, test_idx))

    # assert
    all_classes = set(df[label_col].unique())
    test_eqs_union: set = set()
    for k, (tr, te) in enumerate(splits):
        tr_eqs = set(df.loc[tr, equipment_col].unique())
        te_eqs = set(df.loc[te, equipment_col].unique())
        assert tr_eqs.isdisjoint(te_eqs), f"fold{k} train/test 장비 겹침"
        test_classes = set(df.loc[te, label_col].unique())
        missing_cls = all_classes - test_classes
        assert not missing_cls, f"fold{k} test에 누락 클래스: {missing_cls}"
        test_eqs_union |= te_eqs
    assert test_eqs_union == all_eqs, "전체 장비가 test에 한 번씩 등장 안 함"

    return splits


# === 3. 평가 ===
def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sums, where=row_sums != 0)


def _per_equipment_f1(y_true, y_pred, eq_ids, classes) -> dict:
    df_tmp = pd.DataFrame({"eq": eq_ids, "y": y_true, "p": y_pred})
    return {
        str(eq): float(
            f1_score(g.y, g.p, average="macro", labels=classes, zero_division=0)
        )
        for eq, g in df_tmp.groupby("eq")
    }


def evaluate_fold(y_true, y_pred, classes, equipment_ids=None) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    return {
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
        ),
        "f1_per_class": f1_score(
            y_true, y_pred, average=None, labels=classes, zero_division=0
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=classes, zero_division=0, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_normalized": _row_normalize(cm).tolist(),
        "per_equipment_f1": _per_equipment_f1(y_true, y_pred, equipment_ids, classes)
        if equipment_ids is not None
        else None,
    }


def evaluate_fold_binary(y_true, y_pred, y_proba, equipment_ids=None) -> dict:
    base = evaluate_fold(y_true, y_pred, classes=[0, 1], equipment_ids=equipment_ids)
    base["recall_anomaly"] = float(
        recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    base["precision_anomaly"] = float(
        precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    )
    try:
        base["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        base["roc_auc"] = None
    return base


def aggregate_folds(fold_results, key: str = "f1_macro") -> dict:
    vals = [r[key] for r in fold_results if r.get(key) is not None]
    return {
        f"{key}_mean": float(np.mean(vals)),
        f"{key}_std": float(np.std(vals)),
        f"{key}_per_fold": vals,
        "n_folds": len(fold_results),
    }


# === 4. 메타 저장 ===
def save_meta(out_dir: Path, **extras) -> None:
    import sklearn  # type: ignore
    import xgboost  # type: ignore

    meta = {
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "python": sys.version,
        "trained_at": datetime.datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        **extras,
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
