"""3피처 ML 학습 전 EDA — risk_level / fault_type 분포, (B) 전제 검증, 장비별 편향 확인."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV = Path(r"C:\Users\User\Desktop\git\finalproject\AIpassBackend\training_data.csv")
OUT = Path(r"D:\project\예지보전_v2\증명_3피처ML학습\results")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = ["vibration", "temperature", "motor_current"]
THRESHOLDS = {
    "vibration": (0.8, 2.0),
    "temperature": (60, 80),
    "motor_current": (20, 35),
}


def main():
    df = pd.read_csv(CSV)
    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    # [1] Shape & dtypes & nulls
    log(f"[1] shape={df.shape}")
    log(f"    columns={list(df.columns)}")
    log(f"    dtypes:\n{df.dtypes}")
    log(f"    nulls:\n{df.isnull().sum()}")

    # [2] 라벨 분포
    log("\n[2] risk_level 분포 (count)")
    log(df["risk_level"].value_counts(dropna=False).to_string())
    log("\n    risk_level 분포 (%)")
    log(df["risk_level"].value_counts(normalize=True).mul(100).round(2).to_string())

    log("\n    fault_type 분포 (count)")
    log(df["fault_type"].value_counts(dropna=False).to_string())
    log("\n    fault_type 분포 (%)")
    log(df["fault_type"].value_counts(normalize=True).mul(100).round(2).to_string())

    fault_classes = sorted(df["fault_type"].dropna().unique().tolist())
    risk_classes = sorted(df["risk_level"].dropna().unique().tolist())
    log(f"\n    fault_type 클래스 목록: {fault_classes}")
    log(f"    risk_level 클래스 목록: {risk_classes}")

    # [3] Crosstab
    log("\n[3-a] risk_level × fault_type (count)")
    ct_rf = pd.crosstab(df["risk_level"], df["fault_type"], margins=True)
    log(ct_rf.to_string())

    log("\n[3-a'] risk_level × fault_type (row %)")
    ct_rf_row = (
        pd.crosstab(df["risk_level"], df["fault_type"], normalize="index")
        .mul(100)
        .round(2)
    )
    log(ct_rf_row.to_string())

    # (B) 전제 검증
    low_normal_pct = None
    if "LOW" in ct_rf_row.index and "NORMAL" in ct_rf_row.columns:
        low_normal_pct = float(ct_rf_row.loc["LOW", "NORMAL"])
        log(f"\n    >>> (B) 전제 검증: risk=LOW 행 중 fault=NORMAL 비율 = {low_normal_pct}%")
        if low_normal_pct >= 99.5:
            log("    >>> 전제 성립 (≥99.5%): 계층 파이프라인 안전")
        else:
            log(f"    >>> 주의: 전제 불완전. risk=LOW에 NORMAL 외 라벨 존재")

    log("\n[3-b] equipment_id × risk_level")
    log(pd.crosstab(df["equipment_id"], df["risk_level"]).to_string())

    log("\n[3-c] equipment_id × fault_type")
    log(pd.crosstab(df["equipment_id"], df["fault_type"]).to_string())

    # [4] 피처 통계 (클래스별)
    log("\n[4-a] risk_level 별 피처 통계")
    log(df.groupby("risk_level")[FEATURES].describe().round(3).to_string())
    log("\n[4-b] fault_type 별 피처 통계")
    log(df.groupby("fault_type")[FEATURES].describe().round(3).to_string())

    # [5] tick 버킷 (recorded_at 정렬 기반, 장비별 rank 후 5분위)
    df_sorted = df.sort_values(["equipment_id", "recorded_at"]).reset_index(drop=True)
    df_sorted["tick_rank"] = df_sorted.groupby("equipment_id").cumcount()
    df_sorted["tick_bucket"] = df_sorted.groupby("equipment_id")["tick_rank"].transform(
        lambda s: pd.qcut(s, 5, labels=[f"Q{i+1}" for i in range(5)], duplicates="drop")
    )
    log("\n[5] tick 버킷(장비별 5분위) × risk_level (row %)")
    log(
        pd.crosstab(df_sorted["tick_bucket"], df_sorted["risk_level"], normalize="index")
        .mul(100)
        .round(2)
        .to_string()
    )

    # [6] 장비별 요약
    log("\n[6] 장비별 요약")
    eq_summary = df.groupby("equipment_id").agg(
        n_rows=("risk_level", "size"),
        n_risk=("risk_level", "nunique"),
        n_fault=("fault_type", "nunique"),
    )
    log(eq_summary.to_string())

    # [7] equipment 목록
    eqs = sorted(df["equipment_id"].unique().tolist())
    log(f"\n[7] equipment_id 수={len(eqs)}")
    log(f"    목록={eqs}")

    # 텍스트 저장
    (OUT / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # 이미지
    sample = df.sample(min(len(df), 5000), random_state=42)

    sns.pairplot(
        sample,
        vars=FEATURES,
        hue="risk_level",
        diag_kind="hist",
        plot_kws={"alpha": 0.4, "s": 10},
    )
    plt.savefig(OUT / "pairplot_by_risk.png", dpi=100, bbox_inches="tight")
    plt.close()

    sns.pairplot(
        sample,
        vars=FEATURES,
        hue="fault_type",
        diag_kind="hist",
        plot_kws={"alpha": 0.4, "s": 10},
    )
    plt.savefig(OUT / "pairplot_by_fault.png", dpi=100, bbox_inches="tight")
    plt.close()

    # threshold overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, f in zip(axes, FEATURES):
        ax.hist(df[f], bins=80)
        for t in THRESHOLDS[f]:
            ax.axvline(t, color="red", linestyle="--", label=f"thr={t}")
        ax.set_title(f)
        ax.legend()
    plt.tight_layout()
    plt.savefig(OUT / "threshold_overlay.png", dpi=100)
    plt.close()

    # metadata
    meta = {
        "n_rows": int(len(df)),
        "n_equipment": len(eqs),
        "fault_classes": fault_classes,
        "risk_classes": risk_classes,
        "low_normal_row_pct": low_normal_pct,
    }
    (OUT / "eda_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[완료] 결과 저장 경로: {OUT}")


if __name__ == "__main__":
    main()
