"""고장분류 3-class 학습 — BEARING/MOTOR/COMPOUND, 4-fold 수동 + random 병기."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    FAULT_CLASSES_3,
    FEATURES,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    aggregate_folds,
    evaluate_fold,
    filter_fault_data,
    load_data,
    make_4fold_split,
    save_meta,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = RESULTS_DIR / "fault"
N_TRIALS = 30
CLASS_INTS = [0, 1, 2]  # LabelEncoder 정수


# ───────────────────── 모델 빌더 ─────────────────────
def build_lr() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_rf(params: dict) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgb(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ───────────────────── CV 평균 F1 (Optuna objective) ─────────────────────
def _cv_f1_macro(builder, params, X, y, splits, use_sample_weight: bool = False) -> float:
    from sklearn.metrics import f1_score

    scores = []
    for tr, te in splits:
        model = builder(params) if params is not None else builder()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y[tr])
            model.fit(X[tr], y[tr], sample_weight=sw)
        else:
            model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        scores.append(
            f1_score(y[te], p, average="macro", labels=CLASS_INTS, zero_division=0)
        )
    return float(np.mean(scores))


def rf_objective(trial: optuna.Trial, X, y, splits) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }
    return _cv_f1_macro(build_rf, params, X, y, splits)


def xgb_objective(trial: optuna.Trial, X, y, splits) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }
    return _cv_f1_macro(build_xgb, params, X, y, splits, use_sample_weight=True)


# ───────────────────── fold 평가 ─────────────────────
def eval_model_on_splits(model_factory, X, y, eq_ids, splits, use_sample_weight: bool = False):
    results = []
    for tr, te in splits:
        model = model_factory()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y[tr])
            model.fit(X[tr], y[tr], sample_weight=sw)
        else:
            model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        eq_te = eq_ids[te] if eq_ids is not None else None
        results.append(evaluate_fold(y[te], pred, classes=CLASS_INTS, equipment_ids=eq_te))
    return results


def per_fault_fold_table(fold_results, le: LabelEncoder) -> dict:
    """클래스별 × fold F1 테이블 생성."""
    rows = []
    for k, r in enumerate(fold_results):
        row = {"fold": k}
        for i, cls in enumerate(le.classes_):
            row[cls] = r["f1_per_class"][i]
        rows.append(row)
    df_table = pd.DataFrame(rows).set_index("fold")
    df_table.loc["mean"] = df_table.mean()
    df_table.loc["std"] = df_table.std(ddof=0)
    return df_table.round(4).to_dict()


# ───────────────────── 메인 ─────────────────────
def main():
    df_full = load_data()
    df = filter_fault_data(df_full)
    print(f"[fault filter] {len(df)} rows, {df['equipment_id'].nunique()} equipments")
    print("  fault_type 분포:")
    print(df["fault_type"].value_counts().to_string())

    # 장비별 보유 패턴 로그
    patterns = df.groupby("equipment_id")["fault_type"].apply(lambda s: sorted(pd.unique(s))).to_dict()
    print("\n  [장비별 fault 패턴]")
    for eq in sorted(patterns.keys()):
        print(f"    eq{eq}: {patterns[eq]}")

    # 라벨 인코딩
    le = LabelEncoder().fit(FAULT_CLASSES_3)
    y = le.transform(df["fault_type"].values)
    X = df[FEATURES].values.astype(float)
    eq_ids = df["equipment_id"].values

    # GroupKFold (4-fold 수동, 전체 장비 패턴 배분)
    print("\n[GroupKFold 4-fold 생성 — single_class_as_normal=False]")
    group_splits = make_4fold_split(df, label_col="fault_type", single_class_as_normal=False)
    for k, (tr, te) in enumerate(group_splits):
        te_eqs = sorted(df.loc[te, "equipment_id"].unique())
        motor_n = int((df.loc[te, "fault_type"] == "MOTOR_FAULT").sum())
        bearing_n = int((df.loc[te, "fault_type"] == "BEARING_FAULT").sum())
        compound_n = int((df.loc[te, "fault_type"] == "COMPOUND_FAULT").sum())
        print(
            f"  fold{k} test 장비 {te_eqs}  "
            f"BEARING={bearing_n} MOTOR={motor_n} COMPOUND={compound_n}"
        )

    # Random StratifiedKFold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_splits = list(skf.split(X, y))

    all_results = {}
    for split_name, splits in [("group", group_splits), ("random", random_splits)]:
        print(f"\n============ {split_name.upper()} SPLIT ============")

        # LR
        print("[LR]")
        lr_folds = eval_model_on_splits(build_lr, X, y, eq_ids, splits, use_sample_weight=False)
        lr_agg = aggregate_folds(lr_folds, "f1_macro")
        print(f"  F1-Macro {lr_agg['f1_macro_mean']:.4f} ± {lr_agg['f1_macro_std']:.4f}")

        # RF
        print("[RF Optuna 30 trials]")
        rf_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        rf_study.optimize(lambda t: rf_objective(t, X, y, splits), n_trials=N_TRIALS)
        rf_best = rf_study.best_params
        print(f"  best F1={rf_study.best_value:.4f}, params={rf_best}")
        rf_folds = eval_model_on_splits(
            lambda: build_rf(rf_best), X, y, eq_ids, splits, use_sample_weight=False
        )
        rf_agg = aggregate_folds(rf_folds, "f1_macro")

        # XGB (sample_weight balanced)
        print("[XGB Optuna 30 trials]")
        xgb_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        xgb_study.optimize(lambda t: xgb_objective(t, X, y, splits), n_trials=N_TRIALS)
        xgb_best = xgb_study.best_params
        print(f"  best F1={xgb_study.best_value:.4f}, params={xgb_best}")
        xgb_folds = eval_model_on_splits(
            lambda: build_xgb(xgb_best), X, y, eq_ids, splits, use_sample_weight=True
        )
        xgb_agg = aggregate_folds(xgb_folds, "f1_macro")

        all_results[split_name] = {
            "lr": {"folds": lr_folds, "agg": lr_agg},
            "rf": {"folds": rf_folds, "agg": rf_agg, "best_params": rf_best},
            "xgb": {"folds": xgb_folds, "agg": xgb_agg, "best_params": xgb_best},
        }

    # ── 저장 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _serialize(results_block):
        return {
            m: {
                "folds": v["folds"],
                "aggregate": v["agg"],
                "best_params": v.get("best_params"),
            }
            for m, v in results_block.items()
        }

    summary = {
        "group": _serialize(all_results["group"]),
        "random": _serialize(all_results["random"]),
        "leakage_gap_f1": {
            m: round(
                all_results["random"][m]["agg"]["f1_macro_mean"]
                - all_results["group"][m]["agg"]["f1_macro_mean"],
                4,
            )
            for m in ["lr", "rf", "xgb"]
        },
        "per_fault_fold_table_xgb_group": per_fault_fold_table(
            all_results["group"]["xgb"]["folds"], le
        ),
    }
    (OUT_DIR / "fault_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 최고 모델: group-XGB 기준 전체 데이터 재학습 (sample_weight balanced)
    best_params = all_results["group"]["xgb"]["best_params"]
    best_model = build_xgb(best_params)
    sw_all = compute_sample_weight("balanced", y)
    best_model.fit(X, y, sample_weight=sw_all)
    joblib.dump(best_model, OUT_DIR / "fault_model.joblib")
    joblib.dump(le, OUT_DIR / "fault_label_encoder.joblib")
    (OUT_DIR / "fault_label_map.json").write_text(
        json.dumps({str(i): c for i, c in enumerate(le.classes_)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_meta(
        OUT_DIR,
        task="fault_3class",
        best_model="xgb",
        best_params=best_params,
        n_rows=int(len(df)),
        n_equipments=int(df["equipment_id"].nunique()),
        classes=list(le.classes_),
        features=FEATURES,
    )

    # ── 최종 콘솔 요약 ──
    print("\n========== FAULT FINAL ==========")
    for split_name in ["group", "random"]:
        print(f"\n[{split_name.upper()} KFold 4]")
        for m in ["lr", "rf", "xgb"]:
            agg = all_results[split_name][m]["agg"]
            print(f"  {m.upper():3} F1-Macro {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}")
    gap = summary["leakage_gap_f1"]
    print(f"\n[누수 진단 gap = random - group] {gap}")

    # per-fault × fold 테이블
    print("\n[per-fault × fold F1 (group XGB)]")
    table = summary["per_fault_fold_table_xgb_group"]
    print(pd.DataFrame(table).to_string())

    print(f"\n저장: {OUT_DIR / 'fault_model.joblib'}")


if __name__ == "__main__":
    main()
