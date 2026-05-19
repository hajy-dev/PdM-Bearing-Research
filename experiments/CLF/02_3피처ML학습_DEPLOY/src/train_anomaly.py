"""이상탐지 binary 분류기 학습 — LR/RF/XGB, 4-fold 수동 + random 병기."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    ANOMALY_CLASSES,
    ANOMALY_NAMES,
    FEATURES,
    N_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
    aggregate_folds,
    evaluate_fold_binary,
    load_data,
    make_4fold_split,
    save_meta,
    verify_anomaly_label_consistency,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR = RESULTS_DIR / "anomaly"
N_TRIALS = 30


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
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ───────────────────── Optuna objective ─────────────────────
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
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0),
    }
    return _cv_f1_macro(build_xgb, params, X, y, splits)


def _cv_f1_macro(builder, params, X, y, splits) -> float:
    from sklearn.metrics import f1_score

    scores = []
    for tr, te in splits:
        model = builder(params)
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        scores.append(f1_score(y[te], p, average="macro", labels=ANOMALY_CLASSES, zero_division=0))
    return float(np.mean(scores))


# ───────────────────── fold 평가 ─────────────────────
def eval_model_on_splits(model_factory, X, y, eq_ids, splits) -> list:
    results = []
    for tr, te in splits:
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        proba = model.predict_proba(X[te])[:, 1]
        eq_te = eq_ids[te] if eq_ids is not None else None
        results.append(evaluate_fold_binary(y[te], pred, proba, equipment_ids=eq_te))
    return results


# ───────────────────── 메인 ─────────────────────
def main():
    df = load_data()
    consistency = verify_anomaly_label_consistency(df)
    print(f"[라벨 일관성] is_anomaly vs risk_level!=LOW 불일치: {consistency}")

    y = df["is_anomaly_int"].values.astype(int)
    X = df[FEATURES].values.astype(float)
    eq_ids = df["equipment_id"].values

    # GroupKFold (4-fold 수동 2단 배분)
    print("\n[GroupKFold 4-fold 생성]")
    group_splits = make_4fold_split(df, label_col="is_anomaly_int", single_class_as_normal=True)
    for k, (tr, te) in enumerate(group_splits):
        te_eqs = sorted(df.loc[te, "equipment_id"].unique())
        print(f"  fold{k} test 장비({len(te_eqs)}): {te_eqs}")

    # Random StratifiedKFold (누수 진단)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    random_splits = list(skf.split(X, y))

    all_results = {}
    for split_name, splits in [("group", group_splits), ("random", random_splits)]:
        print(f"\n============ {split_name.upper()} SPLIT ============")

        # LR
        print("[LR]")
        lr_folds = eval_model_on_splits(build_lr, X, y, eq_ids, splits)
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
        rf_folds = eval_model_on_splits(lambda: build_rf(rf_best), X, y, eq_ids, splits)
        rf_agg = aggregate_folds(rf_folds, "f1_macro")

        # XGB
        print("[XGB Optuna 30 trials]")
        xgb_study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        )
        xgb_study.optimize(lambda t: xgb_objective(t, X, y, splits), n_trials=N_TRIALS)
        xgb_best = xgb_study.best_params
        print(f"  best F1={xgb_study.best_value:.4f}, params={xgb_best}")
        xgb_folds = eval_model_on_splits(lambda: build_xgb(xgb_best), X, y, eq_ids, splits)
        xgb_agg = aggregate_folds(xgb_folds, "f1_macro")

        all_results[split_name] = {
            "lr": {"folds": lr_folds, "agg": lr_agg},
            "rf": {"folds": rf_folds, "agg": rf_agg, "best_params": rf_best},
            "xgb": {"folds": xgb_folds, "agg": xgb_agg, "best_params": xgb_best},
        }

    # ── 저장 ──
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (OUT_DIR / "label_consistency.json").write_text(
        json.dumps(consistency, indent=2, ensure_ascii=False), encoding="utf-8"
    )

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
    }
    (OUT_DIR / "anomaly_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 최고 모델: group-XGB 기준 전체 데이터 재학습 → 저장
    best_params = all_results["group"]["xgb"]["best_params"]
    best_model = build_xgb(best_params)
    best_model.fit(X, y)
    joblib.dump(best_model, OUT_DIR / "anomaly_model.joblib")
    (OUT_DIR / "anomaly_label_encoder.json").write_text(
        json.dumps(ANOMALY_NAMES, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_meta(
        OUT_DIR,
        task="anomaly_binary",
        best_model="xgb",
        best_params=best_params,
        n_rows=int(len(df)),
        features=FEATURES,
    )

    # ── 최종 콘솔 요약 ──
    print("\n========== ANOMALY FINAL ==========")
    print(f"라벨 일관성: 불일치 {consistency['mismatch_count']}건 ({consistency['mismatch_pct']}%)")
    for split_name in ["group", "random"]:
        print(f"\n[{split_name.upper()} KFold 4]")
        for m in ["lr", "rf", "xgb"]:
            agg = all_results[split_name][m]["agg"]
            last = all_results[split_name][m]["folds"][-1]
            print(
                f"  {m.upper():3} F1-Macro {agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}  "
                f"Recall(1) {last.get('recall_anomaly', 0):.3f}  AUC {last.get('roc_auc') or 0:.3f}"
            )
    gap = summary["leakage_gap_f1"]
    print(f"\n[누수 진단 gap = random - group] {gap}")
    print(f"\n저장: {OUT_DIR / 'anomaly_model.joblib'}")


if __name__ == "__main__":
    main()
