"""
═══════════════════════════════════════════════════════════════════
  RUL 위험도 4등급 분류 모델 비교 실험 (v2)
  AI-PASS 예지보전 | 22 bearings, Optuna 최적화
═══════════════════════════════════════════════════════════════════

변경사항 (v1 대비):
  - 회귀(R²) → 4등급 분류(Accuracy)로 전환
  - 데이터: FEMTO(11) + IMS(4) + Zenodo(6) + KAIST(1) = 22 bearings
  - 피처: 시간도메인 8개 + 주파수도메인 6개 + 추세 3개 = 17개 공통
  - 실험 A: 전체 22 bearings 진동 기반 (LOBO-CV)
  - 실험 B: Zenodo(6)+KAIST(1) 환경보정 포함 (LOBO-CV)
  - Optuna 하이퍼파라미터 최적화
  - Health Index 기반 피스와이즈 RUL

출력:
  D:/project/예지보전_v2/compare_rul_v2/
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import logging
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import load_all_rtf
from feature_engineering import (add_trend_features, compute_health_index,
                                  add_severity_labels, SEVERITY_NAMES)

OUT_DIR = r"D:\project\예지보전_v2\compare_rul_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 로거
log_path = os.path.join(OUT_DIR, "comparison_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger()


def section(title: str):
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ══════════════════════════════════════════════
# Optuna 최적화
# ══════════════════════════════════════════════
def optuna_xgboost(X, y, groups, n_trials=80):
    """XGBoost 하이퍼파라미터 Optuna 최적화"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 15),
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            'eval_metric': 'mlogloss',
        }
        # 3-fold GroupKFold로 빠르게 평가
        gkf = GroupKFold(n_splits=min(len(np.unique(groups)), 5))
        scores = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y[tr_idx])
            scores.append(f1_score(y[te_idx], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna XGBoost] Best F1={study.best_value:.4f}")
    log.info(f"  [Optuna XGBoost] Best params: {study.best_params}")
    return study.best_params


def optuna_lgbm(X, y, groups, n_trials=80):
    """LightGBM 하이퍼파라미터 Optuna 최적화"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        gkf = GroupKFold(n_splits=min(len(np.unique(groups)), 5))
        scores = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y[tr_idx])
            scores.append(f1_score(y[te_idx], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna LightGBM] Best F1={study.best_value:.4f}")
    return study.best_params


# ══════════════════════════════════════════════
# LOBO-CV 평가
# ══════════════════════════════════════════════
def evaluate_model(model_builder, X, y, groups, model_name="") -> dict:
    """LOBO-CV로 분류 모델을 평가한다."""
    unique_groups = np.unique(groups)
    n_splits = min(len(unique_groups), 10)
    gkf = GroupKFold(n_splits=n_splits)

    acc_list, f1_list, train_acc_list, gap_list = [], [], [], []
    all_y_true, all_y_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        model = model_builder()
        model.fit(X_tr, y[tr_idx])

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)

        train_acc = accuracy_score(y[tr_idx], pred_tr)
        test_acc = accuracy_score(y[te_idx], pred_te)
        test_f1 = f1_score(y[te_idx], pred_te, average='macro')

        acc_list.append(test_acc)
        f1_list.append(test_f1)
        train_acc_list.append(train_acc)
        gap_list.append(train_acc - test_acc)
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(pred_te.tolist())

        test_bears = sorted(set(groups[te_idx]))
        log.info(f"    Fold {fold+1}: bearings={test_bears[:3]}{'...' if len(test_bears)>3 else ''}  "
                 f"Acc={test_acc:.4f}  F1={test_f1:.4f}  gap={train_acc-test_acc:.4f}")

    return {
        "acc_scores": acc_list, "f1_scores": f1_list,
        "train_acc": train_acc_list, "overfit_gaps": gap_list,
        "all_y_true": all_y_true, "all_y_pred": all_y_pred,
    }


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def plot_results(results: dict, experiment_name: str, save_path: str):
    model_names = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(model_names)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{experiment_name} — 4-Class Severity Classification (LOBO-CV)',
                 fontsize=13, fontweight='bold')

    # Accuracy
    ax = axes[0, 0]
    data = [results[m]['acc_scores'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('Accuracy'); ax.grid(axis='y', alpha=0.3)

    # F1-Macro
    ax = axes[0, 1]
    data = [results[m]['f1_scores'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('F1-Macro'); ax.grid(axis='y', alpha=0.3)

    # Overfit Gap
    ax = axes[1, 0]
    data = [results[m]['overfit_gaps'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('Overfit Gap'); ax.grid(axis='y', alpha=0.3)

    # Summary
    ax = axes[1, 1]
    mean_acc = [np.mean(results[m]['acc_scores']) for m in model_names]
    mean_f1 = [np.mean(results[m]['f1_scores']) for m in model_names]
    x = np.arange(len(model_names))
    ax.bar(x - 0.15, mean_acc, 0.3, label='Accuracy', color=colors, alpha=0.8)
    ax.bar(x + 0.15, mean_f1, 0.3, label='F1-Macro', color=colors, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_title('Mean Accuracy vs F1'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (a, f) in enumerate(zip(mean_acc, mean_f1)):
        ax.text(i-0.15, a+0.005, f'{a:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"RUL 4등급 분류 비교 v2 — {start.strftime('%Y-%m-%d %H:%M')}")

    # ── 데이터 로드
    section("Step 1. RTF 데이터 로드")
    df_all = load_all_rtf()
    if df_all.empty:
        log.error("데이터 없음 — 종료")
        return

    # ── 피처 엔지니어링
    section("Step 2. 피처 엔지니어링")
    df_all = add_trend_features(df_all)

    # Health Index 계산
    hi_features = ['vibration_rms', 'short_trend', 'rolling_trend']
    df_all = compute_health_index(df_all, hi_features)

    # 4등급 라벨 생성
    df_all = add_severity_labels(df_all)
    log.info(f"  등급 분포:\n{df_all['severity_name'].value_counts().to_string()}")

    # ── 공통 피처 정의
    common_features = [
        'vibration_rms', 'short_trend', 'rolling_trend',
        'health_index',
        # 시간 도메인 통계
        'vibration_std', 'vibration_peak', 'vibration_kurtosis',
        'vibration_skewness', 'crest_factor', 'impulse_factor', 'shape_factor',
        # 주파수 도메인
        'spectral_energy', 'spectral_centroid', 'spectral_spread',
        'band_energy_low', 'band_energy_mid', 'band_energy_high',
    ]

    # ── 베어링별 피처 정규화 (health_index 제외)
    normalize_features = [f for f in common_features if f != 'health_index']
    for feat in normalize_features:
        df_all[feat] = df_all.groupby('bearing_id')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-10)
        )
    log.info(f"  베어링별 z-score 정규화 완료: {len(normalize_features)}개 피처")

    # ══════════════════════════════════════════
    # 실험 A: 전체 22 bearings — 진동 기반
    # ══════════════════════════════════════════
    section("실험 A: 전체 RTF 진동 기반 (22 bearings)")

    df_a = df_all.dropna(subset=common_features + ['severity'])
    X_a = df_a[common_features].values.astype(np.float32)
    y_a = df_a['severity'].values.astype(int)
    groups_a = df_a['bearing_id'].values

    log.info(f"  데이터: {len(X_a):,} rows / {len(np.unique(groups_a))} bearings")
    log.info(f"  피처: {common_features}")

    # Optuna 최적화
    section("실험 A — Optuna 최적화")
    best_xgb_a = optuna_xgboost(X_a, y_a, groups_a, n_trials=60)
    best_lgb_a = optuna_lgbm(X_a, y_a, groups_a, n_trials=60)

    # 모델 평가
    section("실험 A — LOBO-CV 평가")
    results_a = {}

    log.info("  [XGBoost]")
    results_a['XGBoost'] = evaluate_model(
        lambda: xgb.XGBClassifier(**best_xgb_a, random_state=42, verbosity=0,
                                    n_jobs=-1, eval_metric='mlogloss'),
        X_a, y_a, groups_a)

    log.info("  [LightGBM]")
    results_a['LightGBM'] = evaluate_model(
        lambda: lgb.LGBMClassifier(**best_lgb_a, random_state=42, verbose=-1, n_jobs=-1),
        X_a, y_a, groups_a)

    log.info("  [RandomForest]")
    results_a['RandomForest'] = evaluate_model(
        lambda: RandomForestClassifier(n_estimators=300, max_depth=12,
                                        min_samples_leaf=5, random_state=42, n_jobs=-1),
        X_a, y_a, groups_a)

    # ══════════════════════════════════════════
    # 실험 B: 환경보정 포함 (Zenodo + KAIST)
    # ══════════════════════════════════════════
    section("실험 B: 환경보정 포함 (Zenodo + KAIST)")

    df_b = df_all[df_all['has_temp'] == True].copy()
    env_features = common_features + ['temp_residual']
    # ambient_temp가 있으면 추가
    if 'ambient_temp' in df_b.columns:
        env_features.append('ambient_temp')

    df_b = df_b.dropna(subset=env_features + ['severity'])

    if len(df_b) > 50 and df_b['bearing_id'].nunique() >= 3:
        X_b = df_b[env_features].values.astype(np.float32)
        y_b = df_b['severity'].values.astype(int)
        groups_b = df_b['bearing_id'].values

        log.info(f"  데이터: {len(X_b):,} rows / {len(np.unique(groups_b))} bearings")
        log.info(f"  피처: {env_features}")

        # Optuna
        section("실험 B — Optuna 최적화")
        best_xgb_b = optuna_xgboost(X_b, y_b, groups_b, n_trials=60)

        # 평가
        section("실험 B — LOBO-CV 평가")
        results_b = {}

        log.info("  [XGBoost + 환경보정]")
        results_b['XGB+Env'] = evaluate_model(
            lambda: xgb.XGBClassifier(**best_xgb_b, random_state=42, verbosity=0,
                                        n_jobs=-1, eval_metric='mlogloss'),
            X_b, y_b, groups_b)

        # 비교: 환경보정 없이 (같은 데이터에서 진동만)
        log.info("  [XGBoost 진동만 (비교용)]")
        X_b_vib = df_b[common_features].values.astype(np.float32)
        results_b['XGB VibOnly'] = evaluate_model(
            lambda: xgb.XGBClassifier(**best_xgb_a, random_state=42, verbosity=0,
                                        n_jobs=-1, eval_metric='mlogloss'),
            X_b_vib, y_b, groups_b)
    else:
        log.warning("  환경보정 데이터 부족 — 실험 B 생략")
        results_b = {}

    # ══════════════════════════════════════════
    # 결과 정리
    # ══════════════════════════════════════════
    section("결과 요약")

    all_results = {}
    for exp_name, results in [("실험A", results_a), ("실험B", results_b)]:
        if not results:
            continue
        log.info(f"\n  [{exp_name}]")
        log.info(f"  {'모델':<20} {'Acc':>8} {'F1':>8} {'Gap':>8}")
        log.info("  " + "-" * 50)
        for m, r in results.items():
            acc = np.mean(r['acc_scores'])
            f1 = np.mean(r['f1_scores'])
            gap = np.mean(r['overfit_gaps'])
            log.info(f"  {m:<20} {acc:>8.4f} {f1:>8.4f} {gap:>8.4f}")

            # Classification Report
            report = classification_report(r['all_y_true'], r['all_y_pred'],
                                            target_names=SEVERITY_NAMES, digits=4)
            log.info(f"\n{report}")

        all_results[exp_name] = {
            m: {
                "mean_acc": float(np.mean(r['acc_scores'])),
                "mean_f1": float(np.mean(r['f1_scores'])),
                "mean_gap": float(np.mean(r['overfit_gaps'])),
                "fold_acc": [float(x) for x in r['acc_scores']],
                "fold_f1": [float(x) for x in r['f1_scores']],
            }
            for m, r in results.items()
        }

    # 환경보정 효과 비교
    if results_b and 'XGB+Env' in results_b and 'XGB VibOnly' in results_b:
        env_acc = np.mean(results_b['XGB+Env']['acc_scores'])
        vib_acc = np.mean(results_b['XGB VibOnly']['acc_scores'])
        improvement = env_acc - vib_acc
        log.info(f"\n  [환경보정 효과]")
        log.info(f"    진동만: Acc={vib_acc:.4f}")
        log.info(f"    +환경보정: Acc={env_acc:.4f}")
        log.info(f"    개선: +{improvement:.4f} ({improvement*100:.1f}%p)")
        all_results['env_improvement'] = float(improvement)

    # 저장
    plot_results(results_a, "Experiment A: All RTF",
                 os.path.join(OUT_DIR, "exp_a_chart.png"))
    if results_b:
        plot_results(results_b, "Experiment B: Env Correction",
                     os.path.join(OUT_DIR, "exp_b_chart.png"))

    json_path = os.path.join(OUT_DIR, "comparison_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"  [저장] {json_path}")

    elapsed = (datetime.datetime.now() - start).total_seconds() / 60
    log.info(f"\n  총 소요시간: {elapsed:.1f}분")


if __name__ == "__main__":
    main()
