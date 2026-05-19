"""
═══════════════════════════════════════════════════════════════════
  RUL 잔여수명 회귀 예측 모델 비교 실험 (v3 — 회귀 전환)
  AI-PASS 예지보전 | 22 bearings, Optuna 최적화
═══════════════════════════════════════════════════════════════════

변경사항 (v2 대비):
  - 4등급 분류 → rul_ratio(0~1) 회귀 예측으로 전환
  - 모델: XGBRegressor, LGBMRegressor (RF 제외 — CPU 병목)
  - 평가: MAE, RMSE, R², ±0.1 적중률 (Tolerance Accuracy)
  - sample_weight 제거 (회귀에 불필요)
  - 시각화: Scatter(예측vs실제), 잔차분포, 베어링별 곡선
  - 등급 분류는 배포 시 Spring Boot에서 후처리

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

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import load_all_rtf
from feature_engineering import add_trend_features, compute_health_index

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
# 회귀 평가 지표
# ══════════════════════════════════════════════
def tolerance_accuracy(y_true, y_pred, tol=0.1):
    """±tol 이내 적중률 계산"""
    return float(np.mean(np.abs(y_true - y_pred) <= tol))


# ══════════════════════════════════════════════
# Optuna 최적화 (회귀)
# ══════════════════════════════════════════════
def optuna_xgboost_reg(X, y, groups, n_trials=60):
    """XGBoost Regressor Optuna 최적화 (과적합 억제)"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 30),
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
        }
        gkf = GroupKFold(n_splits=min(len(np.unique(groups)), 5))
        scores = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y[tr_idx])
            pred = model.predict(X_te)
            scores.append(mean_absolute_error(y[te_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna XGBoost] Best MAE={study.best_value:.4f}")
    log.info(f"  [Optuna XGBoost] Best params: {study.best_params}")
    return study.best_params


def optuna_lgbm_reg(X, y, groups, n_trials=60):
    """LightGBM Regressor Optuna 최적화 (과적합 억제)"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 7, 31),
            'max_depth': trial.suggest_int('max_depth', 3, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'random_state': 42, 'verbose': -1, 'n_jobs': -1,
        }
        gkf = GroupKFold(n_splits=min(len(np.unique(groups)), 5))
        scores = []
        for tr_idx, te_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y[tr_idx])
            pred = model.predict(X_te)
            scores.append(mean_absolute_error(y[te_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna LightGBM] Best MAE={study.best_value:.4f}")
    log.info(f"  [Optuna LightGBM] Best params: {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════
# LOBO-CV 평가 (회귀)
# ══════════════════════════════════════════════
def evaluate_regression(model_builder, X, y, groups, model_name="") -> dict:
    """LOBO-CV로 회귀 모델을 평가한다."""
    unique_groups = np.unique(groups)
    n_splits = min(len(unique_groups), 10)
    gkf = GroupKFold(n_splits=n_splits)

    mae_list, rmse_list, r2_list = [], [], []
    tol_acc_list = []
    train_mae_list, gap_list = [], []
    all_y_true, all_y_pred, all_groups = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        model = model_builder()
        model.fit(X_tr, y[tr_idx])

        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)

        # Train 성능
        train_mae = mean_absolute_error(y[tr_idx], pred_tr)

        # Test 성능
        test_mae = mean_absolute_error(y[te_idx], pred_te)
        test_rmse = float(np.sqrt(mean_squared_error(y[te_idx], pred_te)))
        test_r2 = r2_score(y[te_idx], pred_te)
        test_tol = tolerance_accuracy(y[te_idx], pred_te, tol=0.1)

        mae_list.append(test_mae)
        rmse_list.append(test_rmse)
        r2_list.append(test_r2)
        tol_acc_list.append(test_tol)
        train_mae_list.append(train_mae)
        gap_list.append(test_mae - train_mae)  # 양수 = 과적합 (test가 더 나쁨)
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(pred_te.tolist())
        all_groups.extend(groups[te_idx].tolist())

        test_bears = sorted(set(groups[te_idx]))
        log.info(f"    Fold {fold+1}: bearings={test_bears[:3]}{'...' if len(test_bears)>3 else ''}  "
                 f"MAE={test_mae:.4f}  R²={test_r2:.4f}  ±0.1Hit={test_tol:.4f}  "
                 f"gap(MAE)={test_mae-train_mae:.4f}")

    return {
        "mae_scores": mae_list, "rmse_scores": rmse_list,
        "r2_scores": r2_list, "tol_acc_scores": tol_acc_list,
        "train_mae": train_mae_list, "overfit_gaps": gap_list,
        "all_y_true": all_y_true, "all_y_pred": all_y_pred,
        "all_groups": all_groups,
    }


# ══════════════════════════════════════════════
# 시각화 (회귀)
# ══════════════════════════════════════════════
def plot_regression_results(results: dict, experiment_name: str, save_path: str):
    """회귀 결과 시각화: Scatter, 잔차, 적중률, 요약"""
    model_names = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(model_names)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{experiment_name} — RUL Regression (LOBO-CV)',
                 fontsize=13, fontweight='bold')

    # 1. MAE Boxplot
    ax = axes[0, 0]
    data = [results[m]['mae_scores'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('MAE (lower is better)'); ax.grid(axis='y', alpha=0.3)

    # 2. R² Boxplot
    ax = axes[0, 1]
    data = [results[m]['r2_scores'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('R² (higher is better)'); ax.grid(axis='y', alpha=0.3)

    # 3. ±0.1 Tolerance Accuracy Boxplot
    ax = axes[1, 0]
    data = [results[m]['tol_acc_scores'] for m in model_names]
    bp = ax.boxplot(data, labels=model_names, patch_artist=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.set_title('±0.1 Tolerance Accuracy (higher is better)')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target 80%')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    # 4. Summary Bar
    ax = axes[1, 1]
    mean_mae = [np.mean(results[m]['mae_scores']) for m in model_names]
    mean_tol = [np.mean(results[m]['tol_acc_scores']) for m in model_names]
    x = np.arange(len(model_names))
    ax.bar(x - 0.15, mean_mae, 0.3, label='MAE', color=colors, alpha=0.8)
    ax.bar(x + 0.15, mean_tol, 0.3, label='±0.1 Hit Rate', color=colors, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(model_names)
    ax.set_title('Mean MAE vs ±0.1 Hit Rate'); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, (m_val, t_val) in enumerate(zip(mean_mae, mean_tol)):
        ax.text(i-0.15, m_val+0.005, f'{m_val:.3f}', ha='center', fontsize=9)
        ax.text(i+0.15, t_val+0.005, f'{t_val:.1%}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  [저장] {save_path}")


def plot_scatter(results: dict, model_name: str, save_path: str):
    """예측 vs 실제 Scatter Plot + 잔차 분포"""
    r = results[model_name]
    y_true = np.array(r['all_y_true'])
    y_pred = np.array(r['all_y_pred'])
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} — Prediction Analysis', fontsize=13, fontweight='bold')

    # 1. Scatter: 예측 vs 실제
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.1, s=5, c='#2196F3')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect')
    ax.fill_between([0, 1], [-0.1, 0.9], [0.1, 1.1], alpha=0.1, color='green', label='±0.1 zone')
    ax.set_xlabel('Actual rul_ratio'); ax.set_ylabel('Predicted rul_ratio')
    ax.set_title('Predicted vs Actual'); ax.legend(); ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # 2. 잔차 분포
    ax = axes[1]
    ax.hist(residuals, bins=50, alpha=0.7, color='#4CAF50', edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=0.1, color='orange', linestyle='--', alpha=0.5, label='±0.1')
    ax.axvline(x=-0.1, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('Residual (Pred - Actual)'); ax.set_ylabel('Count')
    ax.set_title('Residual Distribution'); ax.legend(); ax.grid(alpha=0.3)

    # 3. 구간별 적중률
    ax = axes[2]
    bins = [(0.0, 0.15, 'CRITICAL'), (0.15, 0.4, 'HIGH'),
            (0.4, 0.7, 'MEDIUM'), (0.7, 1.0, 'LOW')]
    bin_names, bin_hits = [], []
    for lo, hi, name in bins:
        mask = (y_true >= lo) & (y_true < hi)
        if mask.sum() > 0:
            hit = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) <= 0.1))
        else:
            hit = 0.0
        bin_names.append(f'{name}\n({lo:.2f}~{hi:.2f})')
        bin_hits.append(hit)
    bar_colors = ['#f44336', '#ff9800', '#ffc107', '#4caf50']
    ax.bar(bin_names, bin_hits, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target 80%')
    ax.set_ylabel('±0.1 Hit Rate'); ax.set_title('Hit Rate by RUL Zone')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, h in enumerate(bin_hits):
        ax.text(i, h + 0.02, f'{h:.1%}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"RUL 회귀 예측 비교 v3 — {start.strftime('%Y-%m-%d %H:%M')}")

    # ── 데이터 로드
    section("Step 1. RTF 데이터 로드")
    df_all = load_all_rtf()
    if df_all.empty:
        log.error("데이터 없음 — 종료")
        return

    # ── 피처 엔지니어링
    section("Step 2. 피처 엔지니어링")
    df_all = add_trend_features(df_all)

    # Health Index 계산 (참조용 — 피처로 사용하지 않음)
    hi_features = ['vibration_rms', 'short_trend', 'rolling_trend']
    df_all = compute_health_index(df_all, hi_features)

    # rul_ratio 분포 확인
    log.info(f"  rul_ratio 분포:")
    log.info(f"    mean={df_all['rul_ratio'].mean():.4f}  std={df_all['rul_ratio'].std():.4f}")
    log.info(f"    min={df_all['rul_ratio'].min():.4f}  max={df_all['rul_ratio'].max():.4f}")
    log.info(f"    >0.7 비율: {(df_all['rul_ratio'] > 0.7).mean():.1%}")
    log.info(f"    ≤0.3 비율: {(df_all['rul_ratio'] <= 0.3).mean():.1%}")

    # ── 공통 피처 정의 (16개 — 분류와 동일)
    common_features = [
        'vibration_rms', 'short_trend', 'rolling_trend',
        'vibration_std', 'vibration_peak', 'vibration_kurtosis',
        'vibration_skewness', 'crest_factor', 'impulse_factor', 'shape_factor',
        'spectral_energy', 'spectral_centroid', 'spectral_spread',
        'band_energy_low', 'band_energy_mid', 'band_energy_high',
    ]

    # ── 베어링별 피처 정규화
    for feat in common_features:
        df_all[feat] = df_all.groupby('bearing_id')[feat].transform(
            lambda x: (x - x.mean()) / max(x.std(), 1e-10)
        )
    log.info(f"  베어링별 z-score 정규화 완료: {len(common_features)}개 피처")

    # ══════════════════════════════════════════
    # 실험 A: 전체 22 bearings — 진동 기반 회귀
    # ══════════════════════════════════════════
    section("실험 A: 전체 RTF 진동 기반 회귀 (22 bearings)")

    df_a = df_all.dropna(subset=common_features + ['rul_ratio'])
    X_a = df_a[common_features].values.astype(np.float32)
    y_a = df_a['rul_ratio'].values.astype(np.float32)
    groups_a = df_a['bearing_id'].values

    log.info(f"  데이터: {len(X_a):,} rows / {len(np.unique(groups_a))} bearings")
    log.info(f"  피처: {common_features}")
    log.info(f"  타겟: rul_ratio (0~1 연속값)")

    # Optuna 최적화
    section("실험 A — Optuna 최적화")
    best_xgb_a = optuna_xgboost_reg(X_a, y_a, groups_a, n_trials=60)
    best_lgb_a = optuna_lgbm_reg(X_a, y_a, groups_a, n_trials=60)

    # 모델 평가
    section("실험 A — LOBO-CV 평가")
    results_a = {}

    log.info("  [XGBoost Regressor]")
    results_a['XGBoost'] = evaluate_regression(
        lambda: xgb.XGBRegressor(**best_xgb_a, random_state=42, verbosity=0, n_jobs=-1),
        X_a, y_a, groups_a)

    log.info("  [LightGBM Regressor]")
    results_a['LightGBM'] = evaluate_regression(
        lambda: lgb.LGBMRegressor(**best_lgb_a, random_state=42, verbose=-1, n_jobs=-1),
        X_a, y_a, groups_a)

    # ══════════════════════════════════════════
    # 실험 B: 환경보정 포함 (Zenodo + KAIST)
    # ══════════════════════════════════════════
    section("실험 B: 환경보정 포함 (Zenodo + KAIST)")

    df_b = df_all[df_all['has_temp'] == True].copy()
    env_features = common_features + ['temp_residual']
    if 'ambient_temp' in df_b.columns:
        env_features.append('ambient_temp')

    df_b = df_b.dropna(subset=env_features + ['rul_ratio'])

    results_b = {}
    if len(df_b) > 50 and df_b['bearing_id'].nunique() >= 3:
        X_b = df_b[env_features].values.astype(np.float32)
        y_b = df_b['rul_ratio'].values.astype(np.float32)
        groups_b = df_b['bearing_id'].values

        log.info(f"  데이터: {len(X_b):,} rows / {len(np.unique(groups_b))} bearings")
        log.info(f"  피처: {env_features}")

        # Optuna
        section("실험 B — Optuna 최적화")
        best_xgb_b = optuna_xgboost_reg(X_b, y_b, groups_b, n_trials=60)

        # 평가
        section("실험 B — LOBO-CV 평가")

        log.info("  [XGBoost + 환경보정]")
        results_b['XGB+Env'] = evaluate_regression(
            lambda: xgb.XGBRegressor(**best_xgb_b, random_state=42, verbosity=0, n_jobs=-1),
            X_b, y_b, groups_b)

        # 비교: 환경보정 없이
        log.info("  [XGBoost 진동만 (비교용)]")
        X_b_vib = df_b[common_features].values.astype(np.float32)
        results_b['XGB VibOnly'] = evaluate_regression(
            lambda: xgb.XGBRegressor(**best_xgb_a, random_state=42, verbosity=0, n_jobs=-1),
            X_b_vib, y_b, groups_b)
    else:
        log.warning("  환경보정 데이터 부족 — 실험 B 생략")

    # ══════════════════════════════════════════
    # 결과 정리
    # ══════════════════════════════════════════
    section("결과 요약")

    all_results = {}
    for exp_name, results in [("실험A", results_a), ("실험B", results_b)]:
        if not results:
            continue
        log.info(f"\n  [{exp_name}]")
        log.info(f"  {'모델':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'±0.1Hit':>8} {'Gap(MAE)':>10}")
        log.info("  " + "-" * 65)
        for m, r in results.items():
            mae = np.mean(r['mae_scores'])
            rmse = np.mean(r['rmse_scores'])
            r2 = np.mean(r['r2_scores'])
            tol = np.mean(r['tol_acc_scores'])
            gap = np.mean(r['overfit_gaps'])
            log.info(f"  {m:<20} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {tol:>8.1%} {gap:>10.4f}")

            # 구간별 적중률
            yt = np.array(r['all_y_true'])
            yp = np.array(r['all_y_pred'])
            for lo, hi, name in [(0, 0.15, 'CRITICAL'), (0.15, 0.4, 'HIGH'),
                                  (0.4, 0.7, 'MEDIUM'), (0.7, 1.0, 'LOW')]:
                mask = (yt >= lo) & (yt < hi)
                if mask.sum() > 0:
                    zone_hit = float(np.mean(np.abs(yt[mask] - yp[mask]) <= 0.1))
                    zone_mae = float(np.mean(np.abs(yt[mask] - yp[mask])))
                    log.info(f"    {name:>10} ({lo:.2f}~{hi:.2f}): "
                             f"n={mask.sum():>6,}  MAE={zone_mae:.4f}  ±0.1Hit={zone_hit:.1%}")

        all_results[exp_name] = {
            m: {
                "mean_mae": float(np.mean(r['mae_scores'])),
                "mean_rmse": float(np.mean(r['rmse_scores'])),
                "mean_r2": float(np.mean(r['r2_scores'])),
                "mean_tol_acc": float(np.mean(r['tol_acc_scores'])),
                "mean_gap": float(np.mean(r['overfit_gaps'])),
                "fold_mae": [float(x) for x in r['mae_scores']],
                "fold_r2": [float(x) for x in r['r2_scores']],
                "fold_tol_acc": [float(x) for x in r['tol_acc_scores']],
            }
            for m, r in results.items()
        }

    # 환경보정 효과 비교
    if results_b and 'XGB+Env' in results_b and 'XGB VibOnly' in results_b:
        env_mae = np.mean(results_b['XGB+Env']['mae_scores'])
        vib_mae = np.mean(results_b['XGB VibOnly']['mae_scores'])
        improvement = vib_mae - env_mae  # 양수면 환경보정이 더 좋음
        log.info(f"\n  [환경보정 효과]")
        log.info(f"    진동만: MAE={vib_mae:.4f}")
        log.info(f"    +환경보정: MAE={env_mae:.4f}")
        log.info(f"    개선: MAE {improvement:+.4f}")
        all_results['env_improvement_mae'] = float(improvement)

    # 시각화 저장
    plot_regression_results(results_a, "Experiment A: All RTF Regression",
                            os.path.join(OUT_DIR, "exp_a_regression_chart.png"))
    if results_b:
        plot_regression_results(results_b, "Experiment B: Env Correction Regression",
                                os.path.join(OUT_DIR, "exp_b_regression_chart.png"))

    # 최고 모델 Scatter Plot
    best_model = min(results_a.keys(), key=lambda m: np.mean(results_a[m]['mae_scores']))
    plot_scatter(results_a, best_model,
                 os.path.join(OUT_DIR, f"exp_a_{best_model}_scatter.png"))
    if results_b:
        best_b = min(results_b.keys(), key=lambda m: np.mean(results_b[m]['mae_scores']))
        plot_scatter(results_b, best_b,
                     os.path.join(OUT_DIR, f"exp_b_{best_b}_scatter.png"))

    # JSON 저장
    json_path = os.path.join(OUT_DIR, "comparison_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    log.info(f"  [저장] {json_path}")

    elapsed = (datetime.datetime.now() - start).total_seconds() / 60
    log.info(f"\n  총 소요시간: {elapsed:.1f}분")


if __name__ == "__main__":
    main()
