"""
═══════════════════════════════════════════════════════════════════
  고장모드 분류 모델 비교 실험 (v2)
  AI-PASS 예지보전 | XJTU-SY + Paderborn = 47 bearings
═══════════════════════════════════════════════════════════════════

변경사항 (v1 대비):
  - Paderborn 추가 (진동 + 전류 + 온도)
  - 전류 FFT 피처 추가 (Paderborn만)
  - 샘플링 레이트 차이 → FFT 피처 레벨 통합
  - 클래스: outer_race, inner_race, ball, cage, (healthy)

출력:
  D:/project/예지보전_v2/compare_clf_v2/
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

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score as f1_score_fn
import xgboost as xgb
import optuna

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import load_all_classification, WINDOW_SIZE
from feature_engineering import compute_signal_stats, compute_freq_features

OUT_DIR = r"D:\project\예지보전_v2\compare_clf_v2"
os.makedirs(OUT_DIR, exist_ok=True)

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


def section(title):
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ══════════════════════════════════════════════
# Optuna 최적화
# ══════════════════════════════════════════════
def optuna_xgboost_clf(X, y, n_splits=5, n_trials=60):
    """XGBoost 고장분류 Optuna 최적화"""
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
            'eval_metric': 'mlogloss',
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y[tr])
            scores.append(f1_score_fn(y[te], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna XGB] Best F1={study.best_value:.4f}")
    return study.best_params


def optuna_rf_clf(X, y, n_splits=3, n_trials=15):
    """RandomForest 고장분류 Optuna 최적화 (탐색: 3-fold 15trials, n_estimators 고정)"""
    def objective(trial):
        params = {
            'n_estimators': 400,  # 고정 (많을수록 좋음, 탐색 불필요)
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
            'max_features': trial.suggest_float('max_features', 0.3, 0.7),
            'random_state': 42, 'n_jobs': -1,
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y[tr])
            scores.append(f1_score_fn(y[te], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    log.info(f"  [Optuna RF] Best F1={study.best_value:.4f}")
    return study.best_params


# ══════════════════════════════════════════════
# CNN-1D
# ══════════════════════════════════════════════
def build_cnn(n_classes, win_size=WINDOW_SIZE):
    inp = keras.Input(shape=(win_size, 1))
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════
# 평가 함수
# ══════════════════════════════════════════════
def evaluate_cnn(X_raw, y, n_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list, gap_list = [], [], []
    all_yt, all_yp = [], []

    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        X_tr = X_raw[tr][..., np.newaxis]
        X_te = X_raw[te][..., np.newaxis]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, WINDOW_SIZE)).reshape(X_tr.shape)
        X_te = scaler.transform(X_te.reshape(-1, WINDOW_SIZE)).reshape(X_te.shape)

        model = build_cnn(n_classes)
        model.fit(X_tr, y[tr], validation_data=(X_te, y[te]),
                  epochs=50, batch_size=256, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor='val_accuracy',
                             patience=10, restore_best_weights=True, verbose=0)])

        pred_tr = np.argmax(model.predict(X_tr, verbose=0), axis=1)
        pred_te = np.argmax(model.predict(X_te, verbose=0), axis=1)
        tr_acc = accuracy_score(y[tr], pred_tr)
        te_acc = accuracy_score(y[te], pred_te)
        te_f1 = f1_score(y[te], pred_te, average='macro')

        acc_list.append(te_acc); f1_list.append(te_f1); gap_list.append(tr_acc - te_acc)
        all_yt.extend(y[te].tolist()); all_yp.extend(pred_te.tolist())
        log.info(f"    Fold {fold+1}: Acc={te_acc:.4f} F1={te_f1:.4f} gap={tr_acc-te_acc:.4f}")
        keras.backend.clear_session()

    return {"acc_scores": acc_list, "f1_scores": f1_list, "overfit_gaps": gap_list,
            "all_y_true": all_yt, "all_y_pred": all_yp}


def evaluate_tree(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list, gap_list = [], [], []
    all_yt, all_yp = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        model = model_fn()
        model.fit(X_tr, y[tr])
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        tr_acc = accuracy_score(y[tr], pred_tr)
        te_acc = accuracy_score(y[te], pred_te)
        te_f1 = f1_score(y[te], pred_te, average='macro')

        acc_list.append(te_acc); f1_list.append(te_f1); gap_list.append(tr_acc - te_acc)
        all_yt.extend(y[te].tolist()); all_yp.extend(pred_te.tolist())
        log.info(f"    Fold {fold+1}: Acc={te_acc:.4f} F1={te_f1:.4f} gap={tr_acc-te_acc:.4f}")

    return {"acc_scores": acc_list, "f1_scores": f1_list, "overfit_gaps": gap_list,
            "all_y_true": all_yt, "all_y_pred": all_yp}


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def plot_comparison(results, class_names, save_path):
    model_names = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(model_names)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Fault Classification (XJTU-SY + Paderborn)', fontsize=13, fontweight='bold')

    for ax_idx, (metric, title) in enumerate([
        ('acc_scores', 'Accuracy'), ('f1_scores', 'F1-Macro'), ('overfit_gaps', 'Overfit Gap')]):
        ax = axes[0, ax_idx]
        data = [results[m][metric] for m in model_names]
        bp = ax.boxplot(data, labels=model_names, patch_artist=True)
        for p, c in zip(bp['boxes'], colors):
            p.set_facecolor(c); p.set_alpha(0.7)
        ax.set_title(title); ax.grid(axis='y', alpha=0.3)

    for idx, m in enumerate(model_names):
        ax = axes[1, idx]
        cm = confusion_matrix(results[m]['all_y_true'], results[m]['all_y_pred'])
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(f'{m} Confusion Matrix')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    log.info(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"고장모드 분류 비교 v2 — {start.strftime('%Y-%m-%d %H:%M')}")

    # 데이터 로드
    section("Step 1. 데이터 로드")
    df = load_all_classification()
    if df.empty:
        log.error("데이터 없음"); return

    # 클래스 균형화
    section("Step 2. 클래스 균형화")
    # healthy 제외 옵션 (고장 유형만 분류할 경우)
    df_fault = df[df['fault_label'] != 'healthy'].copy()
    min_count = df_fault['fault_label'].value_counts().min()
    sampled = []
    for label, g in df_fault.groupby('fault_label'):
        sampled.append(g.sample(min(len(g), min_count), random_state=42))
    df_bal = pd.concat(sampled, ignore_index=True)
    log.info(f"  분포: {df_bal['fault_label'].value_counts().to_dict()}")

    le = LabelEncoder()
    y = le.fit_transform(df_bal['fault_label'].values)
    class_names = le.classes_.tolist()
    n_classes = len(class_names)
    log.info(f"  클래스: {class_names}")

    # Raw 준비
    X_raw = np.stack(df_bal['raw'].values).astype(np.float32)

    section("Step 3. 요약 피처 추출 (13개 코어 + 전류)")
    # 데이터셋별 샘플링 레이트 매핑
    fs_map = {'XJTU': 25600.0, 'PADERBORN': 64000.0}
    sources = df_bal['source'].values

    n = len(X_raw)
    X_summary = np.zeros((n, 13))
    for i in range(n):
        w = X_raw[i]
        fs = fs_map.get(sources[i], 25600.0)
        stats = compute_signal_stats(w)
        freq = compute_freq_features(w, fs=fs)
        X_summary[i] = [
            stats['vibration_std'], stats['vibration_peak'],
            stats['vibration_kurtosis'], stats['vibration_skewness'],
            stats['crest_factor'], stats['impulse_factor'], stats['shape_factor'],
            freq['spectral_energy'], freq['spectral_centroid'], freq['spectral_spread'],
            freq['band_energy_low'], freq['band_energy_mid'], freq['band_energy_high'],
        ]
    log.info(f"  X_summary shape: {X_summary.shape}")

    # 전류 피처 추가 (Paderborn만)
    if 'current_rms' in df_bal.columns:
        current_vals = df_bal['current_rms'].fillna(0).values.reshape(-1, 1)
        X_summary = np.hstack([X_summary, current_vals])
        log.info(f"  전류 피처 추가 → {X_summary.shape}")

    X_fft = X_summary  # 이후 코드 호환을 위해 변수명 유지

    # Optuna 최적화
    section("Step 4. Optuna 최적화")
    best_xgb = optuna_xgboost_clf(X_fft, y)
    best_rf = optuna_rf_clf(X_fft, y)

    # 평가
    results = {}
    section("Model 1: CNN-1D (Raw)")
    results['CNN-1D'] = evaluate_cnn(X_raw, y, n_classes)

    section("Model 2: XGBoost+FFT (Optuna)")
    results['XGB+FFT'] = evaluate_tree(
        lambda: xgb.XGBClassifier(**best_xgb, random_state=42, verbosity=0,
                                   n_jobs=-1, eval_metric='mlogloss'), X_fft, y)

    section("Model 3: RF+FFT (Optuna)")
    results['RF+FFT'] = evaluate_tree(
        lambda: RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1),
        X_fft, y)

    # 결과
    section("결과 요약")
    summary = {}
    for m, r in results.items():
        acc = np.mean(r['acc_scores']); f1 = np.mean(r['f1_scores'])
        gap = np.mean(r['overfit_gaps'])
        log.info(f"  {m:<15} Acc={acc:.4f} F1={f1:.4f} Gap={gap:.4f}")
        report = classification_report(r['all_y_true'], r['all_y_pred'],
                                        target_names=class_names, digits=4)
        log.info(f"\n{report}")
        summary[m] = {"mean_acc": float(acc), "mean_f1": float(f1), "mean_gap": float(gap)}

    plot_comparison(results, class_names, os.path.join(OUT_DIR, "comparison_chart.png"))
    with open(os.path.join(OUT_DIR, "comparison_results.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"\n  소요시간: {(datetime.datetime.now()-start).total_seconds()/60:.1f}분")


if __name__ == "__main__":
    main()
