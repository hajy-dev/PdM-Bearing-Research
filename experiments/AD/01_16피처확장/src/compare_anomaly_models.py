"""
═══════════════════════════════════════════════════════════════════
  이상 탐지 모델 비교 실험 (v2)
  AI-PASS 예지보전 | FEMTO + IMS + Zenodo + KAIST
═══════════════════════════════════════════════════════════════════

변경사항 (v1 대비):
  - IMS, Zenodo 데이터 추가 → 정상/이상 데이터 대폭 확대
  - 피처 확장 (시간+주파수 도메인)
  - 환경 보정 피처 포함 (KAIST+Zenodo)

출력:
  D:/project/예지보전_v2/compare_ad_v2/
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import load_all_rtf
from feature_engineering import add_trend_features

OUT_DIR = r"D:\project\예지보전_v2\compare_ad_v2"
os.makedirs(OUT_DIR, exist_ok=True)

NORMAL_THRESHOLD = 0.5
ANOMALY_THRESHOLD = 0.2

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
# 모델
# ══════════════════════════════════════════════
def build_ae(n_feat):
    inp = keras.Input(shape=(n_feat,))
    e = layers.Dense(32, activation='relu')(inp)
    e = layers.BatchNormalization()(e)
    e = layers.Dense(16, activation='relu')(e)
    e = layers.BatchNormalization()(e)
    code = layers.Dense(8, activation='relu')(e)
    d = layers.Dense(16, activation='relu')(code)
    d = layers.BatchNormalization()(d)
    d = layers.Dense(32, activation='relu')(d)
    d = layers.BatchNormalization()(d)
    out = layers.Dense(n_feat, activation='linear')(d)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='mse')
    return model


def _metrics(pred_n, pred_a, score_n, score_a):
    y_true = np.concatenate([np.zeros(len(pred_n)), np.ones(len(pred_a))])
    y_pred = np.concatenate([(pred_n == -1).astype(int), (pred_a == -1).astype(int)])
    scores = np.concatenate([-score_n, -score_a])
    try:
        auc = roc_auc_score(y_true, scores)
    except:
        auc = 0.5
    return {
        "det_rate": float(recall_score(y_true, y_pred)),
        "fpr": float(np.mean(pred_n == -1)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(auc),
    }


def eval_if(X_tr, X_te_n, X_te_a):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(Xtr)
    return _metrics(model.predict(sc.transform(X_te_n)), model.predict(sc.transform(X_te_a)),
                    model.decision_function(sc.transform(X_te_n)),
                    model.decision_function(sc.transform(X_te_a)))


def eval_ae(X_tr, X_te_n, X_te_a):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    model = build_ae(Xtr.shape[1])
    model.fit(Xtr, Xtr, validation_split=0.1, epochs=200, batch_size=256, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=15,
                         restore_best_weights=True, verbose=0)])
    tr_err = np.mean((Xtr - model.predict(Xtr, verbose=0))**2, axis=1)
    threshold = float(np.percentile(tr_err, 95))
    Xn = sc.transform(X_te_n); Xa = sc.transform(X_te_a)
    err_n = np.mean((Xn - model.predict(Xn, verbose=0))**2, axis=1)
    err_a = np.mean((Xa - model.predict(Xa, verbose=0))**2, axis=1)
    pred_n = np.where(err_n > threshold, -1, 1)
    pred_a = np.where(err_a > threshold, -1, 1)
    keras.backend.clear_session()
    return _metrics(pred_n, pred_a, -err_n, -err_a)


def eval_ocsvm(X_tr, X_te_n, X_te_a):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    if len(Xtr) > 5000:
        idx = np.random.RandomState(42).choice(len(Xtr), 5000, replace=False)
        Xtr = Xtr[idx]
    model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    model.fit(Xtr)
    Xn = sc.transform(X_te_n); Xa = sc.transform(X_te_a)
    return _metrics(model.predict(Xn), model.predict(Xa),
                    model.decision_function(Xn), model.decision_function(Xa))


def run_repeated(X_n, X_a, name, eval_fn, n=5):
    metrics = {"det_rates": [], "fprs": [], "f1s": [], "aucs": []}
    for i in range(n):
        idx = np.random.RandomState(42+i).permutation(len(X_n))
        split = int(len(X_n) * 0.8)
        r = eval_fn(X_n[idx[:split]], X_n[idx[split:]], X_a)
        metrics["det_rates"].append(r["det_rate"])
        metrics["fprs"].append(r["fpr"])
        metrics["f1s"].append(r["f1"])
        metrics["aucs"].append(r["auc"])
        log.info(f"    R{i+1}: Det={r['det_rate']:.4f} FPR={r['fpr']:.4f} "
                 f"F1={r['f1']:.4f} AUC={r['auc']:.4f}")
    return metrics


# ══════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════
def plot_results(results, save_path):
    names = list(results.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Anomaly Detection Comparison', fontsize=13, fontweight='bold')
    for (key, title), ax in zip([('det_rates','Detection Rate'), ('fprs','FPR'),
                                  ('f1s','F1'), ('aucs','AUC-ROC')], axes.flat):
        data = [results[m][key] for m in names]
        bp = ax.boxplot(data, labels=names, patch_artist=True)
        for p, c in zip(bp['boxes'], colors):
            p.set_facecolor(c); p.set_alpha(0.7)
        ax.set_title(title); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    log.info(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"이상 탐지 비교 v2 — {start.strftime('%Y-%m-%d %H:%M')}")

    section("Step 1. 데이터 로드")
    df = load_all_rtf()
    if df.empty:
        log.error("데이터 없음"); return

    section("Step 2. 피처 엔지니어링")
    df = add_trend_features(df)

    features = [
        'vibration_rms', 'short_trend', 'rolling_trend',
        # 시간 도메인 통계
        'vibration_std', 'vibration_peak', 'vibration_kurtosis',
        'vibration_skewness', 'crest_factor', 'impulse_factor', 'shape_factor',
        # 주파수 도메인
        'spectral_energy', 'spectral_centroid', 'spectral_spread',
        'band_energy_low', 'band_energy_mid', 'band_energy_high',
    ]
    if 'temp_residual' in df.columns:
        df['temp_residual'] = df['temp_residual'].fillna(0)
        features.append('temp_residual')

    df = df.dropna(subset=features)

    section("Step 3. 정상/이상 분리")
    df_n = df[df['rul_ratio'] > NORMAL_THRESHOLD]
    df_a = df[df['rul_ratio'] <= ANOMALY_THRESHOLD]
    log.info(f"  정상: {len(df_n):,} / 이상: {len(df_a):,}")

    if len(df_n) < 50 or len(df_a) < 20:
        log.error("데이터 부족"); return

    X_n = df_n[features].values.astype(np.float32)
    X_a = df_a[features].values.astype(np.float32)

    results = {}
    section("Model 1: Isolation Forest")
    results['IsolationForest'] = run_repeated(X_n, X_a, 'IF', eval_if)

    section("Model 2: Autoencoder")
    results['Autoencoder'] = run_repeated(X_n, X_a, 'AE', eval_ae)

    section("Model 3: One-Class SVM")
    results['OneClassSVM'] = run_repeated(X_n, X_a, 'OCSVM', eval_ocsvm)

    section("결과 요약")
    summary = {}
    for m, r in results.items():
        det = np.mean(r['det_rates']); fpr = np.mean(r['fprs'])
        f1 = np.mean(r['f1s']); auc = np.mean(r['aucs'])
        log.info(f"  {m:<18} Det={det:.4f} FPR={fpr:.4f} F1={f1:.4f} AUC={auc:.4f}")
        summary[m] = {"det_rate": float(det), "fpr": float(fpr),
                       "f1": float(f1), "auc": float(auc)}

    plot_results(results, os.path.join(OUT_DIR, "comparison_chart.png"))
    with open(os.path.join(OUT_DIR, "comparison_results.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log.info(f"\n  소요시간: {(datetime.datetime.now()-start).total_seconds()/60:.1f}분")


if __name__ == "__main__":
    main()
