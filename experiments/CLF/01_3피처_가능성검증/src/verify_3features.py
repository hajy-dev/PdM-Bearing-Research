"""
═══════════════════════════════════════════════════════════════════
  3피처 가능성 검증 실험
  Paderborn 32 bearings | vibration_rms + temperature + current_rms
═══════════════════════════════════════════════════════════════════

목적:
  DB의 3개 값(vibration, temperature, motor_current)만으로
  이상탐지 + 고장분류가 가능한지 검증

실험 구성:
  1. 이상탐지: healthy vs fault (2클래스) — Autoencoder + Threshold
  2. 고장분류A: inner_race / outer_race / ball (3클래스, healthy 제외)
  3. 고장분류B: 베어링이상 / 정상 (2클래스, 사용자 시나리오 기반)

데이터:
  Paderborn 32 bearings (healthy 6, outer_race 12, inner_race 11, ball 3)
  채널: vibration_1(rms), temp_2_bearing_module, phase_current_1(rms)

출력:
  D:/project/예지보전_v2/증명_3피처_가능성검증/results/
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
import scipy.io
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score)
import xgboost as xgb
import optuna
from tqdm import tqdm

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 경로
PADERBORN_DIR = r"D:\project\데이터셋\Paderborn Univ_Bearing Dataset"
OUT_DIR = r"D:\project\예지보전_v2\compare_3feature_v1"
os.makedirs(OUT_DIR, exist_ok=True)

WINDOW_SIZE = 256
STEP = 128

LABEL_MAP = {"K": "healthy", "KA": "outer_race", "KI": "inner_race", "KB": "ball"}

# ── 로거
log_path = os.path.join(OUT_DIR, "experiment_log.txt")
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


def compute_rms(signal):
    return float(np.sqrt(np.mean(signal ** 2)))


# ══════════════════════════════════════════════
# 1. 데이터 로드: Paderborn → 3피처 추출
# ══════════════════════════════════════════════
def load_paderborn_3features():
    """Paderborn에서 vibration_rms, temperature, current_rms 3피처만 추출"""
    records = []
    folders = sorted([d for d in os.listdir(PADERBORN_DIR)
                      if os.path.isdir(os.path.join(PADERBORN_DIR, d))])

    for folder in tqdm(folders, desc="[Paderborn 3피처]", unit="bearing"):
        prefix = folder[:2] if folder[:2] in ('KA', 'KI', 'KB') else folder[:1]
        label = LABEL_MAP.get(prefix)
        if label is None:
            continue

        mat_files = sorted(glob.glob(os.path.join(PADERBORN_DIR, folder, "*.mat")))
        for fp in mat_files:
            try:
                mat = scipy.io.loadmat(fp)
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                data = mat[key][0, 0]
                Y = data['Y']

                vib_data = None
                current_data = None
                temp_data = None

                for i in range(Y.shape[1]):
                    ch = Y[0, i]
                    name = str(ch['Name'][0]) if ch['Name'].size > 0 else ''
                    d = ch['Data'].flatten()
                    if 'vibration' in name:
                        vib_data = d
                    elif 'phase_current_1' in name:
                        current_data = d
                    elif 'temp_2' in name:
                        temp_data = d

                if vib_data is None:
                    continue

                # 파일 단위 temperature (평균)
                file_temp = float(np.mean(temp_data)) if temp_data is not None else np.nan

                # 윈도우 슬라이싱
                for wi in range(0, len(vib_data) - WINDOW_SIZE, STEP):
                    raw_vib = vib_data[wi:wi + WINDOW_SIZE]
                    vib_rms = compute_rms(raw_vib)

                    cur_rms = np.nan
                    if current_data is not None and len(current_data) > wi + WINDOW_SIZE:
                        raw_cur = current_data[wi:wi + WINDOW_SIZE]
                        cur_rms = compute_rms(raw_cur)

                    records.append({
                        "bearing_id": folder,
                        "fault_label": label,
                        "vibration_rms": vib_rms,
                        "temperature": file_temp,
                        "current_rms": cur_rms,
                    })
            except Exception:
                continue

    df = pd.DataFrame(records)
    df = df.dropna(subset=['vibration_rms', 'temperature', 'current_rms'])
    return df


# ══════════════════════════════════════════════
# 2. 이상탐지 실험: healthy vs fault
# ══════════════════════════════════════════════
def experiment_anomaly_detection(df):
    """3피처로 이상탐지 (healthy=0, fault=1) — bearing 단위 StratifiedGroupKFold"""
    section("실험 1: 이상탐지 (healthy vs fault)")

    df_exp = df.copy()
    df_exp['label'] = (df_exp['fault_label'] != 'healthy').astype(int)

    # bearing 단위 라벨 (StratifiedGroupKFold용)
    bearing_labels = df_exp.groupby('bearing_id')['label'].first().values

    features = ['vibration_rms', 'temperature', 'current_rms']
    X = df_exp[features].values.astype(np.float32)
    y = df_exp['label'].values
    groups = df_exp['bearing_id'].values

    n_bearings = len(np.unique(groups))
    log.info(f"  데이터: {len(X):,} rows / {n_bearings} bearings")
    log.info(f"  정상 베어링: {(bearing_labels==0).sum()}개 / 이상 베어링: {(bearing_labels==1).sum()}개")
    log.info(f"  정상 rows: {(y==0).sum():,} / 이상 rows: {(y==1).sum():,}")

    n_splits = min(n_bearings, 5)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list, auc_list = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=0, n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_te)
        prob = model.predict_proba(X_te)[:, 1]

        acc = accuracy_score(y[te_idx], pred)
        f1 = f1_score(y[te_idx], pred, average='binary')
        auc = roc_auc_score(y[te_idx], prob)

        acc_list.append(acc)
        f1_list.append(f1)
        auc_list.append(auc)
        test_bears = sorted(set(groups[te_idx]))
        n_healthy_test = (y[te_idx] == 0).sum()
        n_fault_test = (y[te_idx] == 1).sum()
        log.info(f"    Fold {fold+1}: bearings={test_bears[:5]}{'...' if len(test_bears)>5 else ''}  "
                 f"healthy={n_healthy_test:,}/fault={n_fault_test:,}  "
                 f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    log.info(f"\n  [이상탐지 결과]")
    log.info(f"    Mean Acc:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    log.info(f"    Mean F1:   {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    log.info(f"    Mean AUC:  {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")

    # Feature Importance
    log.info(f"\n  [Feature Importance]")
    importance = model.feature_importances_
    for fname, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        log.info(f"    {fname}: {imp:.4f}")

    return {
        "acc": float(np.mean(acc_list)), "f1": float(np.mean(f1_list)),
        "auc": float(np.mean(auc_list)),
        "fold_acc": [float(x) for x in acc_list],
        "fold_f1": [float(x) for x in f1_list],
        "fold_auc": [float(x) for x in auc_list],
    }


# ══════════════════════════════════════════════
# 3. 고장분류A: inner_race / outer_race / ball (세부)
# ══════════════════════════════════════════════
def experiment_fault_classification_detail(df):
    """3피처로 세부 고장분류 (inner/outer/ball, healthy 제외) — bearing 단위 StratifiedGroupKFold"""
    section("실험 2: 고장분류 세부 (inner_race / outer_race / ball)")

    df_exp = df[df['fault_label'] != 'healthy'].copy()
    label_map = {'inner_race': 0, 'outer_race': 1, 'ball': 2}
    df_exp['label'] = df_exp['fault_label'].map(label_map)

    features = ['vibration_rms', 'temperature', 'current_rms']
    X = df_exp[features].values.astype(np.float32)
    y = df_exp['label'].values
    groups = df_exp['bearing_id'].values

    n_bearings = len(np.unique(groups))
    log.info(f"  데이터: {len(X):,} rows / {n_bearings} bearings")
    for name, code in label_map.items():
        n_bears = len(set(groups[y == code]))
        log.info(f"    {name}: {(y==code).sum():,} rows / {n_bears} bearings")

    n_splits = min(n_bearings, 5)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list = [], []
    all_y_true, all_y_pred = [], []

    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups)):
        X_tr, X_te = X[tr_idx], X[te_idx]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=5.0,
            random_state=42, verbosity=0, n_jobs=-1,
            eval_metric='mlogloss'
        )
        model.fit(X_tr, y[tr_idx])
        pred = model.predict(X_te)

        acc = accuracy_score(y[te_idx], pred)
        f1 = f1_score(y[te_idx], pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(pred.tolist())
        test_bears = sorted(set(groups[te_idx]))
        log.info(f"    Fold {fold+1}: bearings={test_bears[:5]}{'...' if len(test_bears)>5 else ''}  "
                 f"Acc={acc:.4f}  F1-Macro={f1:.4f}")

    log.info(f"\n  [세부 고장분류 결과]")
    log.info(f"    Mean Acc:  {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    log.info(f"    Mean F1:   {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")

    report = classification_report(all_y_true, all_y_pred,
                                    target_names=['inner_race', 'outer_race', 'ball'], digits=4)
    log.info(f"\n{report}")

    # Confusion Matrix 저장
    cm = confusion_matrix(all_y_true, all_y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=['inner_race', 'outer_race', 'ball'])
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('3-Feature Fault Classification (Detail)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fault_detail_confusion.png"), dpi=150)
    plt.close()

    # Feature Importance
    log.info(f"\n  [Feature Importance]")
    importance = model.feature_importances_
    for fname, imp in sorted(zip(features, importance), key=lambda x: -x[1]):
        log.info(f"    {fname}: {imp:.4f}")

    return {
        "acc": float(np.mean(acc_list)), "f1": float(np.mean(f1_list)),
        "fold_acc": [float(x) for x in acc_list],
        "fold_f1": [float(x) for x in f1_list],
    }


# ══════════════════════════════════════════════
# 4. 3피처 분포 분석
# ══════════════════════════════════════════════
def analyze_feature_distribution(df):
    """고장 유형별 3피처 분포를 시각화"""
    section("피처 분포 분석")

    features = ['vibration_rms', 'temperature', 'current_rms']
    labels = df['fault_label'].unique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('3-Feature Distribution by Fault Type', fontsize=13, fontweight='bold')

    colors = {'healthy': '#4CAF50', 'inner_race': '#f44336',
              'outer_race': '#FF9800', 'ball': '#2196F3'}

    for idx, feat in enumerate(features):
        ax = axes[idx]
        for label in sorted(labels):
            vals = df[df['fault_label'] == label][feat]
            ax.hist(vals, bins=50, alpha=0.5, label=label,
                    color=colors.get(label, 'gray'), density=True)
        ax.set_title(feat)
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_distribution.png"), dpi=150)
    plt.close()
    log.info(f"  [저장] feature_distribution.png")

    # 통계 요약
    for feat in features:
        log.info(f"\n  [{feat}]")
        for label in sorted(labels):
            vals = df[df['fault_label'] == label][feat]
            log.info(f"    {label:>12}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                     f"min={vals.min():.4f}  max={vals.max():.4f}")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"3피처 가능성 검증 실험 — {start.strftime('%Y-%m-%d %H:%M')}")

    # 1. 데이터 로드
    section("Step 1. Paderborn 3피처 추출")
    df = load_paderborn_3features()
    log.info(f"  전체: {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    log.info(f"  라벨 분포:\n{df['fault_label'].value_counts().to_string()}")
    log.info(f"  피처: vibration_rms, temperature, current_rms")

    # 2. 분포 분석
    analyze_feature_distribution(df)

    # 3. 실험 실행
    results = {}
    results['anomaly_detection'] = experiment_anomaly_detection(df)
    results['fault_detail'] = experiment_fault_classification_detail(df)

    # 4. 종합 결과
    section("종합 결과")
    log.info(f"\n  {'실험':<25} {'Acc':>8} {'F1':>8} {'AUC':>8}")
    log.info("  " + "-" * 55)
    for name, r in results.items():
        auc_str = f"{r.get('auc', 0):.4f}" if 'auc' in r else "  N/A"
        log.info(f"  {name:<25} {r['acc']:>8.4f} {r['f1']:>8.4f} {auc_str:>8}")

    # 5. JSON 저장
    json_path = os.path.join(OUT_DIR, "experiment_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"\n  [저장] {json_path}")

    elapsed = (datetime.datetime.now() - start).total_seconds() / 60
    log.info(f"\n  총 소요시간: {elapsed:.1f}분")


if __name__ == "__main__":
    main()
