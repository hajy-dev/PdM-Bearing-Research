"""
═══════════════════════════════════════════════════════════════════
  3피처 Autoencoder 이상탐지 실험
  Paderborn 32 bearings | vibration_rms + temperature + current_rms
═══════════════════════════════════════════════════════════════════

목적:
  3피처로 Autoencoder 기반 이상탐지가 가능한지 검증
  이전 XGBoost 분류(AUC 0.56)와 비교

방식:
  - 정상(healthy) 데이터로만 Autoencoder 학습
  - 복원 오차(reconstruction error)가 threshold 초과 시 이상 판정
  - GroupKFold: bearing 단위 분할 (leakage 방지)

출력:
  D:/project/예지보전_v2/compare_3feature_autoencoder/
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

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── 경로
PADERBORN_DIR = r"D:\project\데이터셋\Paderborn Univ_Bearing Dataset"
OUT_DIR = r"D:\project\예지보전_v2\compare_3feature_autoencoder"
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
# 1. 데이터 로드 (이전과 동일)
# ══════════════════════════════════════════════
def load_paderborn_3features():
    """Paderborn에서 vibration_rms, temperature, current_rms 3피처 추출"""
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

                vib_data, current_data, temp_data = None, None, None
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

                file_temp = float(np.mean(temp_data)) if temp_data is not None else np.nan

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
# 2. Autoencoder 모델
# ══════════════════════════════════════════════
def build_autoencoder(input_dim=3):
    """3피처 Autoencoder (3 → 2 → 3)"""
    encoder = keras.Sequential([
        keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(2, activation='relu'),  # bottleneck
    ])
    decoder = keras.Sequential([
        keras.layers.Dense(4, activation='relu', input_shape=(2,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(input_dim, activation='linear'),
    ])

    inputs = keras.Input(shape=(input_dim,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder


# ══════════════════════════════════════════════
# 3. Autoencoder 이상탐지 실험
# ══════════════════════════════════════════════
def experiment_autoencoder_anomaly(df):
    """
    Autoencoder 이상탐지:
    - 정상(healthy) 데이터로만 학습
    - 복원 오차로 이상 판단
    - bearing 단위 GroupKFold
    """
    section("실험: 3피처 Autoencoder 이상탐지")

    df_exp = df.copy()
    df_exp['label'] = (df_exp['fault_label'] != 'healthy').astype(int)

    features = ['vibration_rms', 'temperature', 'current_rms']

    # bearing 단위 정보
    healthy_bearings = sorted(df_exp[df_exp['label'] == 0]['bearing_id'].unique())
    fault_bearings = sorted(df_exp[df_exp['label'] == 1]['bearing_id'].unique())
    log.info(f"  정상 베어링: {len(healthy_bearings)}개 {healthy_bearings}")
    log.info(f"  이상 베어링: {len(fault_bearings)}개")
    log.info(f"  정상 rows: {(df_exp['label']==0).sum():,} / 이상 rows: {(df_exp['label']==1).sum():,}")

    # healthy bearing을 기준으로 fold 분할
    # 각 fold에서 1개 healthy를 test로 빼고, 나머지 healthy로 학습
    # fault bearing은 전부 test에 포함 (이상 탐지 평가용)
    n_healthy = len(healthy_bearings)
    auc_list, acc_list, f1_list = [], [], []
    precision_list, recall_list = [], []

    log.info(f"\n  Leave-One-Healthy-Bearing-Out ({n_healthy} folds)")

    for fold_idx, test_healthy in enumerate(healthy_bearings):
        # Train: 나머지 healthy bearings의 정상 데이터만
        train_healthy = [b for b in healthy_bearings if b != test_healthy]
        train_mask = df_exp['bearing_id'].isin(train_healthy) & (df_exp['label'] == 0)
        X_train_raw = df_exp.loc[train_mask, features].values.astype(np.float32)

        # Test: test_healthy(정상) + 일부 fault bearings
        # 각 fold에서 fault bearings 중 ~5개씩 사용
        np.random.seed(fold_idx)
        test_fault = list(np.random.choice(fault_bearings, min(5, len(fault_bearings)), replace=False))
        test_bearings = [test_healthy] + test_fault

        test_mask = df_exp['bearing_id'].isin(test_bearings)
        X_test_raw = df_exp.loc[test_mask, features].values.astype(np.float32)
        y_test = df_exp.loc[test_mask, 'label'].values

        # 정규화 (train 정상 기준)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # Autoencoder 학습 (정상 데이터만)
        model = build_autoencoder(input_dim=3)
        model.fit(X_train, X_train,
                  epochs=50, batch_size=256,
                  validation_split=0.1,
                  verbose=0)

        # 복원 오차 계산
        X_reconstructed = model.predict(X_test, verbose=0)
        mse = np.mean((X_test - X_reconstructed) ** 2, axis=1)

        # AUC 계산 (mse가 높을수록 이상)
        if len(np.unique(y_test)) < 2:
            log.info(f"    Fold {fold_idx+1}: 단일 클래스 — skip")
            continue

        auc = roc_auc_score(y_test, mse)
        auc_list.append(auc)

        # 최적 threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_test, mse)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]

        pred = (mse >= best_threshold).astype(int)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='binary')
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)

        acc_list.append(acc)
        f1_list.append(f1)
        precision_list.append(prec)
        recall_list.append(rec)

        n_h = (y_test == 0).sum()
        n_f = (y_test == 1).sum()
        log.info(f"    Fold {fold_idx+1}: test_healthy={test_healthy}  "
                 f"healthy={n_h:,}/fault={n_f:,}  "
                 f"AUC={auc:.4f}  Acc={acc:.4f}  F1={f1:.4f}  "
                 f"Prec={prec:.4f}  Rec={rec:.4f}")

    log.info(f"\n  [Autoencoder 이상탐지 결과]")
    log.info(f"    Mean AUC:       {np.mean(auc_list):.4f} ± {np.std(auc_list):.4f}")
    log.info(f"    Mean Acc:       {np.mean(acc_list):.4f} ± {np.std(acc_list):.4f}")
    log.info(f"    Mean F1:        {np.mean(f1_list):.4f} ± {np.std(f1_list):.4f}")
    log.info(f"    Mean Precision: {np.mean(precision_list):.4f} ± {np.std(precision_list):.4f}")
    log.info(f"    Mean Recall:    {np.mean(recall_list):.4f} ± {np.std(recall_list):.4f}")

    # 이전 XGBoost 결과와 비교
    log.info(f"\n  [XGBoost 분류 대비 비교]")
    log.info(f"    XGBoost AUC:     0.5642 (StratifiedGroupKFold)")
    log.info(f"    Autoencoder AUC: {np.mean(auc_list):.4f}")
    log.info(f"    차이:            {np.mean(auc_list) - 0.5642:+.4f}")

    # 시각화: 복원 오차 분포
    plot_reconstruction_error(df_exp, features, model, scaler)

    return {
        "auc": float(np.mean(auc_list)),
        "acc": float(np.mean(acc_list)),
        "f1": float(np.mean(f1_list)),
        "precision": float(np.mean(precision_list)),
        "recall": float(np.mean(recall_list)),
        "fold_auc": [float(x) for x in auc_list],
        "fold_acc": [float(x) for x in acc_list],
        "xgb_auc_comparison": 0.5642,
    }


def plot_reconstruction_error(df, features, model, scaler):
    """정상 vs 이상의 복원 오차 분포 시각화"""
    X_all = scaler.transform(df[features].values.astype(np.float32))
    recon = model.predict(X_all, verbose=0)
    mse = np.mean((X_all - recon) ** 2, axis=1)

    healthy_mask = df['fault_label'] == 'healthy'

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(mse[healthy_mask], bins=100, alpha=0.6, label='Healthy', color='#4CAF50', density=True)
    ax.hist(mse[~healthy_mask], bins=100, alpha=0.6, label='Fault', color='#f44336', density=True)
    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title('3-Feature Autoencoder: Reconstruction Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "reconstruction_error_dist.png"), dpi=150)
    plt.close()
    log.info(f"  [저장] reconstruction_error_dist.png")


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
def main():
    start = datetime.datetime.now()
    section(f"3피처 Autoencoder 이상탐지 — {start.strftime('%Y-%m-%d %H:%M')}")

    # 1. 데이터 로드
    section("Step 1. Paderborn 3피처 추출")
    df = load_paderborn_3features()
    log.info(f"  전체: {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    log.info(f"  라벨 분포:\n{df['fault_label'].value_counts().to_string()}")

    # 2. Autoencoder 실험
    results = experiment_autoencoder_anomaly(df)

    # 3. 결과 저장
    section("결과 저장")
    json_path = os.path.join(OUT_DIR, "experiment_results.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"  [저장] {json_path}")

    elapsed = (datetime.datetime.now() - start).total_seconds() / 60
    log.info(f"\n  총 소요시간: {elapsed:.1f}분")


if __name__ == "__main__":
    main()
