"""
예지보전 딥러닝 학습 파이프라인
AI-PASS S-Traffic | 담당: 하재영

실행 전 조건:
  - predictive_maintenance_pipeline.py 실행 완료
  - FEMTO / KAIST / XJTU 데이터 정상 로드 확인

학습 모델:
  DL-1. LSTM-B     → RUL 회귀 (window 단위 RUL 재계산)
  DL-2. CNN-1D     → 고장모드 분류 (raw 진동 신호 기반)
  DL-3. Autoencoder v2 → 이상 탐지 (개선된 구조)

출력:
  D:/project/예지보전_v2/dl/
    ├── lstm_rul_best.keras        ← LSTM-B 최적 가중치
    ├── cnn_fault_best.keras       ← CNN-1D 최적 가중치
    ├── autoencoder_v2.keras       ← Autoencoder v2
    ├── scaler_lstm.pkl            ← LSTM 입력 스케일러
    ├── scaler_cnn.pkl             ← CNN 입력 스케일러
    ├── scaler_ae.pkl              ← AE 입력 스케일러
    ├── training_log.txt           ← 전체 학습 로그
    └── results_summary.json       ← 최종 결과 요약
"""

import os
import sys
import json
import pickle
import logging
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm.keras import TqdmCallback

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, classification_report
import glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════
# 0. 경로 설정
# ══════════════════════════════════════════════
BASE      = r"D:\project\데이터셋"
FEMTO_DIR = os.path.join(BASE, "10. FEMTO Bearing", "FEMTOBearingDataSet", "Full_Test_Set")
KAIST_DIR = os.path.join(BASE, "Vibration_Bearing_RuntoFailure")
XJTU_DIR  = os.path.join(BASE, "XJTU-SY_Bearing_Datasets", "Data", "XJTU-SY_Bearing_Datasets")
OUT_DIR   = r"D:\project\예지보전_v2\dl"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# 1. 로거 설정 — 파일 + 콘솔 동시 출력
# ══════════════════════════════════════════════
log_path = os.path.join(OUT_DIR, "training_log.txt")
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(message)s",
    datefmt  = "%Y-%m-%d %H:%M:%S",
    handlers = [
        logging.FileHandler(log_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger()


def section(title: str):
    log.info("=" * 60)
    log.info(f"  {title}")
    log.info("=" * 60)


# ══════════════════════════════════════════════
# 2. 공통 상수
# ══════════════════════════════════════════════
WINDOW_SIZE  = 256
STEP         = 128
SEQ_LEN      = 20    # LSTM 시퀀스 길이 (window 개수)
XJTU_LABEL_MAP = {
    "bearing1": "outer_race_fault",
    "bearing2": "inner_race_fault",
    "bearing3": "ball_fault",
}


# ══════════════════════════════════════════════
# 3. 데이터 로드 (기존 파이프라인과 동일)
# ══════════════════════════════════════════════
def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def sliding_window(signal, window_size=WINDOW_SIZE, step=STEP):
    return [signal[i:i+window_size]
            for i in range(0, len(signal) - window_size, step)]


def load_femto(femto_dir: str) -> pd.DataFrame:
    records = []
    bearing_dirs = sorted(glob.glob(os.path.join(femto_dir, "Bearing*")))
    for b_dir in tqdm(bearing_dirs, desc="[FEMTO] bearing 로드", unit="bearing"):
        bearing_id = os.path.basename(b_dir)
        csv_files  = sorted(glob.glob(os.path.join(b_dir, "*.csv")))
        if not csv_files:
            continue

        rms_series  = []
        raw_windows = []
        for fp in tqdm(csv_files, desc=f"  {bearing_id}", unit="file", leave=False):
            try:
                with open(fp) as f:
                    first = f.readline()
                sep    = ';' if ';' in first else ','
                df     = pd.read_csv(fp, header=None, sep=sep)
                n_cols = df.shape[1]
                col    = 4 if n_cols >= 6 else (n_cols - 1)
                signal = df.iloc[:, col].values.astype(float)
                for w in sliding_window(signal):
                    rms_series.append(compute_rms(w))
                    raw_windows.append(w)
            except Exception:
                continue

        if not rms_series:
            continue

        total = len(rms_series)
        for idx, (rms, raw) in enumerate(zip(rms_series, raw_windows)):
            records.append({
                "bearing_id"   : bearing_id,
                "window_idx"   : idx,
                "vibration_rms": rms,
                "RUL"          : total - idx,
                "RUL_norm"     : (total - idx) / total,
                "raw"          : raw,
            })

    df = pd.DataFrame(records)
    log.info(f"  [FEMTO] {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    return df


def load_kaist(kaist_dir: str) -> pd.DataFrame:
    records  = []
    csv_list = glob.glob(os.path.join(kaist_dir, "**", "*.csv"), recursive=True)
    for fp in tqdm(csv_list, desc="[KAIST] 파일 로드", unit="file"):
        try:
            df = pd.read_csv(fp, header=None, nrows=100000)
            signal  = df.iloc[:, 0].values.astype(float)
            temp    = df.iloc[:, 2].values.astype(float)
            current = df.iloc[:, 3].values.astype(float)
            for i, w in enumerate(sliding_window(signal)):
                s = i * STEP
                e = s + WINDOW_SIZE
                records.append({
                    "vibration_rms": compute_rms(w),
                    "temperature"  : float(np.mean(temp[s:e])),
                    "motor_current": float(np.mean(current[s:e])),
                })
        except Exception:
            continue
    df = pd.DataFrame(records)
    log.info(f"  [KAIST] {len(df):,} rows")
    return df


def load_xjtu(xjtu_dir: str) -> pd.DataFrame:
    records  = []
    csv_list = glob.glob(os.path.join(xjtu_dir, "**", "*.csv"), recursive=True)
    for fp in tqdm(csv_list, desc="[XJTU] 파일 로드", unit="file"):
        fp_norm = fp.lower().replace("\\", "/")
        label   = "unknown"
        for key, val in XJTU_LABEL_MAP.items():
            if f"/{key}_" in fp_norm:
                label = val
                break
        if label == "unknown":
            continue
        try:
            df  = pd.read_csv(fp, nrows=100000)
            col = 'Horizontal' if 'Horizontal' in df.columns else df.columns[0]
            signal = df[col].values.astype(float)
            for w in sliding_window(signal):
                records.append({
                    "vibration_rms": compute_rms(w),
                    "fault_label"  : label,
                    "raw"          : w,
                })
        except Exception:
            continue
    df = pd.DataFrame(records)
    log.info(f"  [XJTU] {len(df):,} rows  /  분포: {df['fault_label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════
# 4. 피처 엔지니어링
# ══════════════════════════════════════════════
def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    rms = df['vibration_rms']
    df['short_trend']   = (rms - rms.shift(1)) / rms.shift(1).replace(0, np.nan)
    roll_mean           = rms.rolling(window=window, min_periods=1).mean()
    df['rolling_trend'] = (rms - roll_mean) / roll_mean.replace(0, np.nan)
    if 'temperature' in df.columns:
        exp_temp          = df['temperature'].rolling(window=window, min_periods=1).mean().shift(1)
        df['temp_residual'] = df['temperature'] - exp_temp
    else:
        df['temp_residual'] = 0.0
    return df.dropna(subset=['short_trend', 'rolling_trend'])


# ══════════════════════════════════════════════
# 5. DL-1 — LSTM-B RUL 회귀
#    window 단위 RUL 재계산 → 시계열 패턴 학습 가능
# ══════════════════════════════════════════════
def make_lstm_sequences(df: pd.DataFrame,
                        features: list,
                        rul_col: str = 'RUL_norm',
                        seq_len: int = SEQ_LEN) -> tuple:
    """
    bearing 내에서 seq_len 연속 window → 다음 RUL 예측
    각 bearing을 독립적으로 처리 → bearing 간 시퀀스 혼입 방지
    """
    X_seqs, y_seqs, groups = [], [], []
    bearing_ids = df['bearing_id'].unique()
    for bid in tqdm(bearing_ids, desc="  시퀀스 생성", unit="bearing"):
        grp  = df[df['bearing_id'] == bid].sort_values('window_idx').reset_index(drop=True)
        feat = grp[features].values
        rul  = grp[rul_col].values
        for i in range(seq_len, len(grp)):
            X_seqs.append(feat[i-seq_len:i])
            y_seqs.append(rul[i])
            groups.append(bid)
    return np.array(X_seqs), np.array(y_seqs), np.array(groups)


def train_lstm(df_rul: pd.DataFrame, has_kaist: bool, has_temp: bool) -> dict:
    section("DL-1: LSTM-B RUL 회귀")

    features = ['vibration_rms', 'short_trend', 'rolling_trend']
    if has_kaist:
        features.append('motor_current')
    if has_temp:
        features.append('temp_residual')
    log.info(f"  피처: {features}")

    df_rul = df_rul.dropna(subset=features + ['RUL_norm', 'bearing_id', 'window_idx'])

    # 피처 스케일링
    scaler = StandardScaler()
    df_rul[features] = scaler.fit_transform(df_rul[features])
    pickle.dump(scaler, open(os.path.join(OUT_DIR, "scaler_lstm.pkl"), 'wb'))

    X, y, groups = make_lstm_sequences(df_rul, features)
    log.info(f"  시퀀스 수: {len(X):,}  /  shape: {X.shape}")

    # GroupShuffleSplit — bearing 단위 분리
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]
    log.info(f"  train: {len(X_train):,}  test: {len(X_test):,}")
    log.info(f"  train bearings: {sorted(set(groups[tr_idx]))}")
    log.info(f"  test  bearings: {sorted(set(groups[te_idx]))}")

    # 모델 구조
    n_feat = X.shape[2]
    inp    = keras.Input(shape=(SEQ_LEN, n_feat))
    x      = layers.LSTM(128, return_sequences=True)(inp)
    x      = layers.Dropout(0.3)(x)
    x      = layers.LSTM(64)(x)
    x      = layers.Dropout(0.3)(x)
    x      = layers.Dense(32, activation='relu')(x)
    out    = layers.Dense(1, activation='sigmoid')(x)   # 0~1 출력 (RUL_norm)
    model  = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse',
                  metrics=['mae'])
    model.summary(print_fn=lambda s: log.info("    " + s))

    ckpt_path = os.path.join(OUT_DIR, "lstm_rul_best.keras")
    cb_list = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss',
                                   save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=7, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "lstm_history.csv")),
        TqdmCallback(verbose=1, desc="[LSTM-B] 학습"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = 200,
        batch_size      = 512,
        callbacks       = cb_list,
        verbose         = 0,   # TqdmCallback이 대신 출력
    )

    # 평가
    y_pred = model.predict(X_test, verbose=0).flatten()
    r2     = r2_score(y_test, y_pred)
    log.info(f"\n  [LSTM-B] R² = {r2:.4f}")

    # 학습 곡선 저장
    _plot_history(history, "LSTM-B RUL", os.path.join(OUT_DIR, "lstm_history.png"))

    return {"model": model, "r2": r2, "features": features}


# ══════════════════════════════════════════════
# 6. DL-2 — CNN-1D 고장모드 분류
#    raw 진동 신호(256 샘플) 직접 학습
# ══════════════════════════════════════════════
def train_cnn(df_xjtu: pd.DataFrame) -> dict:
    section("DL-2: CNN-1D 고장모드 분류")

    # 클래스 불균형 대응 — 클래스별 최소 샘플 수로 다운샘플링
    min_count = df_xjtu['fault_label'].value_counts().min()
    df_bal    = df_xjtu.groupby('fault_label').apply(
        lambda g: g.sample(min_count, random_state=42)
    ).reset_index(drop=True)
    log.info(f"  다운샘플링 후 분포: {df_bal['fault_label'].value_counts().to_dict()}")

    # raw 신호 배열 (256, 1)
    X = np.stack(df_bal['raw'].values).astype(np.float32)[..., np.newaxis]

    le = LabelEncoder()
    y  = le.fit_transform(df_bal['fault_label'].values)
    n_classes = len(le.classes_)
    log.info(f"  클래스: {le.classes_.tolist()}")

    # 스케일링 (채널 단위)
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, WINDOW_SIZE)).reshape(X.shape)
    pickle.dump(scaler, open(os.path.join(OUT_DIR, "scaler_cnn.pkl"), 'wb'))
    pickle.dump(le,     open(os.path.join(OUT_DIR, "label_encoder_cnn.pkl"), 'wb'))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"  train: {len(X_train):,}  test: {len(X_test):,}")

    # 모델 구조
    inp = keras.Input(shape=(WINDOW_SIZE, 1))
    x   = layers.Conv1D(64,  kernel_size=7,  activation='relu', padding='same')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(pool_size=2)(x)
    x   = layers.Conv1D(128, kernel_size=5,  activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(pool_size=2)(x)
    x   = layers.Conv1D(256, kernel_size=3,  activation='relu', padding='same')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary(print_fn=lambda s: log.info("    " + s))

    ckpt_path = os.path.join(OUT_DIR, "cnn_fault_best.keras")
    cb_list = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy',
                                   save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=7, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "cnn_history.csv")),
        TqdmCallback(verbose=1, desc="[CNN-1D] 학습"),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = 100,
        batch_size      = 256,
        callbacks       = cb_list,
        verbose         = 0,
    )

    # 평가
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    report = classification_report(y_test, y_pred,
                                    target_names=le.classes_)
    log.info(f"\n[CNN-1D] Classification Report\n{report}")

    _plot_history(history, "CNN-1D Fault", os.path.join(OUT_DIR, "cnn_history.png"),
                  metric='accuracy')

    return {"model": model, "report": report, "classes": le.classes_.tolist()}


# ══════════════════════════════════════════════
# 7. DL-3 — Autoencoder v2 이상 탐지
#    구조 개선: Deeper + BatchNorm
# ══════════════════════════════════════════════
def train_autoencoder_v2(df_normal: pd.DataFrame,
                          has_kaist: bool,
                          has_temp: bool) -> dict:
    section("DL-3: Autoencoder v2 이상 탐지")

    ae_feat = ['vibration_rms', 'short_trend', 'rolling_trend']
    if has_kaist and 'motor_current' in df_normal.columns:
        ae_feat.append('motor_current')
    if has_temp and 'temp_residual' in df_normal.columns:
        ae_feat.append('temp_residual')
    log.info(f"  피처: {ae_feat}")

    df_normal = df_normal.dropna(subset=ae_feat)
    X = df_normal[ae_feat].values.astype(np.float32)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pickle.dump(scaler, open(os.path.join(OUT_DIR, "scaler_ae.pkl"), 'wb'))

    n_feat = X_scaled.shape[1]
    X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=42)
    log.info(f"  train: {len(X_train):,}  val: {len(X_val):,}")

    # 개선된 Autoencoder — BatchNorm + Deeper
    inp  = keras.Input(shape=(n_feat,))
    # Encoder
    e    = layers.Dense(64,  activation='relu')(inp)
    e    = layers.BatchNormalization()(e)
    e    = layers.Dense(32,  activation='relu')(e)
    e    = layers.BatchNormalization()(e)
    code = layers.Dense(16,  activation='relu')(e)   # bottleneck
    # Decoder
    d    = layers.Dense(32,  activation='relu')(code)
    d    = layers.BatchNormalization()(d)
    d    = layers.Dense(64,  activation='relu')(d)
    d    = layers.BatchNormalization()(d)
    out  = layers.Dense(n_feat, activation='linear')(d)
    ae   = keras.Model(inp, out)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse')
    ae.summary(print_fn=lambda s: log.info("    " + s))

    ckpt_path = os.path.join(OUT_DIR, "autoencoder_v2.keras")
    cb_list = [
        callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss',
                                   save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=10, min_lr=1e-6, verbose=1),
        callbacks.CSVLogger(os.path.join(OUT_DIR, "ae_history.csv")),
        TqdmCallback(verbose=1, desc="[AE v2] 학습"),
    ]

    history = ae.fit(
        X_train, X_train,
        validation_data = (X_val, X_val),
        epochs          = 200,
        batch_size      = 512,
        callbacks       = cb_list,
        verbose         = 0,
    )

    # threshold 결정 (FPR ≤ 5%)
    X_pred  = ae.predict(X_scaled, verbose=0)
    errors  = np.mean((X_scaled - X_pred) ** 2, axis=1)
    mean, std = errors.mean(), errors.std()
    candidates = {
        "percentile_95": float(np.percentile(errors, 95)),
        "mean+2std"    : float(mean + 2 * std),
        "mean+3std"    : float(mean + 3 * std),
    }

    log.info("\n  [AE v2] Threshold 후보:")
    selected = None
    for name, val in candidates.items():
        fpr  = float(np.mean(errors > val))
        mark = "✅" if fpr <= 0.05 else "  "
        log.info(f"    {mark} {name:<18} threshold={val:.6f}  FPR={fpr:.4f}")
        if fpr <= 0.05 and selected is None:
            selected = (name, val)

    if selected is None:
        selected = ("mean+3std", candidates["mean+3std"])
    log.info(f"  → 선택: {selected[0]}  threshold={selected[1]:.6f}")

    _plot_history(history, "Autoencoder v2", os.path.join(OUT_DIR, "ae_history.png"))

    return {"model": ae, "threshold": selected[1], "features": ae_feat}


# ══════════════════════════════════════════════
# 8. 유틸 — 학습 곡선 저장
# ══════════════════════════════════════════════
def _plot_history(history, title: str, save_path: str, metric: str = 'loss'):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, m in zip(axes, [metric, 'loss' if metric != 'loss' else 'mae']):
        if m in history.history:
            ax.plot(history.history[m],       label=f'train {m}')
            val_key = f'val_{m}'
            if val_key in history.history:
                ax.plot(history.history[val_key], label=f'val {m}')
            ax.set_title(f'{title} — {m}')
            ax.legend()
            ax.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    log.info(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 9. 메인
# ══════════════════════════════════════════════
def main():
    start_time = datetime.datetime.now()
    section(f"AI-PASS 예지보전 딥러닝 학습 시작 — {start_time.strftime('%Y-%m-%d %H:%M')}")

    results = {}

    # ── 데이터 로드
    section("Step 1. 데이터 로드")
    df_femto = load_femto(FEMTO_DIR)
    df_kaist = load_kaist(KAIST_DIR)
    df_xjtu  = load_xjtu(XJTU_DIR)

    has_kaist = not df_kaist.empty
    has_temp  = has_kaist

    # ── 피처 엔지니어링
    section("Step 2. 피처 엔지니어링")
    if not df_femto.empty:
        if has_kaist:
            df_femto['motor_current'] = df_kaist['motor_current'].mean()
            df_femto['temperature']   = df_kaist['temperature'].mean()
        df_rul = engineer_features(df_femto)
        log.info(f"  RUL 데이터: {len(df_rul):,} rows")
    else:
        df_rul = pd.DataFrame()
        log.warning("  ⚠️  FEMTO 없음 — LSTM 학습 불가")

    if not df_xjtu.empty:
        df_xjtu = df_xjtu.reset_index(drop=True)
        df_xjtu['temperature'] = 0.0
        label_series = df_xjtu['fault_label'].copy()
        raw_series   = df_xjtu['raw'].copy()
        df_clf = engineer_features(df_xjtu)
        df_clf['fault_label'] = label_series.iloc[df_clf.index].values
        df_clf['raw']         = raw_series.iloc[df_clf.index].values
        if has_kaist:
            df_clf['motor_current'] = df_kaist['motor_current'].mean()
        log.info(f"  분류 데이터: {len(df_clf):,} rows")
    else:
        df_clf = pd.DataFrame()
        log.warning("  ⚠️  XJTU 없음 — CNN 학습 불가")

    # ── DL-1. LSTM-B
    if not df_rul.empty and len(df_rul) > SEQ_LEN * 5:
        try:
            result_lstm = train_lstm(df_rul, has_kaist, has_temp)
            results['lstm'] = {"r2": float(result_lstm['r2'])}
        except Exception as e:
            log.error(f"  [LSTM] 오류: {e}")
    else:
        log.warning("  ⚠️  RUL 데이터 부족 — LSTM 생략")

    # ── DL-2. CNN-1D
    if not df_clf.empty and 'raw' in df_clf.columns:
        try:
            result_cnn = train_cnn(df_clf)
            results['cnn'] = {"report_summary": result_cnn['report'][:300]}
        except Exception as e:
            log.error(f"  [CNN] 오류: {e}")
    else:
        log.warning("  ⚠️  분류 데이터 없음 — CNN 생략")

    # ── DL-3. Autoencoder v2
    if not df_rul.empty:
        df_normal = df_rul[df_rul['RUL'] > 30].copy()
        if len(df_normal) > 100:
            try:
                result_ae = train_autoencoder_v2(df_normal, has_kaist, has_temp)
                results['autoencoder_v2'] = {"threshold": float(result_ae['threshold'])}
            except Exception as e:
                log.error(f"  [AE v2] 오류: {e}")

    # ── 결과 저장
    elapsed = (datetime.datetime.now() - start_time).total_seconds() / 3600
    results['elapsed_hours'] = round(elapsed, 2)
    results['finished_at']   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    summary_path = os.path.join(OUT_DIR, "results_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    section(f"학습 완료 — 총 소요시간: {elapsed:.2f}시간")
    log.info(f"  결과 저장: {OUT_DIR}")
    log.info(f"  요약 파일: {summary_path}")


if __name__ == "__main__":
    main()