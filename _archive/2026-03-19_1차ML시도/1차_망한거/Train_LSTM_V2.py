"""
AI-Pass 예지보전 LSTM 모델 학습 v2.0
======================================
변경사항 (v1 → v2):
  - train/test 분리를 파일 내 80/20이 아닌
    베어링 단위로 분리 (combined_test.csv 사용)
  - test: 학습에 한 번도 안 쓴 베어링 전체 수명
    → LOW → CRITICAL 전환 패턴 검증 가능

출력:
  D:/project/예지보전/output/
    ├── lstm_model.keras
    ├── training_history.xlsx
    └── evaluation_report.xlsx

실행: python train_lstm_v2.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import time
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path(r"D:\project\예지보전\output_v3")
TRAIN_CSV    = OUTPUT_DIR / "combined_train.csv"
TEST_CSV     = OUTPUT_DIR / "combined_test.csv"
MODEL_PATH   = OUTPUT_DIR / "lstm_model.keras"
HISTORY_PATH = OUTPUT_DIR / "training_history.xlsx"
EVAL_PATH    = OUTPUT_DIR / "evaluation_report.xlsx"

SEQ_LEN       = 10
BATCH_SIZE    = 64
EPOCHS        = 100
LEARNING_RATE = 0.001
PATIENCE      = 15

FEATURE_COLS = [
    "vibration_rms", "temperature", "temp_residual",
    "motor_current", "operating_hours", "ambient_temp",
    "wind_speed", "humidity", "season",
]

# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def rul_to_risk(rul: float) -> str:
    if rul >= 31: return "LOW"
    if rul >= 16: return "MEDIUM"
    if rul >=  3: return "HIGH"
    return "CRITICAL"


def make_sequences(
    df          : pd.DataFrame,
    seq_len     : int,
    feature_cols: list,
    target_col  : str = "rul_days",
) -> tuple[np.ndarray, np.ndarray]:
    """
    시계열 순서 유지 시퀀스 생성
    source + file_name 단위로 그룹핑 후 슬라이딩
    """
    X_list, y_list = [], []

    for (src, fname), group in df.groupby(
        ["source", "file_name"], sort=False
    ):
        group = group.reset_index(drop=True)
        feat  = group[feature_cols].values.astype(np.float32)
        tgt   = group[target_col].values.astype(np.float32)

        if len(group) < seq_len:
            continue

        for i in range(len(group) - seq_len + 1):
            X_list.append(feat[i : i + seq_len])
            y_list.append(tgt[i + seq_len - 1])

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )


# ══════════════════════════════════════════════════════════════
# STEP 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════

def load_and_prepare() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    print("\n" + "=" * 60)
    print("STEP 1. 데이터 로딩")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)
    test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].fillna(0)

    print(f"combined_train.csv : {len(train_df):,}행")
    print_dist(train_df, "train 분포")

    print(f"\ncombined_test.csv  : {len(test_df):,}행")
    print_dist(test_df, "test 분포 (원본)")

    # 시퀀스 생성
    print(f"\nSEQ_LEN={SEQ_LEN} 시퀀스 생성 중...")
    X_train, y_train = make_sequences(train_df, SEQ_LEN, FEATURE_COLS)
    X_test,  y_test  = make_sequences(test_df,  SEQ_LEN, FEATURE_COLS)

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
    print(f"train RUL: {y_train.min():.2f}~{y_train.max():.2f}일")
    print(f"test  RUL: {y_test.min():.2f}~{y_test.max():.2f}일")

    return X_train, y_train, X_test, y_test


def print_dist(df: pd.DataFrame, label: str):
    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"  {label}:")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"    {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 2. 모델 정의
# ══════════════════════════════════════════════════════════════

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    print("\n" + "=" * 60)
    print("STEP 2. LSTM 모델 정의")
    print("=" * 60)

    model = Sequential([
        LSTM(128, input_shape=(seq_len, n_features),
             return_sequences=True, name="lstm_1"),
        BatchNormalization(),
        Dropout(0.2),

        LSTM(64, return_sequences=False, name="lstm_2"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation="relu", name="dense_1"),
        Dropout(0.1),
        Dense(1, activation="linear", name="output"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",
        metrics=["mae"],
    )

    model.summary()
    return model


# ══════════════════════════════════════════════════════════════
# STEP 3. 학습
# ══════════════════════════════════════════════════════════════

def train_model(
    model  : tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test : np.ndarray,
    y_test : np.ndarray,
) -> tf.keras.callbacks.History:

    print("\n" + "=" * 60)
    print("STEP 3. 모델 학습")
    print(f"  train: {X_train.shape[0]:,}개 / test: {X_test.shape[0]:,}개")
    print("=" * 60)

    callbacks = [
        EarlyStopping(
            monitor="val_mae", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            str(MODEL_PATH), monitor="val_mae",
            save_best_only=True, verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,  # 시계열 순서 유지
        verbose=1,
    )

    return history


# ══════════════════════════════════════════════════════════════
# STEP 4. 평가
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model : tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[pd.DataFrame, dict]:

    print("\n" + "=" * 60)
    print("STEP 4. 모델 평가")
    print("=" * 60)

    y_pred = np.maximum(
        model.predict(X_test, verbose=0).flatten(), 0
    )

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    actual_risk = [rul_to_risk(v) for v in y_test]
    pred_risk   = [rul_to_risk(v) for v in y_pred]
    risk_acc    = sum(
        a == p for a, p in zip(actual_risk, pred_risk)
    ) / len(y_test)

    print(f"Test MAE      : {mae:.4f}일")
    print(f"Test RMSE     : {rmse:.4f}일")
    print(f"Test R²       : {r2:.4f}")
    print(f"등급 정확도   : {risk_acc*100:.2f}%")

    print(f"\n등급별 정확도:")
    grade_acc = {}
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        idx     = [i for i, r in enumerate(actual_risk) if r == g]
        if not idx:
            continue
        correct = sum(actual_risk[i] == pred_risk[i] for i in idx)
        acc     = correct / len(idx)
        grade_acc[g] = acc
        print(f"  {g:10s}: {correct:5,}/{len(idx):5,} ({acc*100:.2f}%)")

    # LOW → CRITICAL 전환 감지 분석
    print(f"\n[핵심] LOW→CRITICAL 전환 감지 분석:")
    transitions = []
    for i in range(1, len(actual_risk)):
        if actual_risk[i-1] == "LOW" and actual_risk[i] == "CRITICAL":
            pred_before = pred_risk[i-1]
            pred_after  = pred_risk[i]
            transitions.append({
                "index"       : i,
                "actual_전"   : actual_risk[i-1],
                "actual_후"   : actual_risk[i],
                "pred_전"     : pred_before,
                "pred_후"     : pred_after,
                "rul_actual"  : round(float(y_test[i]), 2),
                "rul_pred"    : round(float(y_pred[i]), 2),
                "감지성공"    : pred_after in ["HIGH", "CRITICAL"],
            })

    if transitions:
        success = sum(t["감지성공"] for t in transitions)
        print(f"  전환 구간 수    : {len(transitions)}")
        print(f"  감지 성공       : {success}/{len(transitions)}"
              f" ({success/len(transitions)*100:.1f}%)")
    else:
        print("  전환 구간 없음 (test 데이터 확인 필요)")

    detail_df = pd.DataFrame({
        "actual_rul"  : np.round(y_test, 4),
        "pred_rul"    : np.round(y_pred, 4),
        "actual_risk" : actual_risk,
        "pred_risk"   : pred_risk,
        "error_days"  : np.round(np.abs(y_test - y_pred), 4),
        "correct"     : [a == p for a, p in zip(actual_risk, pred_risk)],
    })

    metrics = {
        "mae"           : round(mae, 4),
        "rmse"          : round(rmse, 4),
        "r2"            : round(r2, 4),
        "risk_acc"      : round(risk_acc, 4),
        "low_acc"       : round(grade_acc.get("LOW",      0), 4),
        "medium_acc"    : round(grade_acc.get("MEDIUM",   0), 4),
        "high_acc"      : round(grade_acc.get("HIGH",     0), 4),
        "critical_acc"  : round(grade_acc.get("CRITICAL", 0), 4),
    }

    return detail_df, metrics


# ══════════════════════════════════════════════════════════════
# STEP 5. 저장
# ══════════════════════════════════════════════════════════════

def save_results(
    history  : tf.keras.callbacks.History,
    detail_df: pd.DataFrame,
    metrics  : dict,
):
    print("\n" + "=" * 60)
    print("STEP 5. 결과 저장")
    print("=" * 60)

    # training_history.xlsx
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df)+1))
    with pd.ExcelWriter(HISTORY_PATH, engine="openpyxl") as writer:
        hist_df.to_excel(writer, sheet_name="학습이력", index=False)
    print(f"  training_history.xlsx: {len(hist_df)}epoch")

    # evaluation_report.xlsx
    summary_df = pd.DataFrame([{
        "MAE(일)"      : metrics["mae"],
        "RMSE(일)"     : metrics["rmse"],
        "R²"           : metrics["r2"],
        "전체등급정확도": metrics["risk_acc"],
        "LOW정확도"    : metrics["low_acc"],
        "MEDIUM정확도" : metrics["medium_acc"],
        "HIGH정확도"   : metrics["high_acc"],
        "CRITICAL정확도": metrics["critical_acc"],
        "SEQ_LEN"      : SEQ_LEN,
        "BATCH_SIZE"   : BATCH_SIZE,
        "모델구조"     : "LSTM(128→64)→Dense(32→1)",
        "Loss"         : "Huber",
        "검증방식"     : "베어링 단위 분리 (Bearing1_3, 1_5)",
    }])

    with pd.ExcelWriter(EVAL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="평가요약",   index=False)
        detail_df.to_excel( writer, sheet_name="샘플별결과", index=False)

    print(f"  evaluation_report.xlsx: 저장 완료")
    print(f"  lstm_model.keras: {MODEL_PATH}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("TF :", tf.__version__)

    X_train, y_train, X_test, y_test = load_and_prepare()
    model   = build_model(SEQ_LEN, len(FEATURE_COLS))
    history = train_model(model, X_train, y_train, X_test, y_test)
    detail_df, metrics = evaluate_model(model, X_test, y_test)
    save_results(history, detail_df, metrics)

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("완료.")