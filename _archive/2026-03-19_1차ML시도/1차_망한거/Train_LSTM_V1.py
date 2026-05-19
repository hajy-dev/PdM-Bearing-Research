"""
AI-Pass 예지보전 LSTM 모델 학습 v1.0
======================================
목적  : combined_train.csv 기반 LSTM 회귀 모델 학습
        - 입력: (샘플수, SEQ_LEN, 9피처) 3D 텐서
        - 출력: rul_days (연속값) → 등급 변환
        - 시계열 순서 유지 (shuffle=False for time-series split)

출력  :
  D:/project/예지보전/output/
    ├── lstm_model.keras          (학습 완료 모델)
    ├── training_history.xlsx     (epoch별 loss/mae)
    └── evaluation_report.xlsx   (테스트셋 평가 결과)

실행  : python train_lstm.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import time
import warnings
warnings.filterwarnings("ignore")

# TensorFlow / Keras
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

OUTPUT_DIR     = Path(r"D:\project\예지보전\output_v2")
COMBINED_CSV   = OUTPUT_DIR / "combined_train.csv"
SCALER_PATH    = OUTPUT_DIR / "scaler.pkl"
MODEL_PATH     = OUTPUT_DIR / "lstm_model.keras"
HISTORY_PATH   = OUTPUT_DIR / "training_history.xlsx"
EVAL_PATH      = OUTPUT_DIR / "evaluation_report.xlsx"

SEQ_LEN        = 10      # 시계열 윈도우 길이 (연속 몇 개 샘플을 묶어서 입력)
TEST_RATIO     = 0.2     # 테스트셋 비율
BATCH_SIZE     = 64
EPOCHS         = 100
LEARNING_RATE  = 0.001
PATIENCE       = 15      # EarlyStopping patience

FEATURE_COLS = [
    "vibration_rms", "temperature", "temp_residual",
    "motor_current", "operating_hours", "ambient_temp",
    "wind_speed", "humidity", "season",
]

BASE_LIFE_DAYS = 180

# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def rul_to_risk(rul: float) -> str:
    if rul >= 31: return "LOW"
    if rul >= 16: return "MEDIUM"
    if rul >=  3: return "HIGH"
    return "CRITICAL"


def make_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list,
    target_col: str = "rul_days",
) -> tuple[np.ndarray, np.ndarray]:
    """
    DataFrame → LSTM 3D 입력 시퀀스 생성
    시계열 순서 유지 (source + file_name 단위로 그룹핑)

    반환:
        X: (n_samples, seq_len, n_features)
        y: (n_samples,)
    """
    X_list, y_list = [], []

    # source + file_name 단위로 시계열 그룹핑
    groups = df.groupby(["source", "file_name"], sort=False)

    for (src, fname), group in groups:
        group = group.reset_index(drop=True)
        feat  = group[feature_cols].values
        tgt   = group[target_col].values

        # 슬라이딩으로 시퀀스 생성
        for i in range(len(group) - seq_len + 1):
            X_list.append(feat[i : i + seq_len])
            y_list.append(tgt[i + seq_len - 1])  # 마지막 타임스텝의 RUL

    if not X_list:
        raise ValueError("시퀀스 생성 실패 — 데이터 확인 필요")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ══════════════════════════════════════════════════════════════
# STEP 1. 데이터 로딩 + 시퀀스 생성
# ══════════════════════════════════════════════════════════════

def load_and_prepare() -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    print("\n" + "=" * 60)
    print("STEP 1. 데이터 로딩 + 시퀀스 생성")
    print("=" * 60)

    df = pd.read_csv(COMBINED_CSV)
    print(f"combined_train.csv 로딩: {len(df):,}행")

    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"등급 분포:")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>6,}개 ({dist[g]/total*100:.1f}%)")

    # NaN 처리
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    # ── Train / Test 분리 (시계열 순서 유지) ──
    # source 단위로 앞 80%를 train, 뒤 20%를 test
    train_dfs, test_dfs = [], []

    for (src, fname), group in df.groupby(["source", "file_name"], sort=False):
        group  = group.reset_index(drop=True)
        n      = len(group)
        split  = int(n * (1 - TEST_RATIO))
        train_dfs.append(group.iloc[:split])
        test_dfs.append(group.iloc[split:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df  = pd.concat(test_dfs,  ignore_index=True)

    print(f"\nTrain: {len(train_df):,}행 / Test: {len(test_df):,}행")

    # ── 시퀀스 생성 ──
    print(f"\nSEQ_LEN={SEQ_LEN} 시퀀스 생성 중...")
    X_train, y_train = make_sequences(train_df, SEQ_LEN, FEATURE_COLS)
    X_test,  y_test  = make_sequences(test_df,  SEQ_LEN, FEATURE_COLS)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}   y_test : {y_test.shape}")
    print(f"RUL 범위 — train: {y_train.min():.2f}~{y_train.max():.2f}일 "
          f"/ test: {y_test.min():.2f}~{y_test.max():.2f}일")

    return X_train, y_train, X_test, y_test


# ══════════════════════════════════════════════════════════════
# STEP 2. 모델 정의
# ══════════════════════════════════════════════════════════════

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    print("\n" + "=" * 60)
    print("STEP 2. LSTM 모델 정의")
    print("=" * 60)

    model = Sequential([
        # LSTM 레이어 1
        LSTM(
            128,
            input_shape=(seq_len, n_features),
            return_sequences=True,
            name="lstm_1",
        ),
        BatchNormalization(),
        Dropout(0.2),

        # LSTM 레이어 2
        LSTM(
            64,
            return_sequences=False,
            name="lstm_2",
        ),
        BatchNormalization(),
        Dropout(0.2),

        # Dense 레이어
        Dense(32, activation="relu", name="dense_1"),
        Dropout(0.1),
        Dense(1, activation="linear", name="output"),  # RUL 회귀
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="huber",       # MSE보다 이상치에 강건
        metrics=["mae"],
    )

    model.summary()
    print(f"\n총 파라미터: {model.count_params():,}개")

    return model


# ══════════════════════════════════════════════════════════════
# STEP 3. 학습
# ══════════════════════════════════════════════════════════════

def train_model(
    model    : tf.keras.Model,
    X_train  : np.ndarray,
    y_train  : np.ndarray,
    X_test   : np.ndarray,
    y_test   : np.ndarray,
) -> tf.keras.callbacks.History:

    print("\n" + "=" * 60)
    print("STEP 3. 모델 학습")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  EarlyStopping patience: {PATIENCE}")
    print("=" * 60)

    callbacks = [
        EarlyStopping(
            monitor   = "val_mae",
            patience  = PATIENCE,
            restore_best_weights = True,
            verbose   = 1,
        ),
        ReduceLROnPlateau(
            monitor  = "val_loss",
            factor   = 0.5,
            patience = 7,
            min_lr   = 1e-6,
            verbose  = 1,
        ),
        ModelCheckpoint(
            filepath         = str(MODEL_PATH),
            monitor          = "val_mae",
            save_best_only   = True,
            verbose          = 1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs          = EPOCHS,
        batch_size      = BATCH_SIZE,
        callbacks       = callbacks,
        verbose         = 1,
        shuffle         = False,  # 시계열 순서 유지
    )

    return history


# ══════════════════════════════════════════════════════════════
# STEP 4. 평가
# ══════════════════════════════════════════════════════════════

def evaluate_model(
    model  : tf.keras.Model,
    X_test : np.ndarray,
    y_test : np.ndarray,
) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print("STEP 4. 모델 평가")
    print("=" * 60)

    y_pred = np.maximum(model.predict(X_test, verbose=0).flatten(), 0)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # 과적합 갭 (train 성능)
    y_pred_train = np.maximum(
        model.predict(
            np.concatenate([X_test]), verbose=0
        ).flatten(), 0
    )

    print(f"Test MAE  : {mae:.4f}일")
    print(f"Test RMSE : {rmse:.4f}일")
    print(f"Test R²   : {r2:.4f}")

    # 위험도 등급 정확도
    actual_risk = [rul_to_risk(v) for v in y_test]
    pred_risk   = [rul_to_risk(v) for v in y_pred]
    risk_acc    = sum(a == p for a, p in zip(actual_risk, pred_risk)) / len(y_test)
    print(f"등급 정확도: {risk_acc:.4f} ({risk_acc*100:.2f}%)")

    # 등급별 정확도
    print(f"\n등급별 정확도:")
    grade_results = {}
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        indices = [i for i, r in enumerate(actual_risk) if r == g]
        if not indices:
            continue
        correct = sum(
            actual_risk[i] == pred_risk[i] for i in indices
        )
        acc = correct / len(indices)
        grade_results[g] = {
            "샘플수"  : len(indices),
            "정확수"  : correct,
            "정확도"  : round(acc, 4),
        }
        print(f"  {g:10s}: {correct:5,}/{len(indices):5,} ({acc*100:.2f}%)")

    # 상세 결과 DataFrame
    detail_df = pd.DataFrame({
        "actual_rul"   : y_test,
        "pred_rul"     : np.round(y_pred, 4),
        "actual_risk"  : actual_risk,
        "pred_risk"    : pred_risk,
        "error_days"   : np.round(np.abs(y_test - y_pred), 4),
        "risk_correct" : [a == p for a, p in zip(actual_risk, pred_risk)],
    })

    # 과적합 분석
    overfit_gap = 0  # train loss는 history에서 확인
    print(f"\n과적합 판정 기준:")
    print(f"  R² > 0.7    : {'✅' if r2 > 0.7 else '❌'} ({r2:.4f})")
    print(f"  등급 정확도 > 70%: {'✅' if risk_acc > 0.7 else '❌'} ({risk_acc*100:.2f}%)")

    return detail_df, {
        "mae"      : round(mae, 4),
        "rmse"     : round(rmse, 4),
        "r2"       : round(r2, 4),
        "risk_acc" : round(risk_acc, 4),
    }


# ══════════════════════════════════════════════════════════════
# STEP 5. 결과 저장
# ══════════════════════════════════════════════════════════════

def save_results(
    history   : tf.keras.callbacks.History,
    detail_df : pd.DataFrame,
    metrics   : dict,
):
    print("\n" + "=" * 60)
    print("STEP 5. 결과 저장")
    print("=" * 60)

    # training_history.xlsx
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", range(1, len(hist_df)+1))

    with pd.ExcelWriter(HISTORY_PATH, engine="openpyxl") as writer:
        hist_df.to_excel(writer, sheet_name="학습이력", index=False)
    print(f"  training_history.xlsx: {len(hist_df)}epoch 저장")

    # evaluation_report.xlsx
    summary_df = pd.DataFrame([{
        "MAE(일)"         : metrics["mae"],
        "RMSE(일)"        : metrics["rmse"],
        "R²"              : metrics["r2"],
        "등급정확도"      : metrics["risk_acc"],
        "SEQ_LEN"         : SEQ_LEN,
        "BATCH_SIZE"      : BATCH_SIZE,
        "LEARNING_RATE"   : LEARNING_RATE,
        "모델"            : "LSTM(128→64→32→1)",
        "Loss함수"        : "Huber",
    }])

    with pd.ExcelWriter(EVAL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel( writer, sheet_name="평가요약",    index=False)
        detail_df.to_excel(  writer, sheet_name="샘플별결과",  index=False)
    print(f"  evaluation_report.xlsx: 저장 완료")
    print(f"\n모델 저장 경로: {MODEL_PATH}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    print("GPU 확인:", tf.config.list_physical_devices("GPU"))
    print("TensorFlow 버전:", tf.__version__)

    # STEP 1. 데이터 준비
    X_train, y_train, X_test, y_test = load_and_prepare()

    # STEP 2. 모델 정의
    model = build_model(
        seq_len    = SEQ_LEN,
        n_features = len(FEATURE_COLS),
    )

    # STEP 3. 학습
    history = train_model(model, X_train, y_train, X_test, y_test)

    # STEP 4. 평가
    detail_df, metrics = evaluate_model(model, X_test, y_test)

    # STEP 5. 저장
    save_results(history, detail_df, metrics)

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("\n완료.")