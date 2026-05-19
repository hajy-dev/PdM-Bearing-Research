"""
예지보전 AI 파이프라인
AI-PASS S-Traffic | 담당: 하재영

[데이터셋 구조 가정]
FEMTO  : CSV, 헤더 없음, 컬럼 = [시간, 가속도1, 가속도2]  → 가속도1 사용
KAIST  : CSV, 헤더 없음, 컬럼 = [시간, vibration, motor_current, ...]
XJTU   : CSV, 헤더 있음, 컬럼 = ['Horizontal', 'Vertical'] or ['CH1','CH2']

⚠️  실행 후 구조가 다르면 [수정 필요] 태그가 붙은 줄만 바꾸면 됨
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor
import tensorflow as tf
import keras
from keras import layers
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════
# 0. 경로 설정
# ══════════════════════════════════════════════
BASE      = r"D:\project\데이터셋"
FEMTO_DIR = os.path.join(BASE, "10. FEMTO Bearing", "FEMTOBearingDataSet", "Full_Test_Set")
KAIST_DIR = os.path.join(BASE, "Vibration_Bearing_RuntoFailure")
XJTU_DIR  = os.path.join(BASE, "XJTU-SY_Bearing_Datasets", "Data", "XJTU-SY_Bearing_Datasets")
OUT_DIR   = r"D:\project\예지보전_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# 1. 공통 유틸
# ══════════════════════════════════════════════
WINDOW_SIZE = 256   # 진동 신호 window 크기 (샘플 수)
STEP        = 128   # sliding window 이동 간격
T_FAILURE   = 124   # FEMTO RUL 기준 window index (RMS+온도 그래프 변곡점 확인값)


def compute_rms(signal: np.ndarray) -> float:
    """raw 진동 신호 → RMS 값"""
    return float(np.sqrt(np.mean(signal ** 2)))


def sliding_window(signal: np.ndarray,
                   window_size: int = WINDOW_SIZE,
                   step: int = STEP) -> list:
    """1D 신호 → window 리스트"""
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[i: i + window_size])
    return windows


def assign_rul_grade_norm(rul_ratio: float) -> str:
    """RUL 비율 (0~1) → 등급. RUL_norm 컬럼 전용."""
    if rul_ratio <= 0.05:
        return "CRITICAL"
    elif rul_ratio <= 0.25:
        return "HIGH"
    elif rul_ratio <= 0.50:
        return "MEDIUM"
    else:
        return "LOW"


def assign_rul_grade_clip(rul_val: float) -> str:
    """RUL 클리핑 값 (0~200 window) → 등급. RUL_clip 컬럼 전용."""
    if rul_val <= 10:
        return "CRITICAL"
    elif rul_val <= 50:
        return "HIGH"
    elif rul_val <= 100:
        return "MEDIUM"
    else:
        return "LOW"


# ══════════════════════════════════════════════
# 2. 데이터 로드
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# 2-1. FEMTO → vibration_rms + RUL 생성
#   가정: 폴더 구조 Bearing1_1/, Bearing1_2/, ...
#         각 폴더 안 acc_*.csv, 컬럼 = [시간, acc_h, acc_v]
# ──────────────────────────────────────────────
def load_femto(femto_dir: str) -> pd.DataFrame:
    """
    FEMTO Bearing 데이터셋 로드 (Full_Test_Set 기준)
    폴더 구조: Full_Test_Set/Bearing*_*/*.csv
    컬럼: [시간, acc_h, acc_v] → acc_h(인덱스 1) 사용
    """
    records = []
    # Full_Test_Set 직하위 Bearing* 폴더 탐색
    bearing_dirs = sorted(glob.glob(os.path.join(femto_dir, "Bearing*")))

    if not bearing_dirs:
        print(f"  [FEMTO] Bearing 폴더 없음: {femto_dir}")
        return pd.DataFrame()

    for b_dir in bearing_dirs:
        bearing_id = os.path.basename(b_dir)
        # acc_*.csv 우선, 없으면 *.csv 전체
        csv_files = sorted(glob.glob(os.path.join(b_dir, "acc_*.csv")))
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))

        if not csv_files:
            continue

        rms_series = []
        for fp in csv_files:
            try:
                # 구분자 자동 감지 (FEMTO는 ',' 또는 ';' 혼용)
                with open(fp, 'r') as f:
                    first_line = f.readline()
                sep = ';' if ';' in first_line else ','

                df = pd.read_csv(fp, header=None, sep=sep)
                n_cols = df.shape[1]

                # 컬럼 구조: [시, 분, 초, 샘플번호, acc_h, acc_v]
                # acc_h = 인덱스 4 (6컬럼 표준 FEMTO)
                # 컬럼이 6개 미만인 비정형 파일은 마지막 컬럼 사용
                vib_col = 4 if n_cols >= 6 else (n_cols - 1)
                signal = df.iloc[:, vib_col].values.astype(float)
                for w in sliding_window(signal):
                    rms_series.append(compute_rms(w))
            except Exception as e:
                print(f"  [FEMTO] 읽기 실패: {fp} / {e}")
                continue

        if not rms_series:
            continue

        total_windows = len(rms_series)
        for idx, rms_val in enumerate(rms_series):
            rul_raw = total_windows - idx
            records.append({
                "bearing_id"    : bearing_id,
                "window_idx"    : idx,
                "vibration_rms" : rms_val,
                "RUL"           : rul_raw,
                "RUL_norm"      : rul_raw / total_windows,   # 옵션A: 0~1 정규화
                "RUL_clip"      : min(rul_raw, 200),          # 옵션B: 최대 200 클리핑
                "source"        : "FEMTO",
            })

    df_femto = pd.DataFrame(records)
    if df_femto.empty:
        print(f"  [FEMTO] 로드 실패 — CSV 없음")
    else:
        print(f"  [FEMTO] 로드 완료: {len(df_femto):,} rows / {df_femto['bearing_id'].nunique()} bearings")
    return df_femto


# ──────────────────────────────────────────────
# 2-2. KAIST → vibration_rms + motor_current
#   가정: CSV, 헤더 없음
#         컬럼 = [시간, vibration, motor_current, temperature, ...]
# ──────────────────────────────────────────────
def load_kaist(kaist_dir: str) -> pd.DataFrame:
    """
    Mendeley KAIST 데이터셋 로드
    컬럼 구조 (헤더 없음):
      0: vibration_h  1: vibration_v  2: temperature  3: motor_current
    반환: vibration_rms, motor_current, temperature 컬럼 포함 DataFrame
    """
    csv_files = glob.glob(os.path.join(kaist_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"  [KAIST] 경로 없음: {kaist_dir}")
        return pd.DataFrame()

    records = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, header=None, nrows=100000)

            vib_col     = 0   # vibration_h
            temp_col    = 2   # temperature
            current_col = 3   # motor_current

            signal  = df.iloc[:, vib_col].values.astype(float)
            temp    = df.iloc[:, temp_col].values.astype(float)
            current = df.iloc[:, current_col].values.astype(float)

            windows = sliding_window(signal)
            for i, w in enumerate(windows):
                s = i * STEP
                e = s + WINDOW_SIZE
                records.append({
                    "vibration_rms": compute_rms(w),
                    "temperature"  : float(np.mean(temp[s:e])),
                    "motor_current": float(np.mean(current[s:e])),
                    "source"       : "KAIST",
                    "file"         : os.path.basename(fp),
                })
        except Exception as e:
            print(f"  [KAIST] 읽기 실패: {fp} / {e}")
            continue

    df_kaist = pd.DataFrame(records)
    print(f"  [KAIST] 로드 완료: {len(df_kaist):,} rows")
    return df_kaist


# ──────────────────────────────────────────────
# 2-3. XJTU-SY → 고장모드 라벨
#   폴더 구조: {운전조건}/BearingX_Y/N.csv
#   고장 유형: bearing 번호로 결정
#     Bearing1_x → outer_race_fault
#     Bearing2_x → inner_race_fault
#     Bearing3_x → ball_fault
# ──────────────────────────────────────────────
XJTU_LABEL_MAP = {
    "bearing1": "outer_race_fault",
    "bearing2": "inner_race_fault",
    "bearing3": "ball_fault",
}


def load_xjtu(xjtu_dir: str) -> pd.DataFrame:
    """
    XJTU-SY 데이터셋 로드
    폴더 구조: {운전조건}/{BearingX_Y}/{N}.csv
    컬럼: Horizontal, Vertical (헤더 있음)
    고장 유형: bearing 번호로 결정
    """
    records = []
    csv_files = glob.glob(os.path.join(xjtu_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print(f"  [XJTU] 경로 없음: {xjtu_dir}")
        return pd.DataFrame()

    for fp in csv_files:
        label = "unknown"
        fp_norm = fp.lower().replace("\\", "/")
        for key, val in XJTU_LABEL_MAP.items():
            if f"/{key}_" in fp_norm:
                label = val
                break
        if label == "unknown":
            continue

        try:
            df = pd.read_csv(fp, nrows=100000)

            if 'Horizontal' in df.columns:
                signal = df['Horizontal'].values.astype(float)
            elif 'horizontal' in df.columns:
                signal = df['horizontal'].values.astype(float)
            else:
                signal = df.iloc[:, 0].values.astype(float)

            for w in sliding_window(signal):
                records.append({
                    "vibration_rms": compute_rms(w),
                    "motor_current": 0.0,
                    "fault_label"  : label,
                    "source"       : "XJTU",
                })
        except Exception as e:
            print(f"  [XJTU] 읽기 실패: {fp} / {e}")
            continue

    df_xjtu = pd.DataFrame(records)
    if not df_xjtu.empty:
        print(f"  [XJTU] 로드 완료: {len(df_xjtu):,} rows")
        print(f"  [XJTU] 클래스 분포:\n{df_xjtu['fault_label'].value_counts()}")
    return df_xjtu


# ══════════════════════════════════════════════
# 3. 피처 엔지니어링
# ══════════════════════════════════════════════
def engineer_features(df: pd.DataFrame,
                      rms_col: str = "vibration_rms",
                      window: int = 5) -> pd.DataFrame:
    """
    short_trend, rolling_trend, temp_residual 생성
    temperature 컬럼이 없으면 temp_residual = 0 으로 처리
    """
    df = df.copy().reset_index(drop=True)

    # short_trend: 순간 변화율 (이전 window 대비)
    df['short_trend'] = (
        df[rms_col] - df[rms_col].shift(1)
    ) / df[rms_col].shift(1).replace(0, np.nan)

    # rolling_trend: 중기 열화 추세 (5창 평균 대비)
    roll_mean = df[rms_col].rolling(window=window, min_periods=1).mean()
    df['rolling_trend'] = (
        df[rms_col] - roll_mean
    ) / roll_mean.replace(0, np.nan)

    # temp_residual
    if 'temperature' in df.columns:
        # expected_temp: 과거 데이터만 (미래 정보 차단 — target leakage 없음)
        df['expected_temp'] = df['temperature'].rolling(
            window=window, min_periods=1
        ).mean().shift(1)
        df['temp_residual'] = df['temperature'] - df['expected_temp']
    else:
        df['temp_residual'] = 0.0

    df = df.dropna(subset=['short_trend', 'rolling_trend'])
    return df


# ══════════════════════════════════════════════
# 4. 사전 검증 — corr(temp_residual, RUL)
# ══════════════════════════════════════════════
def check_leakage_corr(df: pd.DataFrame) -> float:
    """
    학습 전 temp_residual ↔ RUL 상관관계 확인
    corr > 0.7 이면 leakage 가능성 → 경고 출력
    """
    if 'RUL' not in df.columns or 'temp_residual' not in df.columns:
        return 0.0

    corr_val = df['temp_residual'].corr(df['RUL'])
    print(f"\n[사전 검증] corr(temp_residual, RUL) = {corr_val:.4f}")

    if abs(corr_val) > 0.7:
        print("  ⚠️  leakage 가능성 높음 → temp_residual 제거 검토 필요")
    else:
        print("  ✅  허용 범위 → 학습 진행")

    return corr_val


# ══════════════════════════════════════════════
# 5. Model 1 — XGBoost RUL 회귀
# ══════════════════════════════════════════════
# 기본 피처 목록 — 실제 사용 피처는 데이터 상태에 따라 동적 결정
FEATURES_RUL_BASE = ['vibration_rms', 'short_trend', 'rolling_trend',
                     'motor_current', 'temp_residual']


def resolve_features(df: pd.DataFrame, has_kaist: bool, has_temp: bool) -> list:
    """
    데이터 가용성에 따라 실제 사용 피처 결정
    - KAIST 없음 → motor_current 제외 (상수값은 feature 아님)
    - temperature 실데이터 없음 → temp_residual 제외 (항상 0이면 의미 없음)
    """
    features = ['vibration_rms', 'short_trend', 'rolling_trend']
    if has_kaist:
        features.append('motor_current')
    else:
        print("  ℹ️  motor_current 제외 (KAIST 없음 — 상수값은 피처 아님)")
    if has_temp:
        features.append('temp_residual')
    else:
        print("  ℹ️  temp_residual 제외 (temperature 실데이터 없음 — 항상 0)")
    return features


def train_xgboost_rul(df: pd.DataFrame,
                      has_kaist: bool = False,
                      has_temp: bool = False,
                      rul_col: str = 'RUL',
                      grade_fn=None,
                      label: str = '') -> dict:
    """
    XGBoost RUL 회귀 학습 + 검증 파이프라인
    - GroupShuffleSplit: bearing_id 기준 split (동일 bearing train/test 혼입 방지)
    - rul_col: 사용할 RUL 컬럼 ('RUL' / 'RUL_norm' / 'RUL_clip')
    - grade_fn: RUL 값 → 등급 변환 함수 (rul_col 스케일에 맞는 함수 전달)
    반환: {model, r2, importance, delta_r2, shuffle_drop, features}
    """
    if grade_fn is None:
        grade_fn = assign_rul_grade_norm
    if label:
        print(f"\n{'─'*40}")
        print(f"  실험: {label}")
        print(f"{'─'*40}")

    features = resolve_features(df, has_kaist, has_temp)
    df = df.dropna(subset=features + [rul_col, 'bearing_id'])

    X      = df[features].values
    y      = df[rul_col].values
    groups = df['bearing_id'].values

    # ── GroupShuffleSplit: bearing 단위 분리
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_bearings = set(groups[train_idx])
    test_bearings  = set(groups[test_idx])
    print(f"\n[GroupSplit] train bearings: {sorted(train_bearings)}")
    print(f"[GroupSplit] test  bearings: {sorted(test_bearings)}")
    assert not train_bearings & test_bearings, "bearing overlap 발생 — split 오류"

    # ── 5-1. 사전 제한 파라미터
    params = dict(
        n_estimators    = 300,
        max_depth       = 6,
        learning_rate   = 0.05,
        colsample_bytree= 0.7,   # temp_residual dominance 사전 제한
        subsample       = 0.8,
        reg_lambda      = 1.0,
        random_state    = 42,
    )
    model_A = XGBRegressor(**params)
    model_A.fit(X_train, y_train)

    r2_A = r2_score(y_test, model_A.predict(X_test))
    print(f"\n[XGBoost] R²(전체 피처) = {r2_A:.4f}")
    print(f"  사용 피처: {features}")

    # ── 5-2. feature importance 확인
    importance = dict(zip(features, model_A.feature_importances_))
    print("\n[Feature Importance]")
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(v * 40)
        print(f"  {k:<20} {v:.4f}  {bar}")

    # temp_residual dominance 체크 → 재학습
    if importance.get('temp_residual', 0) > 0.5:
        print("\n  ⚠️  temp_residual > 0.5 → colsample_bytree=0.5 재학습")
        params['colsample_bytree'] = 0.5
        model_A = XGBRegressor(**params)
        model_A.fit(X_train, y_train)
        r2_A = r2_score(y_test, model_A.predict(X_test))
        importance = dict(zip(features, model_A.feature_importances_))
        print(f"  재학습 R² = {r2_A:.4f}")

    # ── 5-3. ablation — temp_residual 제거 (temp 있는 경우만)
    delta_r2 = 0.0
    if 'temp_residual' in features:
        feat_no_temp  = [f for f in features if f != 'temp_residual']
        X_b           = df[feat_no_temp].values
        X_train_b     = X_b[train_idx]   # 동일 GroupSplit 인덱스 사용
        X_test_b      = X_b[test_idx]
        model_B = XGBRegressor(**params)
        model_B.fit(X_train_b, y_train)
        r2_B     = r2_score(y_test, model_B.predict(X_test_b))
        delta_r2 = r2_A - r2_B

        print(f"\n[Ablation] R²(B, temp_residual 제거) = {r2_B:.4f}")
        print(f"[Ablation] ΔR² = {delta_r2:.4f}")
        if delta_r2 > 0.1:
            print("  ⚠️  leakage 가능성 → temp_residual 제거 후 재설계 검토")
        elif delta_r2 > 0.05:
            print("  ⚠️  영향 있음 → colsample 추가 조정 권장")
        else:
            print("  ✅  영향 미미 → 현재 구조 유지")
    else:
        print("\n[Ablation] temp_residual 미사용 — 생략")

    # ── 5-4. trend 유효성 검증
    corr_short   = df['vibration_rms'].corr(df['short_trend'])
    corr_rolling = df['vibration_rms'].corr(df['rolling_trend'])
    print(f"\n[Trend 검증] corr(rms, short_trend)   = {corr_short:.4f}")
    print(f"[Trend 검증] corr(rms, rolling_trend)  = {corr_rolling:.4f}")

    imp_trend_total = (importance.get('short_trend', 0)
                       + importance.get('rolling_trend', 0))
    if imp_trend_total < 0.1:
        print("  ⚠️  trend importance 합계 < 10% → rms 제거 실험 권장")
    else:
        print(f"  ✅  trend importance 합계 = {imp_trend_total:.4f}")

    # ── 5-5. shuffle test
    rng           = np.random.RandomState(42)
    shuffle_idx   = rng.permutation(len(y_train))
    model_shuffle = XGBRegressor(**params)
    model_shuffle.fit(X_train, y_train[shuffle_idx])
    r2_shuffle = r2_score(y_test, model_shuffle.predict(X_test))
    drop       = r2_A - r2_shuffle

    print(f"\n[Shuffle Test] R²(shuffle) = {r2_shuffle:.4f}  /  drop = {drop:.4f}")
    if drop < 0.05:
        print("  ⚠️  구조 문제 의심 → 피처 재검토")
    elif drop > 0.2:
        print("  ✅  패턴 학습 확인")
    else:
        print("  ℹ️  중간 수준 — XGBoost 분포 기반 특성, 추가 분석 권장")

    # ── 5-6. 등급 정확도
    y_pred_grade  = [grade_fn(v) for v in model_A.predict(X_test)]
    y_true_grade  = [grade_fn(v) for v in y_test]
    grade_correct = sum(p == t for p, t in zip(y_pred_grade, y_true_grade))
    grade_acc     = grade_correct / len(y_test)
    print(f"\n[등급 정확도] 전체 = {grade_acc:.4f}")

    # R² 범위별 발표 전략 출력
    print(f"\n[평가 요약] R² = {r2_A:.4f}")
    if r2_A >= 0.5:
        print("  → 정량 예측 중심 발표 가능")
    elif r2_A >= 0.3:
        print("  → 내부 기준 통과. 데이터 한계 명시 후 발표")
    else:
        print("  → 정량 예측 한계. 등급 정확도 중심으로 전환")

    _plot_importance(importance, os.path.join(OUT_DIR, "feature_importance.png"))

    return {
        "model"       : model_A,
        "r2"          : r2_A,
        "importance"  : importance,
        "delta_r2"    : delta_r2,
        "shuffle_drop": drop,
        "grade_acc"   : grade_acc,
        "features"    : features,
    }


def _plot_importance(importance: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = list(importance.keys())
    vals = list(importance.values())
    ax.barh(keys, vals, color='steelblue')
    ax.set_xlabel("Importance")
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 6. Model 2 — RandomForest 고장모드 분류
# ══════════════════════════════════════════════
# temperature 제거 확정 (synthetic rule fitting 문제)
# motor_current는 KAIST 가용 여부에 따라 동적 결정


def train_random_forest(df: pd.DataFrame, has_kaist: bool = False) -> dict:
    """
    RandomForest 고장모드 분류 학습
    df에 fault_label 컬럼 필요
    motor_current: KAIST 없으면 제외 (상수 0은 피처 아님)
    """
    features_clf = ['vibration_rms', 'short_trend', 'rolling_trend']
    if has_kaist and 'motor_current' in df.columns:
        features_clf.append('motor_current')
    else:
        print("  ℹ️  motor_current 제외 (KAIST 없음)")

    df = df.dropna(subset=features_clf + ['fault_label'])
    X  = df[features_clf].values
    y  = df['fault_label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 8,
        class_weight  = 'balanced',
        random_state  = 42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n[RandomForest] Classification Report")
    print(classification_report(y_test, y_pred))

    importance = dict(zip(features_clf, model.feature_importances_))
    print("[Feature Importance]")
    for k, v in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"  {k:<20} {v:.4f}")

    return {"model": model, "importance": importance, "features": features_clf}


# ══════════════════════════════════════════════
# 7. Model 3 — Autoencoder 이상 탐지
# ══════════════════════════════════════════════

def train_autoencoder(df_normal: pd.DataFrame,
                      has_kaist: bool = False,
                      has_temp: bool = False) -> dict:
    """
    Autoencoder 학습 (LOW 구간 정상 데이터만 사용)
    threshold: FPR ≤ 5% 기준으로 3개 후보 중 선택
    피처: XGBoost와 동일한 가용성 기준 적용
    """
    ae_features = ['vibration_rms', 'short_trend', 'rolling_trend']
    if has_kaist and 'motor_current' in df_normal.columns:
        ae_features.append('motor_current')
    if has_temp and 'temp_residual' in df_normal.columns:
        ae_features.append('temp_residual')
    print(f"  AE 사용 피처: {ae_features}")

    df_normal = df_normal.dropna(subset=ae_features)
    X = df_normal[ae_features].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_features = X_scaled.shape[1]

    # ── 네트워크 구조
    inp = keras.Input(shape=(n_features,))
    enc = layers.Dense(32, activation='relu')(inp)
    enc = layers.Dense(16, activation='relu')(enc)   # bottleneck
    dec = layers.Dense(32, activation='relu')(enc)
    out = layers.Dense(n_features, activation='linear')(dec)
    autoencoder = keras.Model(inp, out)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(
        X_scaled, X_scaled,
        epochs          = 50,
        batch_size      = 64,
        validation_split= 0.1,
        verbose         = 0,
    )

    # ── reconstruction error
    X_pred  = autoencoder.predict(X_scaled, verbose=0)
    errors  = np.mean((X_scaled - X_pred) ** 2, axis=1)

    mean, std = errors.mean(), errors.std()
    candidates = {
        "percentile_95": float(np.percentile(errors, 95)),
        "mean+2std"    : float(mean + 2 * std),
        "mean+3std"    : float(mean + 3 * std),
    }

    print("\n[Autoencoder] Threshold 후보 (FPR ≤ 5% 기준으로 선택)")
    selected = None
    for name, val in candidates.items():
        fpr = float(np.mean(errors > val))
        mark = "✅" if fpr <= 0.05 else "  "
        print(f"  {mark} {name:<18} threshold={val:.6f}  FPR={fpr:.4f}")
        if fpr <= 0.05 and selected is None:
            selected = (name, val)

    if selected is None:
        # 모든 후보가 FPR > 5% 이면 가장 보수적인 값 선택
        selected = ("mean+3std", candidates["mean+3std"])
        print(f"  → 전체 FPR 초과. mean+3std 선택")
    else:
        print(f"  → 선택: {selected[0]}  threshold={selected[1]:.6f}")

    threshold = selected[1]

    # threshold 시각화 저장
    _plot_ae_threshold(errors, threshold, os.path.join(OUT_DIR, "ae_threshold.png"))

    return {
        "model"    : autoencoder,
        "scaler"   : scaler,
        "threshold": threshold,
        "errors"   : errors,
    }


def _plot_ae_threshold(errors: np.ndarray, threshold: float, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors, alpha=0.6, label='Reconstruction Error')
    ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold={threshold:.4f}')
    ax.set_xlabel("Window Index")
    ax.set_ylabel("MSE")
    ax.set_title("Autoencoder Reconstruction Error (정상 구간)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 8. T_failure 검증 그래프 (RMS + 온도)
# ══════════════════════════════════════════════
def plot_tfailure_verification(df: pd.DataFrame, save_path: str):
    """
    T_failure=124 근거: RMS + temperature 변곡점 동시 확인
    """
    if 'window_idx' not in df.columns:
        return
    if 'vibration_rms' not in df.columns:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(df['window_idx'], df['vibration_rms'], color='steelblue')
    axes[0].axvline(T_FAILURE, color='red', linestyle='--', label=f'T_failure={T_FAILURE}')
    axes[0].set_ylabel("Vibration RMS")
    axes[0].set_title("RMS 상승 → 기계적 이상 신호")
    axes[0].legend()

    if 'temperature' in df.columns:
        axes[1].plot(df['window_idx'], df['temperature'], color='darkorange')
        axes[1].axvline(T_FAILURE, color='red', linestyle='--')
        axes[1].set_ylabel("Temperature")
        axes[1].set_title("Temperature 상승 → 열적 반응 (두 신호 동시 변화 = 고장 시작)")
    else:
        axes[1].text(0.5, 0.5, "temperature 데이터 없음 (기상청 API 연동 후 추가)",
                     ha='center', va='center', transform=axes[1].transAxes)

    axes[1].set_xlabel("Window Index")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [저장] {save_path}")


# ══════════════════════════════════════════════
# 9. 메인 실행
# ══════════════════════════════════════════════
def main():
    print("=" * 60)
    print("AI-PASS 예지보전 파이프라인 시작")
    print("=" * 60)

    # ── Step 1. 데이터 로드
    print("\n[Step 1] 데이터 로드")
    df_femto = load_femto(FEMTO_DIR)
    df_kaist = load_kaist(KAIST_DIR)
    df_xjtu  = load_xjtu(XJTU_DIR)

    # 데이터셋 중 하나라도 비어 있으면 경고
    if df_femto.empty:
        print("  ⚠️  FEMTO 로드 실패 — 경로 또는 파일명 확인 필요")
    if df_kaist.empty:
        print("  ⚠️  KAIST 로드 실패 — 경로 또는 파일명 확인 필요")
    if df_xjtu.empty:
        print("  ⚠️  XJTU 로드 실패 — 경로 또는 파일명 확인 필요")

    # ── Step 2. 피처 엔지니어링
    print("\n[Step 2] 피처 엔지니어링")

    # 데이터 가용성 플래그
    has_kaist = not df_kaist.empty

    # FEMTO 피처 엔지니어링 (KAIST 상수 병합 제거 — 스칼라 평균은 피처로 무의미)
    if not df_femto.empty:
        if has_kaist:
            print("  ℹ️  KAIST 존재하나 FEMTO와 시간축 매칭 불가 → motor_current/temperature 미사용")
        else:
            print("  ℹ️  KAIST 없음 → motor_current / temperature 피처 제외")
        df_rul = engineer_features(df_femto)
        print(f"  RUL 학습용 데이터: {len(df_rul):,} rows")
    else:
        df_rul = pd.DataFrame()

    # XJTU 피처 엔지니어링
    if not df_xjtu.empty:
        df_xjtu = df_xjtu.reset_index(drop=True)
        df_xjtu['temperature'] = 0.0
        fault_backup = df_xjtu['fault_label'].values  # numpy array (positional)
        df_clf = engineer_features(df_xjtu)
        df_clf['fault_label'] = fault_backup[df_clf.index.values]
        if has_kaist:
            df_clf['motor_current'] = df_kaist['motor_current'].mean()
        print(f"  분류 학습용 데이터: {len(df_clf):,} rows")
    else:
        df_clf = pd.DataFrame()

    # ── Step 3. 사전 검증
    print("\n[Step 3] 사전 검증 — corr(temp_residual, RUL)")
    if not df_rul.empty:
        check_leakage_corr(df_rul)

    # ── Step 4. T_failure 검증 그래프
    print("\n[Step 4] T_failure 검증 그래프 생성")
    if not df_femto.empty:
        sample_bearing = df_femto[df_femto['bearing_id'] == df_femto['bearing_id'].iloc[0]]
        plot_tfailure_verification(
            sample_bearing,
            os.path.join(OUT_DIR, "t_failure_verification.png")
        )

    # ── Step 5. XGBoost RUL 학습 — 옵션A(정규화) / 옵션B(클리핑) 비교
    print("\n[Step 5] XGBoost RUL 회귀 학습 — 옵션 비교 실험")
    result_xgb = {}
    if not df_rul.empty and len(df_rul) > 100:
        # FEMTO 단독 학습 → KAIST 피처 미사용 (상수값은 피처 아님)
        result_A = train_xgboost_rul(
            df_rul, has_kaist=False, has_temp=False,
            rul_col='RUL_norm', grade_fn=assign_rul_grade_norm,
            label='옵션A — RUL 정규화 (0~1)'
        )
        result_B = train_xgboost_rul(
            df_rul, has_kaist=False, has_temp=False,
            rul_col='RUL_clip', grade_fn=assign_rul_grade_clip,
            label='옵션B — RUL 클리핑 (max=200)'
        )

        print("\n[Step 5 비교 요약]")
        print(f"  {'항목':<20} {'옵션A (정규화)':<18} {'옵션B (클리핑)'}")
        print(f"  {'─'*56}")
        print(f"  {'R²':<20} {result_A['r2']:<18.4f} {result_B['r2']:.4f}")
        print(f"  {'ΔR² (leakage)':<20} {result_A['delta_r2']:<18.4f} {result_B['delta_r2']:.4f}")
        print(f"  {'Shuffle drop':<20} {result_A['shuffle_drop']:<18.4f} {result_B['shuffle_drop']:.4f}")
        print(f"  {'등급 정확도':<19} {result_A['grade_acc']:<18.4f} {result_B['grade_acc']:.4f}")

        # 더 나은 쪽을 result_xgb로 확정
        result_xgb = result_A if result_A['r2'] >= result_B['r2'] else result_B
        winner = '옵션A (정규화)' if result_A['r2'] >= result_B['r2'] else '옵션B (클리핑)'
        print(f"\n  → 채택: {winner}  (R² 기준)")
    else:
        print("  ⚠️  RUL 학습 데이터 부족 — FEMTO 로드 확인 필요")

    # ── Step 6. RandomForest 분류 학습
    print("\n[Step 6] RandomForest 고장모드 분류 학습")
    if not df_clf.empty and len(df_clf) > 100:
        result_rf = train_random_forest(df_clf, has_kaist=has_kaist)
    else:
        print("  ⚠️  분류 학습 데이터 부족 — XJTU 로드 확인 필요")
        result_rf = {}

    # ── Step 7. Autoencoder 이상 탐지
    print("\n[Step 7] Autoencoder 이상 탐지 학습")
    if not df_rul.empty:
        df_normal = df_rul[df_rul['RUL_norm'] > 0.5].copy()
        print(f"  정상 구간 (RUL_norm > 0.5) 데이터: {len(df_normal):,} rows")
        if len(df_normal) > 50:
            result_ae = train_autoencoder(df_normal, has_kaist=False, has_temp=False)
        else:
            print("  ⚠️  정상 데이터 부족")
            result_ae = {}
    else:
        result_ae = {}

    # ── 최종 요약
    print("\n" + "=" * 60)
    print("파이프라인 완료 — 결과 요약")
    print("=" * 60)
    if result_xgb:
        print(f"  XGBoost R²      : {result_xgb.get('r2', 'N/A'):.4f}")
        print(f"  ΔR² (leakage)   : {result_xgb.get('delta_r2', 'N/A'):.4f}")
        print(f"  Shuffle drop    : {result_xgb.get('shuffle_drop', 'N/A'):.4f}")
        print(f"  등급 정확도      : {result_xgb.get('grade_acc', 'N/A'):.4f}")
    if result_ae:
        print(f"  AE threshold    : {result_ae.get('threshold', 'N/A'):.6f}")

    # ── 모델 저장
    print("\n[모델 저장]")
    if result_xgb:
        path = os.path.join(OUT_DIR, "xgboost_rul.pkl")
        joblib.dump(result_xgb['model'], path)
        print(f"  [저장] {path}")
    if result_rf:
        path = os.path.join(OUT_DIR, "rf_fault.pkl")
        joblib.dump(result_rf['model'], path)
        print(f"  [저장] {path}")
    if result_ae:
        path_model = os.path.join(OUT_DIR, "autoencoder.keras")
        result_ae['model'].save(path_model)
        print(f"  [저장] {path_model}")
        path_scaler = os.path.join(OUT_DIR, "ae_scaler.pkl")
        joblib.dump(result_ae['scaler'], path_scaler)
        print(f"  [저장] {path_scaler}")
        path_thresh = os.path.join(OUT_DIR, "ae_threshold.pkl")
        joblib.dump(result_ae['threshold'], path_thresh)
        print(f"  [저장] {path_thresh}")

    print(f"\n  출력 파일 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()