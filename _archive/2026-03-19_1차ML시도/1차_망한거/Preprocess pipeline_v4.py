"""
AI-Pass 예지보전 전처리 파이프라인 v4.0
========================================
변경사항 (v3 → v4):
  - 윈도우 슬라이딩 제거
  - 파일 1개 (200만 행) 전체 읽기 → 집계 → 샘플 1개
  - KAIST: 129개 샘플 (200만 행 × 129파일 전체 사용)
  - FEMTO: 파일 1개 = 샘플 1개 (전체 행 집계)
  - StandardScaler → MinMaxScaler (피처 0~1)
  - rul_days → rul_norm (0~1) 정규화
  - 원본 분포 그대로 유지

출력:
  D:/project/예지보전/output_v4/
    ├── kaist_preprocessed.csv
    ├── femto_train.csv
    ├── femto_test.csv
    ├── combined_train.csv
    ├── combined_test.csv
    ├── scaler.pkl            (MinMaxScaler, 피처용)
    ├── rul_scaler.pkl        (MinMaxScaler, RUL용)
    └── preprocess_summary.xlsx

실행: python preprocess_pipeline_v4.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import time
import re
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

KAIST_DIR  = Path(r"D:\project\데이터셋\Vibration_Bearing_RuntoFailure")
FEMTO_ROOT = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")
OUTPUT_DIR = Path(r"D:\project\예지보전\output_v4")

BASE_LIFE_DAYS = 180

# test 전용 베어링 (학습에서 완전 제외)
TEST_BEARINGS = {"Bearing1_3", "Bearing1_5"}

FEMTO_DIRS = [
    FEMTO_ROOT / "Learning_set",
    FEMTO_ROOT / "Full_Test_Set",
]

FEATURE_COLS = [
    "vibration_rms",
    "temperature",
    "temp_residual",
    "motor_current",
    "operating_hours",
    "ambient_temp",
    "wind_speed",
    "humidity",
    "season",
]

# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def rul_to_risk(rul: float) -> str:
    if rul >= 31: return "LOW"
    if rul >= 16: return "MEDIUM"
    if rul >=  3: return "HIGH"
    return "CRITICAL"


def calc_rms(arr: np.ndarray) -> float:
    clean = arr[~np.isnan(arr)]
    if len(clean) == 0:
        return 0.0
    return float(np.sqrt(np.mean(clean ** 2)))


def get_season(dt: datetime) -> int:
    m = dt.month
    if m in [3, 4, 5]:   return 0
    if m in [6, 7, 8]:   return 1
    if m in [9, 10, 11]: return 2
    return 3


def generate_motor_current(rms: float) -> float:
    return round(max(0.5, 0.8 + 0.3 * rms + np.random.normal(0, 0.05)), 4)


def calc_grade_boundaries(n_files: int) -> dict:
    grade_indices = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
    for i in range(n_files):
        rul = BASE_LIFE_DAYS * (n_files - 1 - i) / (n_files - 1)
        grade_indices[rul_to_risk(rul)].append(i)
    return {
        g: (min(v), max(v), len(v))
        for g, v in grade_indices.items() if v
    }


def print_dist(df: pd.DataFrame, label: str):
    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"\n{label}: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 전처리 (200만 행 전체 집계)
# ══════════════════════════════════════════════════════════════

def preprocess_kaist() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1. KAIST 전처리 (200만 행 전체 집계)")
    print("=" * 60)

    all_files = sorted(KAIST_DIR.glob("LogFile_*.csv"))
    file_info = []
    for f in all_files:
        match = re.search(
            r"LogFile_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", f.name
        )
        if match:
            ts = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
            file_info.append((f, ts))
    file_info.sort(key=lambda x: x[1])
    n_files = len(file_info)
    print(f"총 파일 수: {n_files}개 (파일당 약 200만 행 전체 집계)")

    rows = []
    for file_idx, (f, ts) in enumerate(
        tqdm(file_info, desc="KAIST", unit="파일")
    ):
        rul   = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
        grade = rul_to_risk(rul)

        try:
            # 200만 행 전체 읽기
            df = pd.read_csv(
                f, header=None, sep=",",
                names=["vib_h", "vib_v", "temp", "ambient_temp"]
            )
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            tqdm.write(f"[WARN] {f.name}: {e}")
            continue

        # 전체 행 집계
        vib_h    = df["vib_h"].values
        vib_v    = df["vib_v"].values
        temp_arr = df["temp"].values
        amb_arr  = df["ambient_temp"].values

        rms_h = calc_rms(vib_h)
        rms_v = calc_rms(vib_v)
        rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

        # 온도: 전체 평균 (파일 내 저속 갱신값들의 평균)
        temp_val = round(float(np.nanmean(temp_arr)), 4)
        amb_val  = round(float(np.nanmean(amb_arr)), 4)

        avg_heat   = 35.0
        temp_resid = round(temp_val - (amb_val + avg_heat), 4)
        season     = get_season(ts)
        op_hours   = float(file_idx)  # 파일 1개 = 1시간

        rows.append({
            "source"         : "kaist",
            "file_name"      : f.name,
            "timestamp"      : ts,
            "vibration_rms"  : rms,
            "temperature"    : temp_val,
            "temp_residual"  : temp_resid,
            "motor_current"  : generate_motor_current(rms),
            "operating_hours": op_hours,
            "ambient_temp"   : amb_val,
            "wind_speed"     : round(np.random.uniform(1.0, 5.0), 2),
            "humidity"       : round(np.random.uniform(30.0, 80.0), 1),
            "season"         : season,
            "rul_days"       : rul,
            "rul_norm"       : 0.0,   # 정규화 후 채움
            "risk_level"     : grade,
        })

    result = pd.DataFrame(rows)
    print_dist(result, "KAIST")
    print(f"\n  vibration_rms: {result['vibration_rms'].min():.4f}"
          f" ~ {result['vibration_rms'].max():.4f}g")
    print(f"  temperature  : {result['temperature'].min():.2f}"
          f" ~ {result['temperature'].max():.2f}°C")
    print(f"  rul_days     : {result['rul_days'].min():.2f}"
          f" ~ {result['rul_days'].max():.2f}일")
    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. FEMTO 전처리 (파일 단위 집계, train/test 분리)
# ══════════════════════════════════════════════════════════════

def preprocess_femto() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STEP 2. FEMTO 전처리 (파일 단위 집계)")
    print(f"  test 베어링: {TEST_BEARINGS}")
    print("=" * 60)

    train_rows = []
    test_rows  = []

    for femto_dir in FEMTO_DIRS:
        if not femto_dir.exists():
            continue

        for bearing_dir in sorted(
            d for d in femto_dir.iterdir() if d.is_dir()
        ):
            acc_files = sorted(bearing_dir.glob("acc_*.csv"))
            if not acc_files:
                continue

            n_files = len(acc_files)
            is_test = bearing_dir.name in TEST_BEARINGS
            tag     = "[ TEST ]" if is_test else "[ TRAIN]"

            bearing_rows = []
            for file_idx, f in enumerate(
                tqdm(
                    acc_files,
                    desc=f"  {tag} {bearing_dir.name}",
                    unit="파일",
                    leave=False,
                )
            ):
                rul   = round(
                    BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4
                )
                grade = rul_to_risk(rul)

                try:
                    df = pd.read_csv(
                        f, header=None, sep=",",
                        names=["hour","min","sec","usec","h_acc","v_acc"]
                    )
                    df = df.apply(pd.to_numeric, errors="coerce")
                except Exception:
                    continue

                # 전체 행 집계
                rms_h = calc_rms(df["h_acc"].values)
                rms_v = calc_rms(df["v_acc"].values)
                rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

                # FEMTO 온도 없음 → 진동 기반 더미
                temp_val   = round(25.0 + rms * 5.0 + np.random.normal(0, 1), 2)
                amb_val    = round(np.random.uniform(15.0, 30.0), 2)
                temp_resid = round(temp_val - (amb_val + 35.0), 4)
                op_hours   = file_idx / 3600.0

                bearing_rows.append({
                    "source"         : "femto",
                    "file_name"      : bearing_dir.name,
                    "timestamp"      : None,
                    "vibration_rms"  : rms,
                    "temperature"    : temp_val,
                    "temp_residual"  : temp_resid,
                    "motor_current"  : generate_motor_current(rms),
                    "operating_hours": round(op_hours, 4),
                    "ambient_temp"   : amb_val,
                    "wind_speed"     : round(np.random.uniform(1.0, 5.0), 2),
                    "humidity"       : round(np.random.uniform(30.0, 80.0), 1),
                    "season"         : int(np.random.randint(0, 4)),
                    "rul_days"       : rul,
                    "rul_norm"       : 0.0,
                    "risk_level"     : grade,
                })

            dist  = Counter(r["risk_level"] for r in bearing_rows)
            total = len(bearing_rows)
            print(f"  {tag} {bearing_dir.name:20s}: {total:5,}개 | "
                  f"LOW={dist['LOW']:4,} MED={dist['MEDIUM']:3,} "
                  f"HIGH={dist['HIGH']:3,} CRIT={dist['CRITICAL']:3,}")

            if is_test:
                test_rows.extend(bearing_rows)
            else:
                train_rows.extend(bearing_rows)

    femto_train = pd.DataFrame(train_rows)
    femto_test  = pd.DataFrame(test_rows)

    print_dist(femto_train, "FEMTO train")
    print_dist(femto_test,  "FEMTO test (원본 분포)")

    return femto_train, femto_test


# ══════════════════════════════════════════════════════════════
# STEP 3. 정규화 (MinMaxScaler 0~1)
# ══════════════════════════════════════════════════════════════

def normalize_and_combine(
    kaist_df    : pd.DataFrame,
    femto_train : pd.DataFrame,
    femto_test  : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:

    print("\n" + "=" * 60)
    print("STEP 3. MinMaxScaler 정규화 (0~1)")
    print("=" * 60)

    # train 합산 후 scaler fit
    train_all = pd.concat([kaist_df, femto_train], ignore_index=True)

    # 피처 scaler (KAIST + FEMTO train 기준 fit)
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(train_all[FEATURE_COLS])

    # RUL scaler (0~BASE_LIFE_DAYS → 0~1)
    rul_scaler = MinMaxScaler()
    rul_scaler.fit([[0], [BASE_LIFE_DAYS]])

    print(f"  피처 scaler fit 완료 (train {len(train_all):,}개 기준)")
    print(f"  RUL scaler: 0 ~ {BASE_LIFE_DAYS}일 → 0 ~ 1")

    def apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[FEATURE_COLS] = feat_scaler.transform(df[FEATURE_COLS])
        out["rul_norm"]   = rul_scaler.transform(
            df["rul_days"].values.reshape(-1, 1)
        ).flatten().round(6)
        return out

    kaist_scaled       = apply_scaling(kaist_df)
    femto_train_scaled = apply_scaling(femto_train)
    femto_test_scaled  = apply_scaling(femto_test)

    combined_train = pd.concat(
        [kaist_scaled, femto_train_scaled], ignore_index=True
    )
    combined_test = femto_test_scaled.copy()

    print_dist(combined_train, "combined_train")
    print_dist(combined_test,  "combined_test (원본 분포)")

    print(f"\n  rul_norm 범위 확인:")
    print(f"    train: {combined_train['rul_norm'].min():.4f}"
          f" ~ {combined_train['rul_norm'].max():.4f}")
    print(f"    test : {combined_test['rul_norm'].min():.4f}"
          f" ~ {combined_test['rul_norm'].max():.4f}")

    return combined_train, combined_test, feat_scaler, rul_scaler


# ══════════════════════════════════════════════════════════════
# STEP 4. 저장
# ══════════════════════════════════════════════════════════════

def save_outputs(
    kaist_df      : pd.DataFrame,
    femto_train   : pd.DataFrame,
    femto_test    : pd.DataFrame,
    combined_train: pd.DataFrame,
    combined_test : pd.DataFrame,
    feat_scaler   : MinMaxScaler,
    rul_scaler    : MinMaxScaler,
):
    print("\n" + "=" * 60)
    print("STEP 4. 저장")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = {
        "kaist_preprocessed.csv": kaist_df,
        "femto_train.csv"       : femto_train,
        "femto_test.csv"        : femto_test,
        "combined_train.csv"    : combined_train,
        "combined_test.csv"     : combined_test,
    }
    for fname, df in files.items():
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {fname:35s}: {len(df):>8,}행")

    joblib.dump(feat_scaler, OUTPUT_DIR / "scaler.pkl")
    joblib.dump(rul_scaler,  OUTPUT_DIR / "rul_scaler.pkl")
    print(f"  {'scaler.pkl':35s}: 저장 완료 (피처 MinMaxScaler)")
    print(f"  {'rul_scaler.pkl':35s}: 저장 완료 (RUL MinMaxScaler)")

    # 요약 XLSX
    def make_row(label, df):
        dist  = Counter(df["risk_level"])
        total = len(df)
        return {
            "구분"        : label,
            "총샘플수"    : total,
            "LOW"         : dist["LOW"],
            "MEDIUM"      : dist["MEDIUM"],
            "HIGH"        : dist["HIGH"],
            "CRITICAL"    : dist["CRITICAL"],
            "LOW_%"       : round(dist["LOW"]      / total * 100, 1),
            "MEDIUM_%"    : round(dist["MEDIUM"]   / total * 100, 1),
            "HIGH_%"      : round(dist["HIGH"]     / total * 100, 1),
            "CRITICAL_%"  : round(dist["CRITICAL"] / total * 100, 1),
            "rul_norm_min": round(df["rul_norm"].min(), 4),
            "rul_norm_max": round(df["rul_norm"].max(), 4),
        }

    summary = pd.DataFrame([
        make_row("KAIST",          kaist_df),
        make_row("FEMTO train",    femto_train),
        make_row("FEMTO test",     femto_test),
        make_row("combined_train", combined_train),
        make_row("combined_test",  combined_test),
    ])

    # 피처 scaler 파라미터
    scaler_info = pd.DataFrame({
        "feature" : FEATURE_COLS,
        "min"     : feat_scaler.data_min_,
        "max"     : feat_scaler.data_max_,
        "scale"   : feat_scaler.scale_,
    })

    # RUL scaler 파라미터
    rul_info = pd.DataFrame({
        "항목"    : ["rul_min", "rul_max", "정규화범위"],
        "값"      : [0, BASE_LIFE_DAYS, "0~1"],
    })

    xlsx_path = OUTPUT_DIR / "preprocess_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(    writer, sheet_name="최종요약",        index=False)
        scaler_info.to_excel(writer, sheet_name="피처Scaler",      index=False)
        rul_info.to_excel(   writer, sheet_name="RUL_Scaler",      index=False)

    print(f"  {'preprocess_summary.xlsx':35s}: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    total_start = time.time()

    kaist_df              = preprocess_kaist()
    femto_train, femto_test = preprocess_femto()
    combined_train, combined_test, feat_scaler, rul_scaler = normalize_and_combine(
        kaist_df, femto_train, femto_test
    )
    save_outputs(
        kaist_df, femto_train, femto_test,
        combined_train, combined_test,
        feat_scaler, rul_scaler,
    )

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")