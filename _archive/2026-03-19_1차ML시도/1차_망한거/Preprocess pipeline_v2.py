"""
AI-Pass 예지보전 전처리 파이프라인 v2.0
========================================
변경사항 (v1 → v2):
  - 등급별 목표 샘플 수 고정 (TARGET_PER_GRADE = 8,000)
  - LOW  : 언더샘플링 (랜덤 샘플링, 시계열 순서 유지)
  - 나머지: stride 축소로 오버샘플링 (시계열 순서 유지)

출력:
  D:/project/예지보전/output/
    ├── kaist_preprocessed.csv
    ├── femto_preprocessed.csv
    ├── combined_train.csv        (최종 학습용, 32,000개)
    ├── scaler.pkl
    └── preprocess_summary.xlsx

실행: python preprocess_pipeline_v2.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time
import re
from datetime import datetime

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

KAIST_DIR  = Path(r"D:\project\데이터셋\Vibration_Bearing_RuntoFailure")
FEMTO_ROOT = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")
OUTPUT_DIR = Path(r"D:\project\예지보전\output")

BASE_LIFE_DAYS  = 180
KAIST_WINDOW    = 25600
FEMTO_WINDOW    = 256
KAIST_FILE_ROWS = 2_000_000
FEMTO_FILE_ROWS = 2560

TARGET_PER_GRADE = 8000  # 등급당 목표 샘플 수

FEMTO_DIRS = [
    FEMTO_ROOT / "Learning_set",
    FEMTO_ROOT / "Full_Test_Set",
]

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
        grade: (min(v), max(v), len(v))
        for grade, v in grade_indices.items() if v
    }


def calc_stride_for_target(
    total_rows: int,
    window: int,
    target: int,
) -> int:
    """
    목표 샘플 수에 맞는 stride 계산
    target 샘플을 얻으려면 stride가 얼마여야 하는지 역산
    """
    if target <= 0:
        return window
    stride = max(1, (total_rows - window) // target)
    return stride


def undersample_keep_order(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    시계열 순서를 유지하면서 n개 균등 샘플링
    (랜덤 추출이 아닌 등간격 추출)
    """
    if len(df) <= n:
        return df
    indices = np.linspace(0, len(df) - 1, n, dtype=int)
    return df.iloc[indices].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_kaist() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1. KAIST 전처리")
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
    print(f"총 파일 수: {n_files}개")

    # 등급별 경계
    boundaries = calc_grade_boundaries(n_files)

    # 등급별 총 행 수 계산
    grade_total_rows = {}
    for grade, (_, _, cnt) in boundaries.items():
        grade_total_rows[grade] = cnt * KAIST_FILE_ROWS

    # 등급별 stride 계산
    # LOW: 언더샘플링 예정 → 자연 stride 사용 (window 크기)
    # 나머지: TARGET_PER_GRADE에 맞게 stride 역산
    strides = {}
    for grade, total_rows in grade_total_rows.items():
        if grade == "LOW":
            strides[grade] = KAIST_WINDOW  # 자연 샘플링
        else:
            strides[grade] = calc_stride_for_target(
                total_rows, KAIST_WINDOW, TARGET_PER_GRADE
            )

    print(f"\n등급별 stride 및 예상 샘플 수:")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if g not in boundaries:
            continue
        _, _, cnt = boundaries[g]
        total_rows = cnt * KAIST_FILE_ROWS
        stride = strides[g]
        expected = (total_rows - KAIST_WINDOW) // stride + 1
        note = "(언더샘플링 예정)" if g == "LOW" else ""
        print(f"  {g:10s}: stride={stride:>8,} → 예상 {expected:>6,}개 {note}")

    # 등급별 샘플 수집
    grade_rows = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}

    for file_idx, (f, ts) in enumerate(
        tqdm(file_info, desc="KAIST", unit="파일")
    ):
        rul    = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
        grade  = rul_to_risk(rul)
        stride = strides.get(grade, KAIST_WINDOW)

        try:
            df = pd.read_csv(
                f, header=None, sep=",",
                names=["vib_h", "vib_v", "temp", "ambient_temp"]
            )
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            tqdm.write(f"[WARN] {f.name}: {e}")
            continue

        vib_h    = df["vib_h"].values
        vib_v    = df["vib_v"].values
        temp_val = float(df["temp"].iloc[0])
        amb_val  = float(df["ambient_temp"].iloc[0])
        season   = get_season(ts)
        op_hours = float(file_idx)

        start = 0
        while start + KAIST_WINDOW <= len(df):
            end   = start + KAIST_WINDOW
            rms_h = calc_rms(vib_h[start:end])
            rms_v = calc_rms(vib_v[start:end])
            rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

            avg_heat     = 35.0
            expected_tmp = amb_val + avg_heat
            temp_resid   = round(temp_val - expected_tmp, 4)

            grade_rows[grade].append({
                "source"         : "kaist",
                "file_name"      : f.name,
                "timestamp"      : ts,
                "window_start"   : start,
                "vibration_rms"  : rms,
                "temperature"    : round(temp_val, 2),
                "temp_residual"  : temp_resid,
                "motor_current"  : generate_motor_current(rms),
                "operating_hours": op_hours,
                "ambient_temp"   : round(amb_val, 2),
                "wind_speed"     : round(np.random.uniform(1.0, 5.0), 2),
                "humidity"       : round(np.random.uniform(30.0, 80.0), 1),
                "season"         : season,
                "rul_days"       : rul,
                "risk_level"     : grade,
            })
            start += stride

    # 등급별 캡 적용
    print(f"\n등급별 캡 적용 (목표: {TARGET_PER_GRADE:,}개/등급):")
    final_rows = []
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        rows = grade_rows[g]
        before = len(rows)
        if before > TARGET_PER_GRADE:
            # 시계열 순서 유지 등간격 샘플링
            indices = np.linspace(0, before-1, TARGET_PER_GRADE, dtype=int)
            rows = [rows[i] for i in indices]
        after = len(rows)
        print(f"  {g:10s}: {before:>7,} → {after:>6,}개")
        final_rows.extend(rows)

    result = pd.DataFrame(final_rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"\nKAIST 최종: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>6,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. FEMTO 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_femto() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2. FEMTO 전처리")
    print("=" * 60)

    # 전체 베어링 목록 수집
    all_bearing_dirs = []
    for femto_dir in FEMTO_DIRS:
        if not femto_dir.exists():
            continue
        for bearing_dir in sorted(d for d in femto_dir.iterdir() if d.is_dir()):
            if list(bearing_dir.glob("acc_*.csv")):
                all_bearing_dirs.append(bearing_dir)

    print(f"총 베어링 수: {len(all_bearing_dirs)}개")

    # 전체 등급별 샘플 수집
    grade_rows = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}

    for bearing_dir in all_bearing_dirs:
        acc_files = sorted(bearing_dir.glob("acc_*.csv"))
        n_files   = len(acc_files)
        boundaries = calc_grade_boundaries(n_files)

        # 등급별 stride 계산 (베어링 단위)
        grade_total_rows = {}
        for grade, (_, _, cnt) in boundaries.items():
            grade_total_rows[grade] = cnt * FEMTO_FILE_ROWS

        strides = {}
        for grade, total_rows in grade_total_rows.items():
            if grade == "LOW":
                strides[grade] = FEMTO_WINDOW
            else:
                strides[grade] = calc_stride_for_target(
                    total_rows, FEMTO_WINDOW, TARGET_PER_GRADE
                )

        for file_idx, f in enumerate(
            tqdm(acc_files, desc=f"  {bearing_dir.name}", unit="파일", leave=False)
        ):
            rul    = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
            grade  = rul_to_risk(rul)
            stride = strides.get(grade, FEMTO_WINDOW)

            try:
                df = pd.read_csv(
                    f, header=None, sep=",",
                    names=["hour","min","sec","usec","h_acc","v_acc"]
                )
                df = df.apply(pd.to_numeric, errors="coerce")
            except Exception:
                continue

            h_acc   = df["h_acc"].values
            v_acc   = df["v_acc"].values
            op_hours = file_idx / 3600.0

            start = 0
            while start + FEMTO_WINDOW <= len(df):
                end   = start + FEMTO_WINDOW
                rms_h = calc_rms(h_acc[start:end])
                rms_v = calc_rms(v_acc[start:end])
                rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

                temp_val     = round(25.0 + rms * 5.0 + np.random.normal(0, 1), 2)
                amb_val      = round(np.random.uniform(15.0, 30.0), 2)
                expected_tmp = amb_val + 35.0
                temp_resid   = round(temp_val - expected_tmp, 4)

                grade_rows[grade].append({
                    "source"         : f"femto",
                    "file_name"      : bearing_dir.name,
                    "timestamp"      : None,
                    "window_start"   : start,
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
                    "risk_level"     : grade,
                })
                start += stride

    # 등급별 캡 적용
    print(f"\n등급별 캡 적용 (목표: {TARGET_PER_GRADE:,}개/등급):")
    final_rows = []
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        rows = grade_rows[g]
        before = len(rows)
        if before > TARGET_PER_GRADE:
            indices = np.linspace(0, before-1, TARGET_PER_GRADE, dtype=int)
            rows = [rows[i] for i in indices]
        after = len(rows)
        print(f"  {g:10s}: {before:>7,} → {after:>6,}개")
        final_rows.extend(rows)

    result = pd.DataFrame(final_rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"\nFEMTO 최종: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>6,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 3. 정규화 + 합산
# ══════════════════════════════════════════════════════════════

def normalize_and_combine(
    kaist_df: pd.DataFrame,
    femto_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:

    print("\n" + "=" * 60)
    print("STEP 3. 정규화 + 합산")
    print("=" * 60)

    scaler = StandardScaler()
    scaler.fit(kaist_df[FEATURE_COLS])
    print("KAIST 기준 Z-score fit 완료")

    kaist_scaled = kaist_df.copy()
    femto_scaled = femto_df.copy()
    kaist_scaled[FEATURE_COLS] = scaler.transform(kaist_df[FEATURE_COLS])
    femto_scaled[FEATURE_COLS] = scaler.transform(femto_df[FEATURE_COLS])
    print("정규화 완료")

    combined = pd.concat([kaist_scaled, femto_scaled], ignore_index=True)
    dist  = Counter(combined["risk_level"])
    total = len(combined)

    print(f"\n최종 합산: {total:,}개 샘플")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>6,}개 ({dist[g]/total*100:.1f}%)")

    return combined, scaler


# ══════════════════════════════════════════════════════════════
# STEP 4. 저장
# ══════════════════════════════════════════════════════════════

def save_outputs(
    kaist_df: pd.DataFrame,
    femto_df: pd.DataFrame,
    combined: pd.DataFrame,
    scaler  : StandardScaler,
):
    print("\n" + "=" * 60)
    print("STEP 4. 저장")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kaist_df.to_csv(OUTPUT_DIR / "kaist_preprocessed.csv",
                    index=False, encoding="utf-8-sig")
    femto_df.to_csv(OUTPUT_DIR / "femto_preprocessed.csv",
                    index=False, encoding="utf-8-sig")
    combined.to_csv(OUTPUT_DIR / "combined_train.csv",
                    index=False, encoding="utf-8-sig")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")

    print(f"  kaist_preprocessed.csv : {len(kaist_df):>7,}행")
    print(f"  femto_preprocessed.csv : {len(femto_df):>7,}행")
    print(f"  combined_train.csv     : {len(combined):>7,}행")
    print(f"  scaler.pkl             : 저장 완료")

    # 요약 XLSX
    xlsx_path = OUTPUT_DIR / "preprocess_summary.xlsx"

    def make_dist_row(label, df):
        dist  = Counter(df["risk_level"])
        total = len(df)
        return {
            "구분"       : label,
            "총샘플수"   : total,
            "LOW"        : dist["LOW"],
            "MEDIUM"     : dist["MEDIUM"],
            "HIGH"       : dist["HIGH"],
            "CRITICAL"   : dist["CRITICAL"],
            "LOW_%"      : round(dist["LOW"]      / total * 100, 1),
            "MEDIUM_%"   : round(dist["MEDIUM"]   / total * 100, 1),
            "HIGH_%"     : round(dist["HIGH"]     / total * 100, 1),
            "CRITICAL_%" : round(dist["CRITICAL"] / total * 100, 1),
        }

    summary = pd.DataFrame([
        make_dist_row("KAIST",          kaist_df),
        make_dist_row("FEMTO",          femto_df),
        make_dist_row("합산(combined)", combined),
    ])

    kaist_file_summary = (
        kaist_df.groupby("file_name")
        .agg(
            timestamp      = ("timestamp",     "first"),
            vibration_rms  = ("vibration_rms", "mean"),
            temperature    = ("temperature",   "mean"),
            rul_days       = ("rul_days",       "first"),
            risk_level     = ("risk_level",     "first"),
            window_count   = ("window_start",   "count"),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    femto_summary = (
        femto_df.groupby("file_name")
        .agg(
            총샘플수   = ("vibration_rms", "count"),
            rms_min    = ("vibration_rms", "min"),
            rms_max    = ("vibration_rms", "max"),
            rms_mean   = ("vibration_rms", "mean"),
            LOW        = ("risk_level",    lambda x: (x=="LOW").sum()),
            MEDIUM     = ("risk_level",    lambda x: (x=="MEDIUM").sum()),
            HIGH       = ("risk_level",    lambda x: (x=="HIGH").sum()),
            CRITICAL   = ("risk_level",    lambda x: (x=="CRITICAL").sum()),
        )
        .reset_index()
    )

    scaler_info = pd.DataFrame({
        "feature" : FEATURE_COLS,
        "mean"    : scaler.mean_,
        "std"     : scaler.scale_,
    })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(            writer, sheet_name="최종요약",         index=False)
        kaist_file_summary.to_excel( writer, sheet_name="KAIST_파일단위",   index=False)
        femto_summary.to_excel(      writer, sheet_name="FEMTO_베어링요약", index=False)
        scaler_info.to_excel(        writer, sheet_name="Scaler_파라미터",  index=False)

    print(f"  preprocess_summary.xlsx: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    total_start = time.time()

    kaist_df         = preprocess_kaist()
    femto_df         = preprocess_femto()
    combined, scaler = normalize_and_combine(kaist_df, femto_df)
    save_outputs(kaist_df, femto_df, combined, scaler)

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")