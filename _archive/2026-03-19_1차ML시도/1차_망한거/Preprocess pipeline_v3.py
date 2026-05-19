"""
AI-Pass 예지보전 전처리 파이프라인 v3.0
========================================
변경사항 (v2 → v3):
  - test용 베어링 분리 (학습에 한 번도 안 쓴 베어링)
  - train: KAIST 전체 + FEMTO (test 베어링 제외)
  - test : FEMTO Full_Test Bearing1_3, Bearing1_5 (전체 수명, 원본 분포)
  - test는 캡/언더샘플링 없이 원본 분포 그대로 저장

출력:
  D:/project/예지보전/output/
    ├── combined_train.csv   (학습용, 32,000개 균등)
    ├── combined_test.csv    (검증용, 원본 분포)
    ├── kaist_preprocessed.csv
    ├── femto_train.csv
    ├── femto_test.csv
    ├── scaler.pkl
    └── preprocess_summary.xlsx

실행: python preprocess_pipeline_v3.py
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

BASE_LIFE_DAYS   = 180
KAIST_WINDOW     = 25600
FEMTO_WINDOW     = 256
KAIST_FILE_ROWS  = 2_000_000
FEMTO_FILE_ROWS  = 2560
TARGET_PER_GRADE = 8000

# ── test 전용 베어링 (학습에서 완전 제외) ──
# 전체 수명 데이터 보유 + 충분한 파일 수 기준 선택
TEST_BEARINGS = {
    "Bearing1_3",  # Full_Test_Set, 2375개
    "Bearing1_5",  # Full_Test_Set, 2463개
}

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
        g: (min(v), max(v), len(v))
        for g, v in grade_indices.items() if v
    }


def calc_stride_for_target(total_rows: int, window: int, target: int) -> int:
    if target <= 0:
        return window
    return max(1, (total_rows - window) // target)


def apply_cap_keep_order(rows: list, target: int) -> list:
    """시계열 순서 유지 등간격 캡 적용"""
    if len(rows) <= target:
        return rows
    indices = np.linspace(0, len(rows)-1, target, dtype=int)
    return [rows[i] for i in indices]


def print_dist(df: pd.DataFrame, label: str):
    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"\n{label}: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 공통 윈도우 슬라이딩 함수
# ══════════════════════════════════════════════════════════════

def sliding_window_femto(
    bearing_dir : Path,
    n_files     : int,
    acc_files   : list,
    strides     : dict,
    is_test     : bool = False,
) -> list:
    """
    FEMTO 베어링 1개에 대해 윈도우 슬라이딩 수행
    is_test=True이면 stride=FEMTO_WINDOW (겹침 없음, 원본 분포)
    """
    rows = []
    for file_idx, f in enumerate(acc_files):
        rul    = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
        grade  = rul_to_risk(rul)
        stride = FEMTO_WINDOW if is_test else strides.get(grade, FEMTO_WINDOW)

        try:
            df = pd.read_csv(
                f, header=None, sep=",",
                names=["hour","min","sec","usec","h_acc","v_acc"]
            )
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            continue

        h_acc    = df["h_acc"].values
        v_acc    = df["v_acc"].values
        op_hours = file_idx / 3600.0

        start = 0
        while start + FEMTO_WINDOW <= len(df):
            end   = start + FEMTO_WINDOW
            rms_h = calc_rms(h_acc[start:end])
            rms_v = calc_rms(v_acc[start:end])
            rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

            temp_val     = round(25.0 + rms * 5.0 + np.random.normal(0, 1), 2)
            amb_val      = round(np.random.uniform(15.0, 30.0), 2)
            temp_resid   = round(temp_val - (amb_val + 35.0), 4)

            rows.append({
                "source"         : "femto",
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

    return rows


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 전처리 (train 전용)
# ══════════════════════════════════════════════════════════════

def preprocess_kaist() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1. KAIST 전처리 (train 전용)")
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

    boundaries = calc_grade_boundaries(n_files)
    strides    = {}
    for grade, (_, _, cnt) in boundaries.items():
        total_rows = cnt * KAIST_FILE_ROWS
        strides[grade] = (
            KAIST_WINDOW if grade == "LOW"
            else calc_stride_for_target(total_rows, KAIST_WINDOW, TARGET_PER_GRADE)
        )

    print(f"\n등급별 stride:")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if g in boundaries:
            _, _, cnt = boundaries[g]
            print(f"  {g:10s}: {cnt:4d}파일 | stride={strides[g]:>8,}")

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
            temp_resid = round(temp_val - (amb_val + 35.0), 4)

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
        rows   = grade_rows[g]
        before = len(rows)
        rows   = apply_cap_keep_order(rows, TARGET_PER_GRADE)
        print(f"  {g:10s}: {before:>7,} → {len(rows):>6,}개")
        final_rows.extend(rows)

    result = pd.DataFrame(final_rows)
    print_dist(result, "KAIST train")
    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. FEMTO 전처리 (train / test 분리)
# ══════════════════════════════════════════════════════════════

def preprocess_femto() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STEP 2. FEMTO 전처리 (train/test 분리)")
    print(f"  test 베어링: {TEST_BEARINGS}")
    print("=" * 60)

    train_grade_rows = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
    test_rows        = []

    for femto_dir in FEMTO_DIRS:
        if not femto_dir.exists():
            continue

        for bearing_dir in sorted(d for d in femto_dir.iterdir() if d.is_dir()):
            acc_files = sorted(bearing_dir.glob("acc_*.csv"))
            if not acc_files:
                continue

            n_files    = len(acc_files)
            is_test    = bearing_dir.name in TEST_BEARINGS
            tag        = "[ TEST ]" if is_test else "[ TRAIN]"
            print(f"  {tag} {bearing_dir.name:20s}: {n_files:5,}개 파일")

            boundaries = calc_grade_boundaries(n_files)
            strides    = {}
            for grade, (_, _, cnt) in boundaries.items():
                total_rows = cnt * FEMTO_FILE_ROWS
                strides[grade] = (
                    FEMTO_WINDOW if grade == "LOW"
                    else calc_stride_for_target(
                        total_rows, FEMTO_WINDOW, TARGET_PER_GRADE
                    )
                )

            rows = sliding_window_femto(
                bearing_dir, n_files,
                tqdm(acc_files, desc=f"    {bearing_dir.name}", leave=False),
                strides,
                is_test=is_test,
            )

            if is_test:
                # test: 원본 분포 그대로
                test_rows.extend(rows)
                dist = Counter(r["risk_level"] for r in rows)
                print(f"         → {len(rows):,}개 | "
                      f"LOW={dist['LOW']:5,} MED={dist['MEDIUM']:4,} "
                      f"HIGH={dist['HIGH']:4,} CRIT={dist['CRITICAL']:4,}")
            else:
                # train: 등급별로 분류
                for r in rows:
                    train_grade_rows[r["risk_level"]].append(r)

    # train 캡 적용
    print(f"\nFEMTO train 등급별 캡 (목표: {TARGET_PER_GRADE:,}개/등급):")
    train_rows = []
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        rows   = train_grade_rows[g]
        before = len(rows)
        rows   = apply_cap_keep_order(rows, TARGET_PER_GRADE)
        print(f"  {g:10s}: {before:>7,} → {len(rows):>6,}개")
        train_rows.extend(rows)

    femto_train = pd.DataFrame(train_rows)
    femto_test  = pd.DataFrame(test_rows)

    print_dist(femto_train, "FEMTO train")
    print_dist(femto_test,  "FEMTO test (원본 분포)")

    return femto_train, femto_test


# ══════════════════════════════════════════════════════════════
# STEP 3. 정규화 + 합산
# ══════════════════════════════════════════════════════════════

def normalize_and_combine(
    kaist_df    : pd.DataFrame,
    femto_train : pd.DataFrame,
    femto_test  : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:

    print("\n" + "=" * 60)
    print("STEP 3. 정규화 + 합산")
    print("=" * 60)

    # KAIST 기준 fit
    scaler = StandardScaler()
    scaler.fit(kaist_df[FEATURE_COLS])
    print("KAIST 기준 Z-score fit 완료")

    # train 정규화
    kaist_scaled       = kaist_df.copy()
    femto_train_scaled = femto_train.copy()
    femto_test_scaled  = femto_test.copy()

    kaist_scaled[FEATURE_COLS]       = scaler.transform(kaist_df[FEATURE_COLS])
    femto_train_scaled[FEATURE_COLS] = scaler.transform(femto_train[FEATURE_COLS])
    femto_test_scaled[FEATURE_COLS]  = scaler.transform(femto_test[FEATURE_COLS])
    print("정규화 완료")

    # 합산
    combined_train = pd.concat(
        [kaist_scaled, femto_train_scaled], ignore_index=True
    )
    combined_test = femto_test_scaled.copy()

    print_dist(combined_train, "combined_train (학습용)")
    print_dist(combined_test,  "combined_test  (검증용, 원본 분포)")

    return combined_train, combined_test, scaler


# ══════════════════════════════════════════════════════════════
# STEP 4. 저장
# ══════════════════════════════════════════════════════════════

def save_outputs(
    kaist_df      : pd.DataFrame,
    femto_train   : pd.DataFrame,
    femto_test    : pd.DataFrame,
    combined_train: pd.DataFrame,
    combined_test : pd.DataFrame,
    scaler        : StandardScaler,
):
    print("\n" + "=" * 60)
    print("STEP 4. 저장")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    files = {
        "kaist_preprocessed.csv" : kaist_df,
        "femto_train.csv"        : femto_train,
        "femto_test.csv"         : femto_test,
        "combined_train.csv"     : combined_train,
        "combined_test.csv"      : combined_test,
    }
    for fname, df in files.items():
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {fname:35s}: {len(df):>8,}행")

    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    print(f"  {'scaler.pkl':35s}: 저장 완료")

    # 요약 XLSX
    def make_row(label, df):
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
        make_row("KAIST (train)",       kaist_df),
        make_row("FEMTO train",         femto_train),
        make_row("FEMTO test",          femto_test),
        make_row("combined_train",      combined_train),
        make_row("combined_test",       combined_test),
    ])

    # test 베어링 상세
    test_detail = (
        femto_test.groupby("file_name")
        .agg(
            총샘플수  = ("vibration_rms", "count"),
            rms_min   = ("vibration_rms", "min"),
            rms_max   = ("vibration_rms", "max"),
            rul_min   = ("rul_days",      "min"),
            rul_max   = ("rul_days",      "max"),
            LOW       = ("risk_level",    lambda x: (x=="LOW").sum()),
            MEDIUM    = ("risk_level",    lambda x: (x=="MEDIUM").sum()),
            HIGH      = ("risk_level",    lambda x: (x=="HIGH").sum()),
            CRITICAL  = ("risk_level",    lambda x: (x=="CRITICAL").sum()),
        )
        .reset_index()
    )

    scaler_info = pd.DataFrame({
        "feature" : FEATURE_COLS,
        "mean"    : scaler.mean_,
        "std"     : scaler.scale_,
    })

    test_bearing_note = pd.DataFrame([{
        "test_bearings"    : str(TEST_BEARINGS),
        "목적"             : "학습에 한 번도 사용하지 않은 베어링으로 검증",
        "검증 포인트"      : "LOW→CRITICAL 전환 패턴을 처음 보는 데이터로 평가",
        "원본 분포 유지"   : "캡/언더샘플링 없음 (실제 운영 환경과 동일)",
    }])

    xlsx_path = OUTPUT_DIR / "preprocess_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(          writer, sheet_name="최종요약",        index=False)
        test_detail.to_excel(      writer, sheet_name="TEST베어링상세",  index=False)
        test_bearing_note.to_excel(writer, sheet_name="TEST베어링설명",  index=False)
        scaler_info.to_excel(      writer, sheet_name="Scaler파라미터",  index=False)

    print(f"\n  preprocess_summary.xlsx: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    total_start = time.time()

    kaist_df              = preprocess_kaist()
    femto_train, femto_test = preprocess_femto()
    combined_train, combined_test, scaler = normalize_and_combine(
        kaist_df, femto_train, femto_test
    )
    save_outputs(
        kaist_df, femto_train, femto_test,
        combined_train, combined_test, scaler
    )

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")