"""
AI-Pass 예지보전 전처리 파이프라인 v1.0
========================================
목적  : LSTM 학습용 시계열 데이터 생성
        - 시계열 순서 유지
        - 등급별 stride 차등 적용 → 25:25:25:25 분포 달성
        - KAIST / FEMTO 별도 전처리 후 합산

출력  :
  D:/project/예지보전/output/
    ├── kaist_preprocessed.csv   (KAIST 전처리 결과)
    ├── femto_preprocessed.csv   (FEMTO 전처리 결과)
    ├── combined_train.csv        (합산 학습용)
    ├── scaler.pkl                (운영 시 동일 적용)
    └── preprocess_summary.xlsx  (요약 정보)

실행  : python preprocess_pipeline.py
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

KAIST_DIR   = Path(r"D:\project\데이터셋\Vibration_Bearing_RuntoFailure")
FEMTO_ROOT  = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")
OUTPUT_DIR  = Path(r"D:\project\예지보전\output")

BASE_LIFE_DAYS   = 180    # 기준 수명 (일)
KAIST_WINDOW     = 25600  # KAIST 윈도우 (1초 @ 25,600Hz)
FEMTO_WINDOW     = 256    # FEMTO 윈도우 (0.1초 @ 2,560Hz)
KAIST_FILE_ROWS  = 2_000_000
FEMTO_FILE_ROWS  = 2560

# FEMTO 사용 폴더
FEMTO_DIRS = [
    FEMTO_ROOT / "Learning_set",
    FEMTO_ROOT / "Full_Test_Set",
]

# 위험도 등급 임계값
RUL_THRESHOLDS = {
    "LOW"     : 31,
    "MEDIUM"  : 16,
    "HIGH"    :  3,
    "CRITICAL":  0,
}

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
    if m in [3, 4, 5]:   return 0  # 봄
    if m in [6, 7, 8]:   return 1  # 여름
    if m in [9, 10, 11]: return 2  # 가을
    return 3                        # 겨울


def generate_motor_current(rms: float) -> float:
    """모터 전류 더미 생성 (진동 상관관계 기반)"""
    return round(max(0.5, 0.8 + 0.3 * rms + np.random.normal(0, 0.05)), 4)


def calc_grade_boundaries(n_files: int) -> dict:
    """
    파일 수 기준으로 등급별 index 범위 계산
    반환: {등급: (start_idx, end_idx, file_count)}
    """
    grade_indices = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
    for i in range(n_files):
        rul = BASE_LIFE_DAYS * (n_files - 1 - i) / (n_files - 1)
        grade_indices[rul_to_risk(rul)].append(i)

    result = {}
    for grade, indices in grade_indices.items():
        if indices:
            result[grade] = (min(indices), max(indices), len(indices))
    return result


def calc_stride_per_grade(
    boundaries: dict,
    file_rows: int,
    window: int,
) -> dict:
    """
    CRITICAL 자연샘플 수를 기준으로 각 등급의 stride 계산
    반환: {등급: stride}
    """
    # CRITICAL 자연 샘플 수 계산
    if "CRITICAL" not in boundaries:
        return {g: window for g in boundaries}

    crit_file_count = boundaries["CRITICAL"][2]
    crit_total_rows = crit_file_count * file_rows
    crit_natural    = max(1, (crit_total_rows - window) // window + 1)

    strides = {}
    for grade, (_, _, cnt) in boundaries.items():
        total_rows    = cnt * file_rows
        needed_stride = max(1, (total_rows - window) // crit_natural)
        strides[grade] = needed_stride

    return strides


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_kaist() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1. KAIST 전처리")
    print("=" * 60)

    all_files = sorted(KAIST_DIR.glob("LogFile_*.csv"))
    n_files   = len(all_files)
    print(f"총 파일 수: {n_files}개")

    # 파일별 타임스탬프 + RUL 계산
    file_info = []
    for f in all_files:
        match = re.search(
            r"LogFile_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", f.name
        )
        if match:
            ts  = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
            file_info.append((f, ts))

    file_info.sort(key=lambda x: x[1])
    n_files = len(file_info)

    # 등급별 경계 + stride 계산
    boundaries = calc_grade_boundaries(n_files)
    strides    = calc_stride_per_grade(boundaries, KAIST_FILE_ROWS, KAIST_WINDOW)

    print(f"\n등급별 파일 수 및 stride:")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if g in boundaries:
            _, _, cnt = boundaries[g]
            print(f"  {g:10s}: {cnt:4d}파일 | stride={strides[g]:>8,}")

    rows = []
    for file_idx, (f, ts) in enumerate(
        tqdm(file_info, desc="KAIST", unit="파일")
    ):
        rul   = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
        grade = rul_to_risk(rul)
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
        op_hours = file_idx * 1.0  # 파일 1개 = 1시간

        # 윈도우 슬라이딩
        total_r = len(df)
        start   = 0
        while start + KAIST_WINDOW <= total_r:
            end   = start + KAIST_WINDOW
            rms_h = calc_rms(vib_h[start:end])
            rms_v = calc_rms(vib_v[start:end])
            rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

            # temp_residual (ambient + 평균발열 35도 기준)
            avg_heat     = 35.0
            expected_tmp = amb_val + avg_heat
            temp_resid   = round(temp_val - expected_tmp, 4)

            rows.append({
                "source"        : "kaist",
                "file_name"     : f.name,
                "timestamp"     : ts,
                "window_start"  : start,
                "vibration_rms" : rms,
                "temperature"   : round(temp_val, 2),
                "temp_residual" : temp_resid,
                "motor_current" : generate_motor_current(rms),
                "operating_hours": round(op_hours, 2),
                "ambient_temp"  : round(amb_val, 2),
                "wind_speed"    : round(np.random.uniform(1.0, 5.0), 2),  # 더미
                "humidity"      : round(np.random.uniform(30.0, 80.0), 1),  # 더미
                "season"        : season,
                "rul_days"      : rul,
                "risk_level"    : grade,
            })
            start += stride

    result = pd.DataFrame(rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"\nKAIST 전처리 완료: {total:,}개 샘플")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. FEMTO 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_femto() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2. FEMTO 전처리")
    print("=" * 60)

    all_rows = []

    for femto_dir in FEMTO_DIRS:
        if not femto_dir.exists():
            print(f"  [SKIP] {femto_dir} 없음")
            continue

        bearing_dirs = sorted([d for d in femto_dir.iterdir() if d.is_dir()])

        for bearing_dir in bearing_dirs:
            acc_files = sorted(bearing_dir.glob("acc_*.csv"))
            if not acc_files:
                continue

            n_files    = len(acc_files)
            boundaries = calc_grade_boundaries(n_files)
            strides    = calc_stride_per_grade(
                boundaries, FEMTO_FILE_ROWS, FEMTO_WINDOW
            )

            # 파일별 데이터 로딩
            file_data = []
            for f in tqdm(
                acc_files,
                desc=f"  {bearing_dir.name}",
                unit="파일",
                leave=False,
            ):
                try:
                    df = pd.read_csv(
                        f, header=None, sep=",",
                        names=["hour","min","sec","usec","h_acc","v_acc"]
                    )
                    df = df.apply(pd.to_numeric, errors="coerce")
                    file_data.append(df)
                except Exception as e:
                    file_data.append(None)

            # 윈도우 슬라이딩
            bearing_rows = []
            for file_idx, df in enumerate(file_data):
                if df is None:
                    continue

                rul    = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
                grade  = rul_to_risk(rul)
                stride = strides.get(grade, FEMTO_WINDOW)

                h_acc  = df["h_acc"].values
                v_acc  = df["v_acc"].values
                total_r = len(df)
                op_hours = file_idx / 3600.0  # 초 단위 → 시간

                start = 0
                while start + FEMTO_WINDOW <= total_r:
                    end   = start + FEMTO_WINDOW
                    rms_h = calc_rms(h_acc[start:end])
                    rms_v = calc_rms(v_acc[start:end])
                    rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

                    # FEMTO는 온도/기상 없음 → 더미 생성
                    temp_val     = round(25.0 + rms * 5.0 + np.random.normal(0, 1), 2)
                    amb_val      = round(np.random.uniform(15.0, 30.0), 2)
                    avg_heat     = 35.0
                    expected_tmp = amb_val + avg_heat
                    temp_resid   = round(temp_val - expected_tmp, 4)

                    bearing_rows.append({
                        "source"         : f"femto_{femto_dir.name}",
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

            dist = Counter(r["risk_level"] for r in bearing_rows)
            total = len(bearing_rows)
            print(f"  {bearing_dir.name:20s}: {total:6,}개 | "
                  f"LOW={dist['LOW']:5,} MED={dist['MEDIUM']:4,} "
                  f"HIGH={dist['HIGH']:4,} CRIT={dist['CRITICAL']:4,}")
            all_rows.extend(bearing_rows)

    result = pd.DataFrame(all_rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"\nFEMTO 전처리 완료: {total:,}개 샘플")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 3. 정규화 + 합산
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "vibration_rms", "temperature", "temp_residual",
    "motor_current", "operating_hours", "ambient_temp",
    "wind_speed", "humidity", "season",
]

def normalize_and_combine(
    kaist_df: pd.DataFrame,
    femto_df: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:

    print("\n" + "=" * 60)
    print("STEP 3. 정규화 + 합산")
    print("=" * 60)

    # KAIST 기준으로 scaler fit
    scaler = StandardScaler()
    scaler.fit(kaist_df[FEATURE_COLS])
    print("KAIST 기준 Z-score scaler fit 완료")

    # 각각 transform
    kaist_scaled = kaist_df.copy()
    femto_scaled = femto_df.copy()

    kaist_scaled[FEATURE_COLS] = scaler.transform(kaist_df[FEATURE_COLS])
    femto_scaled[FEATURE_COLS] = scaler.transform(femto_df[FEATURE_COLS])
    print("KAIST / FEMTO 정규화 완료")

    # 합산
    combined = pd.concat([kaist_scaled, femto_scaled], ignore_index=True)
    dist     = Counter(combined["risk_level"])
    total    = len(combined)

    print(f"\n최종 합산: {total:,}개 샘플")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")

    return combined, scaler


# ══════════════════════════════════════════════════════════════
# STEP 4. 저장
# ══════════════════════════════════════════════════════════════

def save_outputs(
    kaist_df : pd.DataFrame,
    femto_df : pd.DataFrame,
    combined : pd.DataFrame,
    scaler   : StandardScaler,
):
    print("\n" + "=" * 60)
    print("STEP 4. 저장")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    kaist_path    = OUTPUT_DIR / "kaist_preprocessed.csv"
    femto_path    = OUTPUT_DIR / "femto_preprocessed.csv"
    combined_path = OUTPUT_DIR / "combined_train.csv"
    scaler_path   = OUTPUT_DIR / "scaler.pkl"

    kaist_df.to_csv(kaist_path,    index=False, encoding="utf-8-sig")
    femto_df.to_csv(femto_path,    index=False, encoding="utf-8-sig")
    combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
    joblib.dump(scaler, scaler_path)

    print(f"  kaist_preprocessed.csv : {len(kaist_df):>8,}행")
    print(f"  femto_preprocessed.csv : {len(femto_df):>8,}행")
    print(f"  combined_train.csv     : {len(combined):>8,}행")
    print(f"  scaler.pkl             : 저장 완료")

    # 요약 XLSX
    xlsx_path = OUTPUT_DIR / "preprocess_summary.xlsx"

    # 시트1: 최종요약
    k_dist = Counter(kaist_df["risk_level"])
    f_dist = Counter(femto_df["risk_level"])
    c_dist = Counter(combined["risk_level"])
    c_total = len(combined)

    summary = pd.DataFrame([
        {
            "구분"          : "KAIST",
            "총샘플수"      : len(kaist_df),
            "LOW"           : k_dist["LOW"],
            "MEDIUM"        : k_dist["MEDIUM"],
            "HIGH"          : k_dist["HIGH"],
            "CRITICAL"      : k_dist["CRITICAL"],
            "LOW_%"         : round(k_dist["LOW"]      / len(kaist_df) * 100, 1),
            "MEDIUM_%"      : round(k_dist["MEDIUM"]   / len(kaist_df) * 100, 1),
            "HIGH_%"        : round(k_dist["HIGH"]     / len(kaist_df) * 100, 1),
            "CRITICAL_%"    : round(k_dist["CRITICAL"] / len(kaist_df) * 100, 1),
        },
        {
            "구분"          : "FEMTO",
            "총샘플수"      : len(femto_df),
            "LOW"           : f_dist["LOW"],
            "MEDIUM"        : f_dist["MEDIUM"],
            "HIGH"          : f_dist["HIGH"],
            "CRITICAL"      : f_dist["CRITICAL"],
            "LOW_%"         : round(f_dist["LOW"]      / len(femto_df) * 100, 1),
            "MEDIUM_%"      : round(f_dist["MEDIUM"]   / len(femto_df) * 100, 1),
            "HIGH_%"        : round(f_dist["HIGH"]     / len(femto_df) * 100, 1),
            "CRITICAL_%"    : round(f_dist["CRITICAL"] / len(femto_df) * 100, 1),
        },
        {
            "구분"          : "합산(combined)",
            "총샘플수"      : c_total,
            "LOW"           : c_dist["LOW"],
            "MEDIUM"        : c_dist["MEDIUM"],
            "HIGH"          : c_dist["HIGH"],
            "CRITICAL"      : c_dist["CRITICAL"],
            "LOW_%"         : round(c_dist["LOW"]      / c_total * 100, 1),
            "MEDIUM_%"      : round(c_dist["MEDIUM"]   / c_total * 100, 1),
            "HIGH_%"        : round(c_dist["HIGH"]     / c_total * 100, 1),
            "CRITICAL_%"    : round(c_dist["CRITICAL"] / c_total * 100, 1),
        },
    ])

    # 시트2: KAIST 파일 단위 요약 (129행)
    kaist_file_summary = (
        kaist_df.groupby("file_name")
        .agg(
            timestamp       = ("timestamp",     "first"),
            vibration_rms   = ("vibration_rms", "mean"),
            temperature     = ("temperature",   "mean"),
            rul_days        = ("rul_days",       "first"),
            risk_level      = ("risk_level",     "first"),
            window_count    = ("window_start",   "count"),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    # 시트3: FEMTO 베어링 요약 (17행)
    femto_bearing_summary = (
        femto_df.groupby("file_name")
        .agg(
            source          = ("source",        "first"),
            총샘플수        = ("vibration_rms", "count"),
            rms_min         = ("vibration_rms", "min"),
            rms_max         = ("vibration_rms", "max"),
            rms_mean        = ("vibration_rms", "mean"),
            LOW             = ("risk_level",    lambda x: (x=="LOW").sum()),
            MEDIUM          = ("risk_level",    lambda x: (x=="MEDIUM").sum()),
            HIGH            = ("risk_level",    lambda x: (x=="HIGH").sum()),
            CRITICAL        = ("risk_level",    lambda x: (x=="CRITICAL").sum()),
        )
        .reset_index()
    )

    # 시트4: scaler 파라미터
    scaler_info = pd.DataFrame({
        "feature" : FEATURE_COLS,
        "mean"    : scaler.mean_,
        "std"     : scaler.scale_,
    })

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(              writer, sheet_name="최종요약",        index=False)
        kaist_file_summary.to_excel(   writer, sheet_name="KAIST_파일단위",  index=False)
        femto_bearing_summary.to_excel(writer, sheet_name="FEMTO_베어링요약",index=False)
        scaler_info.to_excel(          writer, sheet_name="Scaler_파라미터", index=False)

    print(f"  preprocess_summary.xlsx: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    total_start = time.time()

    # STEP 1. KAIST
    kaist_df = preprocess_kaist()

    # STEP 2. FEMTO
    femto_df = preprocess_femto()

    # STEP 3. 정규화 + 합산
    combined, scaler = normalize_and_combine(kaist_df, femto_df)

    # STEP 4. 저장
    save_outputs(kaist_df, femto_df, combined, scaler)

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")