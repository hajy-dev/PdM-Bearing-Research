"""
KAIST + FEMTO 전처리 파이프라인
-------------------------------
목적  : 윈도우 슬라이딩 설정별 샘플 수 / 등급 분포 확인 후 최종 설정 선택
출력  : preprocess_result.xlsx (설정별 비교 + 최종 데이터)

실행  : python preprocess_kaist_femto.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import time

# ══════════════════════════════════════════════════════════════
# 설정값 — 여기만 수정하면 됨
# ══════════════════════════════════════════════════════════════

KAIST_DIR   = Path(r"D:\project\데이터셋\Vibration_Bearing_RuntoFailure")
FEMTO_ROOT  = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")
OUTPUT_PATH = Path(r"D:\project\예지보전\preprocess_result.xlsx")

BASE_LIFE_DAYS = 180        # 기준 수명 (일) — PTZ 카메라 베어링 실무 교체 주기
ROWS_PER_FILE  = 2_000_000  # KAIST 파일당 행 수

# 윈도우 슬라이딩 설정 목록 — 직접 확인하고 싶은 설정 추가/제거 가능
WINDOW_CONFIGS = [
    {"window": 25600,  "stride": 12800, "label": "1초/0.5초"},
    {"window": 12800,  "stride":  6400, "label": "0.5초/0.25초"},
    {"window":  6400,  "stride":  3200, "label": "0.25초/0.125초"},
    {"window":  2560,  "stride":  1280, "label": "0.1초/0.05초"},
    {"window":  1280,  "stride":   640, "label": "0.05초/0.025초"},
]

# 최종 사용할 윈도우 설정 (위 목록 중 하나 선택)
FINAL_WINDOW = 2560
FINAL_STRIDE = 1280

# FEMTO 사용 폴더 (Learning_set / Full_Test_Set 중 선택)
FEMTO_DIRS = [
    FEMTO_ROOT / "Learning_set",
    FEMTO_ROOT / "Full_Test_Set",
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


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 파일 단위 집계
# ══════════════════════════════════════════════════════════════

def load_kaist_files() -> pd.DataFrame:
    import re
    from datetime import datetime

    print("\n" + "=" * 60)
    print("STEP 1. KAIST 파일 단위 집계")
    print("=" * 60)

    all_files = sorted(KAIST_DIR.glob("LogFile_*.csv"))
    print(f"총 파일 수: {len(all_files)}개")

    rows = []
    for f in tqdm(all_files, desc="KAIST", unit="파일"):
        match = re.search(
            r"LogFile_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", f.name
        )
        if not match:
            continue
        from datetime import datetime
        ts = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")
        try:
            df = pd.read_csv(
                f, header=None, sep=",",
                names=["vib_h", "vib_v", "temp", "ambient_temp"]
            )
            df = df.apply(pd.to_numeric, errors="coerce")
            rms_h = calc_rms(df["vib_h"].values)
            rms_v = calc_rms(df["vib_v"].values)
            rms   = np.sqrt(rms_h**2 + rms_v**2)
            rows.append({
                "source"       : "kaist",
                "file_name"    : f.name,
                "timestamp"    : ts,
                "vibration_rms": round(rms, 6),
                "temp"         : round(df["temp"].iloc[0], 2),
                "ambient_temp" : round(df["ambient_temp"].iloc[0], 2),
                "total_rows"   : len(df),
            })
        except Exception as e:
            tqdm.write(f"[WARN] {f.name}: {e}")

    result = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    n = len(result)

    # RUL 선형 라벨 부여 (180일 기준)
    result["rul_days"]   = [
        round(BASE_LIFE_DAYS * (n-1-i) / (n-1), 4) for i in range(n)
    ]
    result["risk_level"] = result["rul_days"].apply(rul_to_risk)

    print(f"KAIST 집계 완료: {n}개 파일")
    print(f"  vibration_rms: {result['vibration_rms'].min():.4f} ~ {result['vibration_rms'].max():.4f}g")
    print(f"  temp         : {result['temp'].min():.1f} ~ {result['temp'].max():.1f}°C")
    print(f"  rul_days     : {result['rul_days'].min():.2f} ~ {result['rul_days'].max():.2f}일")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. 윈도우 슬라이딩 설정별 샘플 수 비교 (시뮬레이션)
# ══════════════════════════════════════════════════════════════

def simulate_window_configs(kaist_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2. 윈도우 슬라이딩 설정별 시뮬레이션")
    print("=" * 60)

    kaist_risk_list = kaist_df["risk_level"].tolist()
    femto_total     = sum(
        len(list(d.glob("*/acc_*.csv"))) for d in FEMTO_DIRS if d.exists()
    )
    femto_dist = _get_femto_dist_simulation(femto_total)

    rows = []
    for cfg in WINDOW_CONFIGS:
        ws, st = cfg["window"], cfg["stride"]
        wpf    = (ROWS_PER_FILE - ws) // st + 1
        kaist_total = len(kaist_df) * wpf

        # KAIST 등급 분포
        kaist_risk_expanded = []
        for r in kaist_risk_list:
            kaist_risk_expanded.extend([r] * wpf)
        k_dist = Counter(kaist_risk_expanded)

        grand_total = kaist_total + femto_total
        total_dist  = {
            g: k_dist[g] + femto_dist.get(g, 0)
            for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        }

        mem_mb = grand_total * 9 * 4 / 1024**2

        rows.append({
            "설정"              : cfg["label"],
            "window_size"       : ws,
            "stride"            : st,
            "파일당_window수"   : wpf,
            "KAIST_샘플수"      : kaist_total,
            "FEMTO_샘플수"      : femto_total,
            "총_샘플수"         : grand_total,
            "LOW"               : total_dist["LOW"],
            "MEDIUM"            : total_dist["MEDIUM"],
            "HIGH"              : total_dist["HIGH"],
            "CRITICAL"          : total_dist["CRITICAL"],
            "LOW_%"             : round(total_dist["LOW"]      / grand_total * 100, 1),
            "MEDIUM_%"          : round(total_dist["MEDIUM"]   / grand_total * 100, 1),
            "HIGH_%"            : round(total_dist["HIGH"]     / grand_total * 100, 1),
            "CRITICAL_%"        : round(total_dist["CRITICAL"] / grand_total * 100, 1),
            "메모리_MB"         : round(mem_mb, 1),
        })

        print(f"  [{cfg['label']:15s}] 파일당 {wpf:5,}개 | 총 {grand_total:>9,}개 | "
              f"LOW {total_dist['LOW']:>7,} MED {total_dist['MEDIUM']:>6,} "
              f"HIGH {total_dist['HIGH']:>6,} CRIT {total_dist['CRITICAL']:>5,} | "
              f"{mem_mb:.1f}MB")

    return pd.DataFrame(rows)


def _get_femto_dist_simulation(femto_total: int) -> dict:
    """FEMTO 베어링별 등급 분포를 합산 (파일 수 기반 시뮬레이션)"""
    femto_bearings = {}
    for femto_dir in FEMTO_DIRS:
        if not femto_dir.exists():
            continue
        for bearing_dir in sorted(femto_dir.iterdir()):
            if bearing_dir.is_dir():
                n = len(list(bearing_dir.glob("acc_*.csv")))
                if n > 0:
                    femto_bearings[bearing_dir.name] = n

    dist = Counter()
    for name, n_files in femto_bearings.items():
        for i in range(n_files):
            rul = BASE_LIFE_DAYS * (n_files - 1 - i) / (n_files - 1)
            dist[rul_to_risk(rul)] += 1
    return dict(dist)


# ══════════════════════════════════════════════════════════════
# STEP 3. 최종 설정으로 KAIST 윈도우 슬라이딩 실행
# ══════════════════════════════════════════════════════════════

def apply_window_sliding(kaist_df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print(f"STEP 3. KAIST 윈도우 슬라이딩 (window={FINAL_WINDOW:,} / stride={FINAL_STRIDE:,})")
    print("=" * 60)

    import re
    from datetime import datetime

    all_files = sorted(KAIST_DIR.glob("LogFile_*.csv"))
    n_files   = len(kaist_df)
    rows      = []

    for file_idx, f in enumerate(tqdm(all_files, desc="슬라이딩", unit="파일")):
        match = re.search(
            r"LogFile_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", f.name
        )
        if not match:
            continue
        ts = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S")

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
        temp_val = df["temp"].iloc[0]
        amb_val  = df["ambient_temp"].iloc[0]
        total_r  = len(df)

        # 파일의 RUL (180일 기준 선형)
        rul = round(BASE_LIFE_DAYS * (n_files - 1 - file_idx) / (n_files - 1), 4)

        # 윈도우 슬라이딩
        start = 0
        while start + FINAL_WINDOW <= total_r:
            end   = start + FINAL_WINDOW
            w_h   = vib_h[start:end]
            w_v   = vib_v[start:end]
            rms_h = calc_rms(w_h)
            rms_v = calc_rms(w_v)
            rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)

            rows.append({
                "source"       : "kaist",
                "file_name"    : f.name,
                "timestamp"    : ts,
                "window_start" : start,
                "vibration_rms": rms,
                "temp"         : round(temp_val, 2),
                "ambient_temp" : round(amb_val, 2),
                "rul_days"     : rul,
                "risk_level"   : rul_to_risk(rul),
            })
            start += FINAL_STRIDE

    result = pd.DataFrame(rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"슬라이딩 완료: {total:,}개 샘플")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>8,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 4. FEMTO 전체 파일 단위 집계
# ══════════════════════════════════════════════════════════════

def load_femto_all() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4. FEMTO 전체 집계")
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

            n_files  = len(acc_files)
            rms_list = []

            for f in tqdm(acc_files, desc=f"  {bearing_dir.name}", unit="파일", leave=False):
                try:
                    df = pd.read_csv(
                        f, header=None, sep=",",
                        names=["hour","min","sec","usec","h_acc","v_acc"]
                    )
                    df = df.apply(pd.to_numeric, errors="coerce")
                    rms_h = calc_rms(df["h_acc"].values)
                    rms_v = calc_rms(df["v_acc"].values)
                    rms   = round(np.sqrt(rms_h**2 + rms_v**2), 6)
                    rms_list.append(rms)
                except Exception as e:
                    rms_list.append(np.nan)

            for i, rms in enumerate(rms_list):
                rul = round(BASE_LIFE_DAYS * (n_files - 1 - i) / (n_files - 1), 4)
                all_rows.append({
                    "source"       : f"femto_{femto_dir.name}",
                    "file_name"    : bearing_dir.name,
                    "timestamp"    : None,
                    "vibration_rms": rms,
                    "temp"         : np.nan,  # FEMTO 온도 없음
                    "ambient_temp" : np.nan,
                    "rul_days"     : rul,
                    "risk_level"   : rul_to_risk(rul),
                })

            dist = Counter(rul_to_risk(
                BASE_LIFE_DAYS * (n_files-1-i) / (n_files-1)
            ) for i in range(n_files))
            print(f"  {bearing_dir.name:20s}: {n_files:5,}개 | "
                  f"LOW={dist['LOW']:4d} MED={dist['MEDIUM']:3d} "
                  f"HIGH={dist['HIGH']:3d} CRIT={dist['CRITICAL']:3d}")

    result = pd.DataFrame(all_rows)
    dist   = Counter(result["risk_level"])
    total  = len(result)

    print(f"\nFEMTO 전체: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")

    return result


# ══════════════════════════════════════════════════════════════
# STEP 5. 엑셀 저장
# ══════════════════════════════════════════════════════════════

def save_excel(
    sim_df     : pd.DataFrame,
    kaist_file : pd.DataFrame,
    kaist_slide: pd.DataFrame,
    femto_df   : pd.DataFrame,
):
    print("\n" + "=" * 60)
    print("STEP 5. 엑셀 저장")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 최종 합산
    combined = pd.concat([kaist_slide, femto_df], ignore_index=True)
    dist     = Counter(combined["risk_level"])
    total    = len(combined)

    summary = pd.DataFrame([{
        "항목"          : "최종 합산",
        "KAIST_슬라이딩": len(kaist_slide),
        "FEMTO_전체"    : len(femto_df),
        "총_샘플수"     : total,
        "LOW"           : dist["LOW"],
        "MEDIUM"        : dist["MEDIUM"],
        "HIGH"          : dist["HIGH"],
        "CRITICAL"      : dist["CRITICAL"],
        "LOW_%"         : round(dist["LOW"]      / total * 100, 1),
        "MEDIUM_%"      : round(dist["MEDIUM"]   / total * 100, 1),
        "HIGH_%"        : round(dist["HIGH"]     / total * 100, 1),
        "CRITICAL_%"    : round(dist["CRITICAL"] / total * 100, 1),
        "기준수명_days" : BASE_LIFE_DAYS,
        "window_size"   : FINAL_WINDOW,
        "stride"        : FINAL_STRIDE,
    }])

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        # 시트 1: 윈도우 설정별 비교
        sim_df.to_excel(writer, sheet_name="윈도우설정비교", index=False)

        # 시트 2: KAIST 파일 단위 집계
        kaist_file.to_excel(writer, sheet_name="KAIST_파일단위", index=False)

        # 시트 3: KAIST 슬라이딩 결과 (최대 100,000행 — 전체는 메모리 이슈)
        kaist_slide.head(100_000).to_excel(
            writer, sheet_name="KAIST_슬라이딩(샘플)", index=False
        )

        # 시트 4: FEMTO 집계
        femto_df.to_excel(writer, sheet_name="FEMTO_집계", index=False)

        # 시트 5: 최종 합산 요약
        summary.to_excel(writer, sheet_name="최종요약", index=False)

    print(f"저장 완료: {OUTPUT_PATH}")
    print(f"  시트 목록: 윈도우설정비교 / KAIST_파일단위 / KAIST_슬라이딩(샘플) / FEMTO_집계 / 최종요약")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    # STEP 1. KAIST 파일 단위
    kaist_file_df = load_kaist_files()

    # STEP 2. 윈도우 설정별 시뮬레이션
    sim_df = simulate_window_configs(kaist_file_df)

    # STEP 3. 최종 설정으로 슬라이딩
    kaist_slide_df = apply_window_sliding(kaist_file_df)

    # STEP 4. FEMTO 전체
    femto_df = load_femto_all()

    # STEP 5. 엑셀 저장
    save_excel(sim_df, kaist_file_df, kaist_slide_df, femto_df)

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")