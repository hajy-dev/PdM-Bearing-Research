"""
AI-Pass 예지보전 전처리 파이프라인 v5.0
========================================
핵심 설계:
  - KAIST: 200만행 전체 읽기 → 통계 피처 추출 → 샘플 1개
  - FEMTO: 2560행 전체 읽기 → 통계 피처 추출 → 샘플 1개
  - 기상청 과거 API: KAIST 날짜 기준 wind_speed, humidity 실측값
    API 키: 환경변수 WEATHER_API_KEY
  - MinMaxScaler (0~1) 정규화
  - rul_days → rul_norm (0~1)
  - test 베어링 완전 분리 (Bearing1_3, Bearing1_5)
  - SMOTE로 train 25:25:25:25 균등 분포

피처 (13개):
  진동: vibration_rms, vibration_kurtosis, vibration_crest,
        vibration_peak, vibration_skewness
  환경: temperature, temp_residual, motor_current,
        operating_hours, ambient_temp, wind_speed, humidity, season

실행:
  Windows: set WEATHER_API_KEY=your_key && python preprocess_pipeline_v5.py
  또는 .env 파일에 WEATHER_API_KEY=your_key 작성 후 실행

출력:
  D:/project/예지보전/output_v5/
    ├── kaist_preprocessed.csv
    ├── femto_train.csv
    ├── femto_test.csv
    ├── combined_train.csv       (SMOTE 적용, 균등 분포)
    ├── combined_test.csv        (원본 분포)
    ├── scaler.pkl
    ├── rul_scaler.pkl
    └── preprocess_summary.xlsx
"""

import os
import numpy as np
import pandas as pd
import joblib
import requests
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import kurtosis, skew
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import time
import re
from datetime import datetime, timedelta

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

KAIST_DIR  = Path(r"D:\project\데이터셋\Vibration_Bearing_RuntoFailure")
FEMTO_ROOT = Path(r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet")
OUTPUT_DIR = Path(r"D:\project\예지보전\output_v5")

BASE_LIFE_DAYS   = 180
TARGET_PER_GRADE = 8000

TEST_BEARINGS = {"Bearing1_3", "Bearing1_5"}

FEMTO_DIRS = [
    FEMTO_ROOT / "Learning_set",
    FEMTO_ROOT / "Full_Test_Set",
]

FEATURE_COLS = [
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "vibration_peak",
    "vibration_skewness",
    "temperature",
    "temp_residual",
    "motor_current",
    "operating_hours",
    "ambient_temp",
    "wind_speed",
    "humidity",
    "season",
]

AVG_HEAT = 35.0  # 장비 평균 발열값 (°C)

# ══════════════════════════════════════════════════════════════
# 환경변수에서 API 키 로딩
# ══════════════════════════════════════════════════════════════

def load_api_key() -> str:
    """
    환경변수 WEATHER_API_KEY에서 기상청 API 키 로딩
    .env 파일도 지원 (python-dotenv 설치 시)
    """
    # .env 파일 지원 (선택)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv 없어도 환경변수 직접 사용 가능

    api_key = os.environ.get("WEATHER_API_KEY", "")
    if not api_key:
        print("[WARN] WEATHER_API_KEY 환경변수가 없습니다.")
        print("       wind_speed, humidity는 더미로 대체됩니다.")
        print("       설정 방법: set WEATHER_API_KEY=your_key (Windows)")
    else:
        print(f"[OK] WEATHER_API_KEY 로딩 완료 ({api_key[:6]}...)")
    return api_key


# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def rul_to_risk(rul: float) -> str:
    if rul >= 31: return "LOW"
    if rul >= 16: return "MEDIUM"
    if rul >=  3: return "HIGH"
    return "CRITICAL"


def get_season(dt: datetime) -> int:
    m = dt.month
    if m in [3, 4, 5]:   return 0
    if m in [6, 7, 8]:   return 1
    if m in [9, 10, 11]: return 2
    return 3


def generate_motor_current(rms: float) -> float:
    return round(max(0.5, 0.8 + 0.3 * rms + np.random.normal(0, 0.05)), 4)


def calc_vibration_features(x: np.ndarray, y: np.ndarray) -> dict:
    """
    vib_x, vib_y 전체 신호에서 통계 피처 추출
    x, y 합성 신호 기준
    """
    # NaN 제거
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return {
            "vibration_rms"      : 0.0,
            "vibration_kurtosis" : 0.0,
            "vibration_crest"    : 0.0,
            "vibration_peak"     : 0.0,
            "vibration_skewness" : 0.0,
        }

    # 합성 신호 (x, y 합산 에너지 기준)
    rms_x = float(np.sqrt(np.mean(x**2)))
    rms_y = float(np.sqrt(np.mean(y**2)))
    rms   = float(np.sqrt(rms_x**2 + rms_y**2))

    # x, y 각각 통계 → 평균
    peak_x = float(np.max(np.abs(x)))
    peak_y = float(np.max(np.abs(y)))
    peak   = float(np.sqrt(peak_x**2 + peak_y**2))  # 합성 peak

    crest = peak / rms if rms > 0 else 0.0

    # kurtosis, skewness: x, y 평균
    kurt = float((kurtosis(x, fisher=False) + kurtosis(y, fisher=False)) / 2)
    skew_val = float((skew(x) + skew(y)) / 2)

    return {
        "vibration_rms"      : round(rms,      6),
        "vibration_kurtosis" : round(kurt,      4),
        "vibration_crest"    : round(crest,     4),
        "vibration_peak"     : round(peak,      6),
        "vibration_skewness" : round(skew_val,  4),
    }


def calc_grade_boundaries(n_files: int) -> dict:
    grade_idx = {g: [] for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]}
    for i in range(n_files):
        rul = BASE_LIFE_DAYS * (n_files-1-i) / (n_files-1)
        grade_idx[rul_to_risk(rul)].append(i)
    return {g: (min(v), max(v), len(v)) for g, v in grade_idx.items() if v}


def print_dist(df: pd.DataFrame, label: str):
    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"\n{label}: {total:,}개")
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# 기상청 과거 API
# ══════════════════════════════════════════════════════════════

def fetch_weather_kaist(api_key: str, timestamps: list) -> dict:
    """
    KAIST 날짜 기준 기상청 과거 API 호출
    날짜별 wind_speed, humidity 반환

    반환: {"20220620": {"wind_speed": 2.1, "humidity": 65.0}, ...}
    """
    if not api_key:
        return {}

    # KAIST 날짜 범위 추출 (중복 제거)
    dates = sorted(set(ts.strftime("%Y%m%d") for ts in timestamps if ts))
    weather_data = {}

    print(f"\n[기상청 API] {len(dates)}일치 과거 데이터 호출 중...")

    for date in tqdm(dates, desc="기상청 API", unit="일"):
        try:
            url = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
            params = {
                "serviceKey" : api_key,
                "pageNo"     : 1,
                "numOfRows"  : 24,
                "dataType"   : "JSON",
                "dataCd"     : "ASOS",
                "dateCd"     : "HR",
                "startDt"    : date,
                "startHh"    : "00",
                "endDt"      : date,
                "endHh"      : "23",
                "stnIds"     : "108",  # 서울 관측소
            }
            res = requests.get(url, params=params, timeout=10)
            data = res.json()

            items = (
                data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
            )

            if not items:
                continue

            # 일별 평균 계산
            wind_speeds = []
            humidities  = []
            for item in items:
                ws = item.get("ws")   # 풍속
                hm = item.get("hm")   # 습도
                if ws and ws != "":
                    try: wind_speeds.append(float(ws))
                    except: pass
                if hm and hm != "":
                    try: humidities.append(float(hm))
                    except: pass

            weather_data[date] = {
                "wind_speed" : round(np.mean(wind_speeds), 2) if wind_speeds else 3.0,
                "humidity"   : round(np.mean(humidities),  1) if humidities  else 60.0,
            }

        except Exception as e:
            tqdm.write(f"  [WARN] {date} API 호출 실패: {e}")
            weather_data[date] = {"wind_speed": 3.0, "humidity": 60.0}

    print(f"[기상청 API] {len(weather_data)}일 데이터 수집 완료")
    return weather_data


# ══════════════════════════════════════════════════════════════
# STEP 1. KAIST 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_kaist(api_key: str) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1. KAIST 전처리 (200만행 전체 → 통계 피처)")
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
    print(f"총 파일 수: {n_files}개 (파일당 약 200만행 전체 읽기)")

    # 기상청 API 호출
    timestamps   = [ts for _, ts in file_info]
    weather_data = fetch_weather_kaist(api_key, timestamps)

    rows = []
    for file_idx, (f, ts) in enumerate(
        tqdm(file_info, desc="KAIST", unit="파일")
    ):
        rul   = round(BASE_LIFE_DAYS * (n_files-1-file_idx) / (n_files-1), 4)
        grade = rul_to_risk(rul)

        try:
            df = pd.read_csv(
                f, header=None, sep=",",
                names=["vib_x", "vib_y", "temp", "ambient_temp"]
            )
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception as e:
            tqdm.write(f"[WARN] {f.name}: {e}")
            continue

        # 통계 피처 추출
        vib_feats = calc_vibration_features(
            df["vib_x"].values,
            df["vib_y"].values,
        )

        temp_val = round(float(df["temp"].mean(skipna=True)), 4)
        amb_val  = round(float(df["ambient_temp"].mean(skipna=True)), 4)
        temp_resid = round(temp_val - (amb_val + AVG_HEAT), 4)
        season     = get_season(ts)
        op_hours   = float(file_idx)

        # 기상청 API 데이터 (없으면 더미)
        date_key   = ts.strftime("%Y%m%d")
        w          = weather_data.get(date_key, {})
        wind_speed = w.get("wind_speed", round(np.random.uniform(1.0, 5.0), 2))
        humidity   = w.get("humidity",   round(np.random.uniform(30.0, 80.0), 1))

        row = {
            "source"          : "kaist",
            "file_name"       : f.name,
            "timestamp"       : ts,
            **vib_feats,
            "temperature"     : temp_val,
            "temp_residual"   : temp_resid,
            "motor_current"   : generate_motor_current(vib_feats["vibration_rms"]),
            "operating_hours" : op_hours,
            "ambient_temp"    : amb_val,
            "wind_speed"      : wind_speed,
            "humidity"        : humidity,
            "season"          : season,
            "rul_days"        : rul,
            "rul_norm"        : 0.0,
            "risk_level"      : grade,
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    print_dist(result, "KAIST")
    print(f"\n  vibration_rms     : {result['vibration_rms'].min():.4f} ~ {result['vibration_rms'].max():.4f}g")
    print(f"  vibration_kurtosis: {result['vibration_kurtosis'].min():.2f} ~ {result['vibration_kurtosis'].max():.2f}")
    print(f"  vibration_crest   : {result['vibration_crest'].min():.4f} ~ {result['vibration_crest'].max():.4f}")
    print(f"  temperature       : {result['temperature'].min():.2f} ~ {result['temperature'].max():.2f}°C")
    print(f"  wind_speed        : {result['wind_speed'].min():.2f} ~ {result['wind_speed'].max():.2f}m/s")
    print(f"  humidity          : {result['humidity'].min():.1f} ~ {result['humidity'].max():.1f}%")
    return result


# ══════════════════════════════════════════════════════════════
# STEP 2. FEMTO 전처리
# ══════════════════════════════════════════════════════════════

def preprocess_femto() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 60)
    print("STEP 2. FEMTO 전처리 (2560행 전체 → 통계 피처)")
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

                # 통계 피처 추출
                vib_feats = calc_vibration_features(
                    df["h_acc"].values,
                    df["v_acc"].values,
                )

                rms      = vib_feats["vibration_rms"]
                op_hours = file_idx / 3600.0

                # FEMTO 온도/기상 없음 → 더미
                temp_val   = round(25.0 + rms * 5.0 + np.random.normal(0, 1), 2)
                amb_val    = round(np.random.uniform(15.0, 30.0), 2)
                temp_resid = round(temp_val - (amb_val + AVG_HEAT), 4)
                wind_speed = round(np.random.uniform(1.0, 5.0), 2)
                humidity   = round(np.random.uniform(30.0, 80.0), 1)

                bearing_rows.append({
                    "source"          : "femto",
                    "file_name"       : bearing_dir.name,
                    "timestamp"       : None,
                    **vib_feats,
                    "temperature"     : temp_val,
                    "temp_residual"   : temp_resid,
                    "motor_current"   : generate_motor_current(rms),
                    "operating_hours" : round(op_hours, 4),
                    "ambient_temp"    : amb_val,
                    "wind_speed"      : wind_speed,
                    "humidity"        : humidity,
                    "season"          : int(np.random.randint(0, 4)),
                    "rul_days"        : rul,
                    "rul_norm"        : 0.0,
                    "risk_level"      : grade,
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
# STEP 3. 정규화
# ══════════════════════════════════════════════════════════════

def normalize(
    kaist_df    : pd.DataFrame,
    femto_train : pd.DataFrame,
    femto_test  : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler, MinMaxScaler]:

    print("\n" + "=" * 60)
    print("STEP 3. MinMaxScaler 정규화 (0~1)")
    print("=" * 60)

    train_all = pd.concat([kaist_df, femto_train], ignore_index=True)

    # 피처 scaler
    feat_scaler = MinMaxScaler()
    feat_scaler.fit(train_all[FEATURE_COLS])

    # RUL scaler
    rul_scaler = MinMaxScaler()
    rul_scaler.fit([[0], [BASE_LIFE_DAYS]])

    print(f"  피처 scaler fit: train {len(train_all):,}개 기준")
    print(f"  RUL scaler    : 0 ~ {BASE_LIFE_DAYS}일 → 0 ~ 1")

    def apply_scale(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[FEATURE_COLS] = feat_scaler.transform(df[FEATURE_COLS])
        out["rul_norm"]   = rul_scaler.transform(
            df["rul_days"].values.reshape(-1, 1)
        ).flatten().round(6)
        return out

    kaist_scaled       = apply_scale(kaist_df)
    femto_train_scaled = apply_scale(femto_train)
    femto_test_scaled  = apply_scale(femto_test)

    combined_pre  = pd.concat(
        [kaist_scaled, femto_train_scaled], ignore_index=True
    )
    combined_test = femto_test_scaled.copy()

    print_dist(combined_pre,  "정규화 후 train (SMOTE 전)")
    print_dist(combined_test, "정규화 후 test (원본 분포)")

    return combined_pre, combined_test, feat_scaler, rul_scaler


# ══════════════════════════════════════════════════════════════
# STEP 4. SMOTE
# ══════════════════════════════════════════════════════════════

def apply_smote(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 4. SMOTE (train 균등 분포)")
    print("=" * 60)

    dist  = Counter(df["risk_level"])
    total = len(df)
    print(f"SMOTE 전: {total:,}개")
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")

    X = df[FEATURE_COLS].values
    y = df["risk_level"].values

    sampling = {
        g: TARGET_PER_GRADE
        for g in ["LOW","MEDIUM","HIGH","CRITICAL"]
        if dist[g] < TARGET_PER_GRADE
    }
    # 이미 TARGET 초과한 등급은 언더샘플링
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        if dist[g] > TARGET_PER_GRADE and g not in sampling:
            sampling[g] = TARGET_PER_GRADE

    smote = SMOTE(
        sampling_strategy={
            g: TARGET_PER_GRADE
            for g in ["LOW","MEDIUM","HIGH","CRITICAL"]
            if dist[g] < TARGET_PER_GRADE
        },
        random_state=42,
        k_neighbors=min(5, min(dist.values()) - 1),
    )

    X_res, y_res = smote.fit_resample(X, y)

    # 오버샘플된 등급은 언더샘플링
    result_df = pd.DataFrame(X_res, columns=FEATURE_COLS)
    result_df["risk_level"] = y_res
    result_df["rul_norm"]   = 0.0

    # rul_norm 재부여 (등급 중간값 기준)
    rul_mid = {"LOW": 90.0, "MEDIUM": 23.0, "HIGH": 9.0, "CRITICAL": 1.0}
    rul_sc  = MinMaxScaler()
    rul_sc.fit([[0], [BASE_LIFE_DAYS]])
    for g, mid in rul_mid.items():
        mask = result_df["risk_level"] == g
        result_df.loc[mask, "rul_norm"] = round(
            float(rul_sc.transform([[mid]])[0][0]), 6
        )

    # 언더샘플링
    final_dfs = []
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        g_df = result_df[result_df["risk_level"] == g]
        if len(g_df) > TARGET_PER_GRADE:
            g_df = g_df.sample(TARGET_PER_GRADE, random_state=42)
        final_dfs.append(g_df)

    final = pd.concat(final_dfs, ignore_index=True).sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    dist2  = Counter(final["risk_level"])
    total2 = len(final)
    print(f"\nSMOTE 후: {total2:,}개")
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        print(f"  {g:10s}: {dist2[g]:>7,}개 ({dist2[g]/total2*100:.1f}%)")

    # 필요 컬럼 추가
    final["source"]    = "smote"
    final["file_name"] = "smote"
    final["rul_days"]  = final["rul_norm"].apply(
        lambda x: round(x * BASE_LIFE_DAYS, 4)
    )

    return final


# ══════════════════════════════════════════════════════════════
# STEP 5. 저장
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
    print("STEP 5. 저장")
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
    print(f"  {'scaler.pkl':35s}: 저장 완료")
    print(f"  {'rul_scaler.pkl':35s}: 저장 완료")

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
        }

    summary = pd.DataFrame([
        make_row("KAIST",              kaist_df),
        make_row("FEMTO train",        femto_train),
        make_row("FEMTO test",         femto_test),
        make_row("combined_train",     combined_train),
        make_row("combined_test",      combined_test),
    ])

    feat_info = pd.DataFrame({
        "feature" : FEATURE_COLS,
        "min"     : feat_scaler.data_min_,
        "max"     : feat_scaler.data_max_,
        "scale"   : feat_scaler.scale_,
    })

    kaist_summary = (
        kaist_df[["file_name","vibration_rms","vibration_kurtosis",
                  "vibration_crest","temperature","rul_days","risk_level"]]
        .copy()
    )

    xlsx_path = OUTPUT_DIR / "preprocess_summary.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(      writer, sheet_name="최종요약",       index=False)
        kaist_summary.to_excel(writer, sheet_name="KAIST_샘플",     index=False)
        feat_info.to_excel(    writer, sheet_name="Scaler_파라미터", index=False)

    print(f"  {'preprocess_summary.xlsx':35s}: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    np.random.seed(42)
    total_start = time.time()

    # API 키 로딩
    api_key = load_api_key()

    # STEP 1. KAIST
    kaist_df = preprocess_kaist(api_key)

    # STEP 2. FEMTO
    femto_train, femto_test = preprocess_femto()

    # STEP 3. 정규화
    combined_pre, combined_test, feat_scaler, rul_scaler = normalize(
        kaist_df, femto_train, femto_test
    )

    # STEP 4. SMOTE
    combined_train = apply_smote(combined_pre)

    # STEP 5. 저장
    save_outputs(
        kaist_df, femto_train, femto_test,
        combined_train, combined_test,
        feat_scaler, rul_scaler,
    )

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")