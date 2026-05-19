"""
═══════════════════════════════════════════════════════════════════
  데이터 로더 통합 모듈
  AI-PASS 예지보전 | #1 KAIST, #2 FEMTO, #3 XJTU-SY,
                     #5 Paderborn, #6 IMS, #18 Zenodo
═══════════════════════════════════════════════════════════════════
"""

import os
import sys
import glob
import logging
import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weather_api import fetch_kaist_weather
from feature_engineering import compute_signal_stats, compute_freq_features

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════
# 경로 설정
# ══════════════════════════════════════════════
BASE      = r"D:\project\데이터셋"
FEMTO_DIR = os.path.join(BASE, "10. FEMTO Bearing", "FEMTOBearingDataSet", "Full_Test_Set")
KAIST_DIR = os.path.join(BASE, "Vibration_Bearing_RuntoFailure")
XJTU_DIR  = os.path.join(BASE, "XJTU-SY_Bearing_Datasets", "Data", "XJTU-SY_Bearing_Datasets")
IMS_DIR   = os.path.join(BASE, "4. Bearings", "IMS")
ZENODO_DIR = os.path.join(BASE, "Zenodo")
PADERBORN_DIR = os.path.join(BASE, "Paderborn Univ_Bearing Dataset")

WINDOW_SIZE = 256
STEP = 128
FAILURE_THRESHOLD_G = 20.0  # FEMTO 논문 기준


def compute_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


# ══════════════════════════════════════════════
# 1. FEMTO 로더
# ══════════════════════════════════════════════
def load_femto() -> pd.DataFrame:
    """
    FEMTO 데이터: 11 bearings, RTF, 진동만
    Failure Threshold(20g) 기반 RUL 계산
    """
    records = []
    bearing_dirs = sorted(glob.glob(os.path.join(FEMTO_DIR, "Bearing*")))

    for b_dir in tqdm(bearing_dirs, desc="[FEMTO]", unit="bearing"):
        bearing_id = "FEMTO_" + os.path.basename(b_dir)
        csv_files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))
        if not csv_files:
            continue

        # 파일별 피처 계산
        file_features = []
        for fp in csv_files:
            try:
                with open(fp) as f:
                    first = f.readline()
                sep = ';' if ';' in first else ','
                df = pd.read_csv(fp, header=None, sep=sep)
                col = 4 if df.shape[1] >= 6 else (df.shape[1] - 1)
                signal = df.iloc[:, col].values.astype(float)
                feat = {"vibration_rms": compute_rms(signal)}
                feat.update(compute_signal_stats(signal))
                feat.update(compute_freq_features(signal, fs=25600.0))
                file_features.append(feat)
            except Exception:
                file_features.append(None)

        if not file_features:
            continue

        # Failure Threshold 시점
        failure_idx = len(file_features) - 1
        for idx, feat in enumerate(file_features):
            if feat is not None and feat["vibration_rms"] > FAILURE_THRESHOLD_G:
                failure_idx = idx
                break

        total = failure_idx + 1
        for idx in range(total):
            rul_ratio = (failure_idx - idx) / max(total - 1, 1)
            rec = {
                "source": "FEMTO",
                "bearing_id": bearing_id,
                "file_idx": idx,
                "rul_ratio": rul_ratio,
                "total_files": total,
                "has_temp": False,
            }
            if file_features[idx] is not None:
                rec.update(file_features[idx])
            else:
                rec["vibration_rms"] = 0.0
            records.append(rec)

    df = pd.DataFrame(records)
    log.info(f"  [FEMTO] {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    return df


# ══════════════════════════════════════════════
# 2. KAIST 로더
# ══════════════════════════════════════════════
def load_kaist() -> pd.DataFrame:
    """
    KAIST 데이터: 1 bearing, RTF, 진동+온도+기상청 ambient
    환경 보정(temp_residual) 계산 포함
    """
    records = []
    csv_list = sorted(glob.glob(os.path.join(KAIST_DIR, "**", "*.csv"), recursive=True))
    if not csv_list:
        log.warning("  [KAIST] CSV 없음")
        return pd.DataFrame()

    # 기상청 API 매핑
    log.info("  [KAIST] 기상청 API 매핑...")
    weather_df = fetch_kaist_weather(csv_list)
    weather_map = {}
    if not weather_df.empty:
        for _, row in weather_df.iterrows():
            weather_map[row['filepath']] = {
                'ambient_temp': row['ambient_temp'],
                'wind_speed': row['wind_speed'],
                'humidity': row['humidity'],
                'season': row['season'],
            }

    # 센서 데이터 로드
    file_stats = []
    for fp in tqdm(csv_list, desc="[KAIST]", unit="file"):
        try:
            df = pd.read_csv(fp, header=None, nrows=100000)
            if df.shape[1] < 4:
                continue
            vib_x = df.iloc[:, 0].values.astype(float)
            vib_y = df.iloc[:, 1].values.astype(float)
            combined = np.sqrt(vib_x**2 + vib_y**2)
            weather = weather_map.get(fp, {})
            ambient = weather.get('ambient_temp', float(np.mean(df.iloc[:, 3].values)))

            feat = {"vibration_rms": compute_rms(combined)}
            feat.update(compute_signal_stats(combined))
            feat.update(compute_freq_features(combined, fs=25600.0))
            feat.update({
                "filepath": fp,
                "device_temp": float(df.iloc[:, 2].mean()),
                "ambient_temp": ambient if not np.isnan(ambient) else float(df.iloc[:, 3].mean()),
                "wind_speed": weather.get('wind_speed', 0.0),
                "humidity": weather.get('humidity', 0.0),
                "season": weather.get('season', 'unknown'),
            })
            file_stats.append(feat)
        except Exception:
            continue

    if not file_stats:
        return pd.DataFrame()

    # baseline_heat 계산 (NaN 제외)
    n_normal = max(len(file_stats) // 2, 1)
    normal_diffs = [
        s["device_temp"] - s["ambient_temp"]
        for s in file_stats[:n_normal]
        if not (np.isnan(s["device_temp"]) or np.isnan(s["ambient_temp"]))
    ]
    baseline_heat = float(np.mean(normal_diffs)) if normal_diffs else 10.0
    log.info(f"  [KAIST] 장비 평균 발열값: {baseline_heat:.2f}°C")

    total = len(file_stats)
    for idx, s in enumerate(file_stats):
        expected = s["ambient_temp"] + baseline_heat
        rec = {
            "source": "KAIST",
            "bearing_id": "KAIST_001",
            "file_idx": idx,
            "device_temp": s["device_temp"],
            "ambient_temp": s["ambient_temp"],
            "temp_residual": s["device_temp"] - expected,
            "wind_speed": s["wind_speed"],
            "humidity": s["humidity"],
            "season": s["season"],
            "rul_ratio": (total - 1 - idx) / max(total - 1, 1),
            "total_files": total,
            "has_temp": True,
        }
        # 진동 피처 merge (vibration_rms, signal_stats, freq_features)
        for k in s:
            if k not in ('filepath', 'device_temp', 'ambient_temp',
                         'wind_speed', 'humidity', 'season'):
                rec[k] = s[k]
        records.append(rec)

    df = pd.DataFrame(records)
    log.info(f"  [KAIST] {len(df):,} rows / temp_residual: "
             f"{df['temp_residual'].min():.1f} ~ {df['temp_residual'].max():.1f}°C")
    return df


# ══════════════════════════════════════════════
# 3. IMS 로더
# ══════════════════════════════════════════════
#
# IMS 고장 정보 (NASA 문서 기준):
#   1st_test: 8채널(4 bearings×2축), Bearing3=inner, Bearing4=roller
#   2nd_test: 4채널(4 bearings×1축), Bearing1=outer
#   4th_test: 4채널(=원본 3rd_test), Bearing3=outer
#
IMS_FAULT_MAP = {
    "1st_test": {"fault_bearing": [2, 3], "n_channels": 8, "n_bearings": 4},
    "2nd_test": {"fault_bearing": [0],    "n_channels": 4, "n_bearings": 4},
    "4th_test": {"fault_bearing": [2],    "n_channels": 4, "n_bearings": 4},
}

def load_ims() -> pd.DataFrame:
    """
    IMS 데이터: 3세트 × 4 bearings = 최대 12, 고장 bearings만 RTF로 사용
    진동만, 온도 없음
    """
    records = []

    for test_name, info in IMS_FAULT_MAP.items():
        test_dir = os.path.join(IMS_DIR, test_name)
        if test_name == "4th_test":
            test_dir = os.path.join(test_dir, "txt")
        if not os.path.isdir(test_dir):
            continue

        files = sorted(os.listdir(test_dir))
        files = [f for f in files if not f.endswith(('.zip', '.txt.gz'))]
        total = len(files)
        if total == 0:
            continue

        # 고장 베어링별로 RUL 계산
        for fault_ch in info["fault_bearing"]:
            bearing_id = f"IMS_{test_name}_B{fault_ch}"
            file_features = []

            for fp_name in tqdm(files, desc=f"[IMS] {test_name} B{fault_ch}", unit="file", leave=False):
                fp = os.path.join(test_dir, fp_name)
                try:
                    data = np.loadtxt(fp, delimiter='\t')
                    if data.ndim == 1:
                        file_features.append(None)
                        continue
                    # 해당 베어링 채널 추출
                    if info["n_channels"] == 8:
                        ch1 = data[:, fault_ch * 2]
                        ch2 = data[:, fault_ch * 2 + 1]
                        combined = np.sqrt(ch1**2 + ch2**2)
                    else:
                        combined = data[:, fault_ch]
                    feat = {"vibration_rms": compute_rms(combined)}
                    feat.update(compute_signal_stats(combined))
                    feat.update(compute_freq_features(combined, fs=20480.0))
                    file_features.append(feat)
                except Exception:
                    file_features.append(None)

            if not file_features:
                continue

            for idx in range(total):
                rul_ratio = (total - 1 - idx) / max(total - 1, 1)
                rec = {
                    "source": "IMS",
                    "bearing_id": bearing_id,
                    "file_idx": idx,
                    "rul_ratio": rul_ratio,
                    "total_files": total,
                    "has_temp": False,
                }
                if idx < len(file_features) and file_features[idx] is not None:
                    rec.update(file_features[idx])
                else:
                    rec["vibration_rms"] = 0.0
                records.append(rec)

    df = pd.DataFrame(records)
    log.info(f"  [IMS] {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    return df


# ══════════════════════════════════════════════
# 4. Zenodo 로더
# ══════════════════════════════════════════════
#
# 7 bearings (B01~B07), RTF, 진동+온도+ambient
# B05 = 의도적 중단 → 제외
# B06 = 7파트, B07 = 3파트 → 병합
# 진동 단위: Voltage(V), 감도 100mV/g → V/0.1 = g
#
ZENODO_V_TO_G = 1.0 / 0.1  # 100mV/g → 1V = 10g

ZENODO_EXCLUDE = {"B05"}  # 의도적 중단, RTF 아님

def load_zenodo() -> pd.DataFrame:
    """
    Zenodo 데이터: 6 bearings (B05 제외), RTF, 진동+온도+ambient
    환경 보정(temp_residual) 계산 포함
    """
    records = []
    # 베어링별 폴더 매핑 (part 분할 처리)
    bearing_parts = {}
    for d in sorted(os.listdir(ZENODO_DIR)):
        full = os.path.join(ZENODO_DIR, d)
        if not os.path.isdir(full):
            continue
        # B01, B06_part1 등에서 베어링 ID 추출
        bid = d.split("_")[0]  # "B06_part1" → "B06"
        if bid in ZENODO_EXCLUDE:
            continue
        if bid not in bearing_parts:
            bearing_parts[bid] = []
        bearing_parts[bid].append(full)

    for bid in sorted(bearing_parts.keys()):
        parts = sorted(bearing_parts[bid])

        # 온도 CSV 로드 (첫 번째 part 폴더에 위치)
        temp_df = None
        for part_dir in parts:
            temp_path = os.path.join(part_dir, f"{bid}_meanTemperatures.csv")
            if os.path.exists(temp_path):
                temp_df = pd.read_csv(temp_path)
                break

        if temp_df is None:
            log.warning(f"  [Zenodo] {bid} 온도 파일 없음 — 건너뜀")
            continue

        # 온도 데이터 추출
        device_temps = temp_df.iloc[:, 1].values  # T1 (베어링 온도)
        ambient_temps = temp_df.iloc[:, 3].values  # Room Temp
        n_measurements = len(temp_df)

        # baseline_heat (초기 50%)
        n_normal = max(n_measurements // 2, 1)
        valid_diffs = []
        for i in range(n_normal):
            dt = device_temps[i]
            at = ambient_temps[i]
            if not (np.isnan(dt) or np.isnan(at)):
                valid_diffs.append(dt - at)
        baseline_heat = float(np.mean(valid_diffs)) if valid_diffs else 10.0

        # 진동 파일 수집 (part 순서대로 병합)
        all_mat_files = []
        for part_dir in parts:
            vib_dir = os.path.join(part_dir, "vibrationData")
            if os.path.isdir(vib_dir):
                mats = sorted(glob.glob(os.path.join(vib_dir, "*.mat")))
                all_mat_files.extend(mats)

        if not all_mat_files:
            continue

        # 파일별 피처 계산
        file_features = []
        for fp in tqdm(all_mat_files, desc=f"[Zenodo] {bid}", unit="file", leave=False):
            try:
                mat = scipy.io.loadmat(fp)
                sig = mat['accHorizRear_A'].flatten()
                sig_g = sig * ZENODO_V_TO_G
                feat = {"vibration_rms": compute_rms(sig_g)}
                feat.update(compute_signal_stats(sig_g))
                feat.update(compute_freq_features(sig_g, fs=64000.0))
                file_features.append(feat)
            except Exception:
                file_features.append(None)

        # 진동 파일 수와 온도 측정 수 맞춤 (min 기준)
        n_use = min(len(file_features), n_measurements)
        bearing_id = f"ZENODO_{bid}"

        for idx in range(n_use):
            dt = device_temps[idx] if idx < n_measurements else np.nan
            at = ambient_temps[idx] if idx < n_measurements else np.nan
            expected = at + baseline_heat if not np.isnan(at) else np.nan
            temp_res = dt - expected if not (np.isnan(dt) or np.isnan(expected)) else np.nan

            rul_ratio = (n_use - 1 - idx) / max(n_use - 1, 1)
            rec = {
                "source": "ZENODO",
                "bearing_id": bearing_id,
                "file_idx": idx,
                "device_temp": dt,
                "ambient_temp": at,
                "temp_residual": temp_res,
                "rul_ratio": rul_ratio,
                "total_files": n_use,
                "has_temp": True,
            }
            if file_features[idx] is not None:
                rec.update(file_features[idx])
            else:
                rec["vibration_rms"] = 0.0
            records.append(rec)

    df = pd.DataFrame(records)
    if not df.empty:
        log.info(f"  [Zenodo] {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
        temp_valid = df['temp_residual'].dropna()
        if not temp_valid.empty:
            log.info(f"    temp_residual: {temp_valid.min():.1f} ~ {temp_valid.max():.1f}°C")
    return df


# ══════════════════════════════════════════════
# 5. XJTU-SY 로더 (고장모드 분류용)
# ══════════════════════════════════════════════
XJTU_LABEL_MAP = {
    "Bearing1_1": "outer_race", "Bearing1_2": "outer_race",
    "Bearing1_3": "outer_race", "Bearing1_4": "cage",
    "Bearing1_5": "outer_race",
    "Bearing2_1": "inner_race", "Bearing2_2": "outer_race",
    "Bearing2_3": "cage",       "Bearing2_4": "outer_race",
    "Bearing2_5": "outer_race",
    "Bearing3_1": "outer_race", "Bearing3_2": "inner_race",
    "Bearing3_3": "outer_race", "Bearing3_4": "inner_race",
    "Bearing3_5": "outer_race",
}

def load_xjtu() -> pd.DataFrame:
    """
    XJTU-SY: 15 bearings, 진동, 고장 라벨
    고장 직전 30% 데이터만 사용 (고장 패턴이 뚜렷한 구간)
    """
    records = []
    condition_dirs = sorted(glob.glob(os.path.join(XJTU_DIR, "*")))

    for cond_dir in condition_dirs:
        if not os.path.isdir(cond_dir):
            continue
        for b_dir in sorted(glob.glob(os.path.join(cond_dir, "Bearing*"))):
            bearing_id = os.path.basename(b_dir)
            label = XJTU_LABEL_MAP.get(bearing_id)
            if label is None:
                continue

            csv_files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))
            n_use = max(len(csv_files) * 30 // 100, 1)
            csv_files = csv_files[-n_use:]

            for fp in tqdm(csv_files, desc=f"[XJTU] {bearing_id}", unit="file", leave=False):
                try:
                    df = pd.read_csv(fp, nrows=100000)
                    col = 'Horizontal' if 'Horizontal' in df.columns else df.columns[0]
                    signal = df[col].values.astype(float)
                    for i in range(0, len(signal) - WINDOW_SIZE, STEP):
                        window = signal[i:i + WINDOW_SIZE]
                        records.append({
                            "source": "XJTU",
                            "bearing_id": bearing_id,
                            "fault_label": label,
                            "raw": window,
                        })
                except Exception:
                    continue

    df = pd.DataFrame(records)
    if not df.empty:
        log.info(f"  [XJTU] {len(df):,} windows / {df['fault_label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════
# 5-b. XJTU-SY RTF 로더 (RUL 예측용)
# ══════════════════════════════════════════════
def load_xjtu_rtf() -> pd.DataFrame:
    """
    XJTU-SY: 15 bearings, RTF, 진동만
    전체 수명 데이터를 파일별 피처로 변환 (fs=25.6kHz)
    """
    records = []
    condition_dirs = sorted(glob.glob(os.path.join(XJTU_DIR, "*")))

    for cond_dir in condition_dirs:
        if not os.path.isdir(cond_dir):
            continue
        for b_dir in sorted(glob.glob(os.path.join(cond_dir, "Bearing*"))):
            bearing_id = "XJTU_" + os.path.basename(b_dir)
            csv_files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))
            if not csv_files:
                continue

            file_features = []
            for fp in tqdm(csv_files, desc=f"[XJTU-RTF] {bearing_id}", unit="file", leave=False):
                try:
                    df = pd.read_csv(fp, nrows=100000)
                    h_col = 'Horizontal' if 'Horizontal' in df.columns else df.columns[0]
                    v_col = 'Vertical' if 'Vertical' in df.columns else df.columns[1]
                    h = df[h_col].values.astype(float)
                    v = df[v_col].values.astype(float)
                    combined = np.sqrt(h**2 + v**2)
                    feat = {"vibration_rms": compute_rms(combined)}
                    feat.update(compute_signal_stats(combined))
                    feat.update(compute_freq_features(combined, fs=25600.0))
                    file_features.append(feat)
                except Exception:
                    file_features.append(None)

            if not file_features:
                continue

            total = len(file_features)
            for idx in range(total):
                rul_ratio = (total - 1 - idx) / max(total - 1, 1)
                rec = {
                    "source": "XJTU",
                    "bearing_id": bearing_id,
                    "file_idx": idx,
                    "rul_ratio": rul_ratio,
                    "total_files": total,
                    "has_temp": False,
                }
                if file_features[idx] is not None:
                    rec.update(file_features[idx])
                else:
                    rec["vibration_rms"] = 0.0
                records.append(rec)

    df = pd.DataFrame(records)
    log.info(f"  [XJTU-RTF] {len(df):,} rows / {df['bearing_id'].nunique()} bearings")
    return df


# ══════════════════════════════════════════════
# 6. Paderborn 로더 (고장모드 분류용)
# ══════════════════════════════════════════════
PADERBORN_LABEL_MAP = {"K": "healthy", "KA": "outer_race", "KI": "inner_race", "KB": "ball"}

def load_paderborn() -> pd.DataFrame:
    """
    Paderborn: 32 bearings, 진동+전류+온도, 고장 라벨
    채널: vibration_1(ch6), phase_current_1(ch1), temp_2_bearing_module(ch4)
    """
    records = []
    folders = sorted([d for d in os.listdir(PADERBORN_DIR)
                      if os.path.isdir(os.path.join(PADERBORN_DIR, d))])

    for folder in tqdm(folders, desc="[Paderborn]", unit="bearing"):
        # 라벨 결정
        prefix = folder[:2] if folder[:2] in ('KA', 'KI', 'KB') else folder[:1]
        label = PADERBORN_LABEL_MAP.get(prefix)
        if label is None:
            continue

        mat_files = sorted(glob.glob(os.path.join(PADERBORN_DIR, folder, "*.mat")))
        for fp in mat_files:
            try:
                mat = scipy.io.loadmat(fp)
                key = [k for k in mat.keys() if not k.startswith('_')][0]
                data = mat[key][0, 0]
                Y = data['Y']

                # 채널 추출
                vib_data = None
                current_data = None
                for i in range(Y.shape[1]):
                    ch = Y[0, i]
                    name = str(ch['Name'][0]) if ch['Name'].size > 0 else ''
                    d = ch['Data'].flatten()
                    if 'vibration' in name:
                        vib_data = d
                    elif 'phase_current_1' in name:
                        current_data = d

                if vib_data is None:
                    continue

                # 윈도우 슬라이싱 (진동)
                for i in range(0, len(vib_data) - WINDOW_SIZE, STEP):
                    raw_vib = vib_data[i:i + WINDOW_SIZE]
                    raw_cur = current_data[i:i + WINDOW_SIZE] if current_data is not None and len(current_data) > i + WINDOW_SIZE else None
                    rec = {
                        "source": "PADERBORN",
                        "bearing_id": folder,
                        "fault_label": label,
                        "raw": raw_vib,
                    }
                    if raw_cur is not None:
                        rec["raw_current"] = raw_cur
                        rec["current_rms"] = compute_rms(raw_cur)
                    records.append(rec)
            except Exception:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        log.info(f"  [Paderborn] {len(df):,} windows / {df['fault_label'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════
# 7. 통합 RTF 로더 (RUL 예측용)
# ══════════════════════════════════════════════
def load_all_rtf() -> pd.DataFrame:
    """
    RUL 예측용 전체 RTF 데이터 통합 로드
    FEMTO(11) + KAIST(1) + IMS(4) + Zenodo(6) = 22 bearings
    """
    log.info("=" * 50)
    log.info("  RTF 데이터 통합 로드")
    log.info("=" * 50)

    frames = []
    for loader, name in [(load_femto, "FEMTO"), (load_kaist, "KAIST"),
                          (load_ims, "IMS"), (load_zenodo, "Zenodo"),
                          (load_xjtu_rtf, "XJTU")]:
        try:
            df = loader()
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log.error(f"  [{name}] 로드 실패: {e}")

    if not frames:
        log.error("  RTF 데이터 없음")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    log.info(f"  [통합] {len(df_all):,} rows / {df_all['bearing_id'].nunique()} bearings")
    log.info(f"  [통합] 소스별: {df_all.groupby('source')['bearing_id'].nunique().to_dict()}")
    return df_all


# ══════════════════════════════════════════════
# 8. 통합 분류 로더 (고장모드 분류용)
# ══════════════════════════════════════════════
def load_all_classification() -> pd.DataFrame:
    """
    고장모드 분류용 전체 데이터 통합 로드
    XJTU-SY(15) + Paderborn(32) = 47 bearings
    """
    log.info("=" * 50)
    log.info("  분류 데이터 통합 로드")
    log.info("=" * 50)

    frames = []
    for loader in [load_xjtu, load_paderborn]:
        try:
            df = loader()
            if not df.empty:
                frames.append(df)
        except Exception as e:
            log.error(f"  로드 실패: {e}")

    if not frames:
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    log.info(f"  [통합] {len(df_all):,} windows / "
             f"소스: {df_all.groupby('source')['bearing_id'].nunique().to_dict()}")
    return df_all
