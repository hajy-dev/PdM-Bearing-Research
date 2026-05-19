"""
═══════════════════════════════════════════════════════════════════
  피처 엔지니어링 모듈
  AI-PASS 예지보전 | 시간 도메인 + 주파수 도메인 + Health Index
═══════════════════════════════════════════════════════════════════

피처 구성 (총 ~22개):
  [시간 도메인 8개] — 원본 진동 신호의 통계적 특성
    vibration_rms, vibration_std, vibration_peak,
    vibration_kurtosis, vibration_skewness,
    crest_factor, impulse_factor, shape_factor

  [주파수 도메인 6개] — FFT 변환 후 스펙트럼 특성
    spectral_energy, spectral_centroid, spectral_spread,
    band_energy_low, band_energy_mid, band_energy_high

  [시계열 파생 3개] — 시간에 따른 변화 추세
    short_trend, rolling_trend, operating_hours

  [환경 보정 5개] — KAIST/Zenodo에서만 (있는 경우)
    temp_residual, ambient_temp, wind_speed, humidity, season
═══════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# 1. 시간 도메인 피처 (파일 단위)
# ══════════════════════════════════════════════
def compute_time_features(rms_value: float, signal_stats: dict = None) -> dict:
    """
    이미 계산된 RMS와 추가 통계값으로 시간 도메인 피처를 반환한다.
    signal_stats가 없으면 RMS만 반환.
    """
    features = {"vibration_rms": rms_value}
    if signal_stats:
        features.update(signal_stats)
    return features


def compute_signal_stats(signal: np.ndarray) -> dict:
    """
    원본 진동 신호에서 통계 피처를 계산한다.

    각 피처의 물리적 의미:
      std:            신호 변동성 (열화 시 증가)
      peak:           최대 충격 크기 (결함 충격 시 피크)
      kurtosis:       뾰족함 (정상=3, 결함 충격 시 증가)
      skewness:       비대칭도 (결함 방향성)
      crest_factor:   Peak/RMS (충격성 — 높으면 간헐적 충격 존재)
      impulse_factor: Peak/Mean(|x|) (충격 감도)
      shape_factor:   RMS/Mean(|x|) (파형 형태)
    """
    rms = float(np.sqrt(np.mean(signal ** 2)))
    std = float(np.std(signal))
    peak = float(np.max(np.abs(signal)))
    mean_abs = float(np.mean(np.abs(signal)))

    # 안전한 나눗셈
    safe_rms = max(rms, 1e-10)
    safe_mean = max(mean_abs, 1e-10)

    return {
        "vibration_std": std,
        "vibration_peak": peak,
        "vibration_kurtosis": float(pd.Series(signal).kurtosis()),
        "vibration_skewness": float(pd.Series(signal).skew()),
        "crest_factor": peak / safe_rms,
        "impulse_factor": peak / safe_mean,
        "shape_factor": rms / safe_mean,
    }


# ══════════════════════════════════════════════
# 2. 주파수 도메인 피처 (파일 단위)
# ══════════════════════════════════════════════
def compute_freq_features(signal: np.ndarray, fs: float = 25600.0) -> dict:
    """
    FFT 변환 후 주파수 도메인 피처를 계산한다.

    Args:
        signal: 원본 진동 신호
        fs: 샘플링 주파수 (Hz) — FEMTO/XJTU=25.6kHz

    피처 설명:
      spectral_energy:    전체 주파수 에너지 (열화 시 증가)
      spectral_centroid:  에너지 중심 주파수 (열화 시 이동)
      spectral_spread:    주파수 분산 (열화 시 확대)
      band_energy_low:    저주파 에너지 (0~1kHz, 불균형/정렬불량)
      band_energy_mid:    중주파 에너지 (1~5kHz, 베어링 결함 주파수)
      band_energy_high:   고주파 에너지 (5kHz+, 초기 결함/마모)
    """
    n = len(signal)
    # FFT (DC 제거, 양의 주파수만)
    fft_vals = np.abs(np.fft.rfft(signal))[1:]
    freqs = np.fft.rfftfreq(n, d=1.0/fs)[1:]

    if len(fft_vals) == 0:
        return {k: 0.0 for k in ['spectral_energy', 'spectral_centroid',
                                   'spectral_spread', 'band_energy_low',
                                   'band_energy_mid', 'band_energy_high']}

    power = fft_vals ** 2
    total_power = float(np.sum(power))
    safe_total = max(total_power, 1e-10)

    # 가중 주파수 (에너지 기준)
    centroid = float(np.sum(freqs * power) / safe_total)
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / safe_total))

    # 대역별 에너지
    low_mask = freqs < 1000
    mid_mask = (freqs >= 1000) & (freqs < 5000)
    high_mask = freqs >= 5000

    return {
        "spectral_energy": total_power,
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "band_energy_low": float(np.sum(power[low_mask])),
        "band_energy_mid": float(np.sum(power[mid_mask])),
        "band_energy_high": float(np.sum(power[high_mask])),
    }


# ══════════════════════════════════════════════
# 3. FFT 피처 추출 (분류용 — 윈도우 배열)
# ══════════════════════════════════════════════
def extract_fft_features_batch(raw_windows: np.ndarray) -> np.ndarray:
    """
    Raw 진동 윈도우 배열에서 FFT+통계 피처를 일괄 추출한다.
    분류 모델(XGBoost+FFT)의 입력으로 사용.

    반환: (n_samples, fft_dim + stat_dim) 배열
    """
    n = len(raw_windows)
    half_win = raw_windows.shape[1] // 2

    fft_features = np.zeros((n, half_win))
    stat_features = np.zeros((n, 7))  # 시간 도메인 통계 7개

    for i in range(n):
        w = raw_windows[i]

        # FFT 진폭 스펙트럼
        fft_vals = np.abs(np.fft.rfft(w))[1:half_win + 1]
        if len(fft_vals) < half_win:
            fft_vals = np.pad(fft_vals, (0, half_win - len(fft_vals)))
        fft_features[i] = fft_vals

        # 시간 도메인 통계
        stats = compute_signal_stats(w)
        stat_features[i] = [
            stats['vibration_std'], stats['vibration_peak'],
            stats['vibration_kurtosis'], stats['vibration_skewness'],
            stats['crest_factor'], stats['impulse_factor'],
            stats['shape_factor'],
        ]

    return np.hstack([fft_features, stat_features])


# ══════════════════════════════════════════════
# 4. 시계열 추세 피처 (베어링 단위)
# ══════════════════════════════════════════════
def add_trend_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    베어링 단위로 시계열 추세 피처를 추가한다.

    short_trend:   직전 대비 RMS 변화율 → 급변 감지
    rolling_trend: 이동평균 대비 편차율 → 서서히 진행되는 열화
    """
    frames = []
    for bid in df['bearing_id'].unique():
        grp = df[df['bearing_id'] == bid].copy().sort_values('file_idx').reset_index(drop=True)
        rms = grp['vibration_rms']

        prev = rms.shift(1).replace(0, np.nan)
        grp['short_trend'] = (rms - prev) / prev

        roll = rms.rolling(window=window, min_periods=1).mean()
        grp['rolling_trend'] = (rms - roll) / roll.replace(0, np.nan)

        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    return result.dropna(subset=['short_trend', 'rolling_trend'])


# ══════════════════════════════════════════════
# 5. Health Index 계산
# ══════════════════════════════════════════════
def compute_health_index(df: pd.DataFrame,
                          features: list,
                          normal_ratio: float = 0.3) -> pd.DataFrame:
    """
    각 베어링의 정상 구간 대비 현재 상태를 0~1로 수치화한다.

    원리:
      1) 각 베어링의 초기 30% 구간 = "정상 기준"
      2) 각 시점의 피처를 정상 기준과의 마할라노비스 거리로 변환
      3) 거리가 클수록 HI가 낮음 (열화 진행)

    간소화 버전:
      HI = 1 - normalized_distance
      distance = sqrt(sum((feature - normal_mean)^2 / normal_std^2))
    """
    df = df.copy()
    df['health_index'] = 1.0

    for bid in df['bearing_id'].unique():
        mask = df['bearing_id'] == bid
        grp = df.loc[mask].sort_values('file_idx')

        n_normal = max(int(len(grp) * normal_ratio), 3)
        normal_data = grp.iloc[:n_normal]

        # 정상 구간의 평균/표준편차
        valid_features = [f for f in features if f in grp.columns]
        if not valid_features:
            continue

        normal_mean = normal_data[valid_features].mean()
        normal_std = normal_data[valid_features].std().replace(0, 1e-6)

        # 표준화된 거리 계산
        normalized = (grp[valid_features] - normal_mean) / normal_std
        distance = np.sqrt((normalized ** 2).sum(axis=1))

        # 0~1 정규화 (0=고장, 1=정상)
        d_min = distance.min()
        d_max = distance.max()
        if d_max > d_min:
            hi = 1.0 - (distance - d_min) / (d_max - d_min)
        else:
            hi = pd.Series(1.0, index=distance.index)

        df.loc[grp.index, 'health_index'] = hi.values

    return df


# ══════════════════════════════════════════════
# 6. RUL → 4등급 변환
# ══════════════════════════════════════════════
def rul_to_severity(rul_ratio: float) -> int:
    """
    RUL 비율을 위험도 4등급으로 변환한다.

    수명 대비 비율 기준 (절대 일수가 아님):
      > 0.6       → 0 (LOW)      — 수명의 60% 이상 남음
      0.3 ~ 0.6   → 1 (MEDIUM)   — 30~60% 남음
      0.1 ~ 0.3   → 2 (HIGH)     — 10~30% 남음
      ≤ 0.1       → 3 (CRITICAL) — 10% 이하 남음
    """
    if rul_ratio > 0.6:
        return 0  # LOW
    elif rul_ratio > 0.3:
        return 1  # MEDIUM
    elif rul_ratio > 0.1:
        return 2  # HIGH
    else:
        return 3  # CRITICAL

SEVERITY_NAMES = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']


def add_severity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """RUL 비율에서 4등급 라벨을 생성한다."""
    df = df.copy()
    df['severity'] = df['rul_ratio'].apply(rul_to_severity)
    df['severity_name'] = df['severity'].map(lambda x: SEVERITY_NAMES[x])
    return df
