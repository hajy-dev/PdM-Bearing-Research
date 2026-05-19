# 증명: 이상탐지 16피처 확장

## 변경 내용
- `compare_anomaly_models.py` 피처 3→16개 확장
- 기존: vibration_rms, short_trend, rolling_trend (+ temp_residual)
- 추가: vibration_std, vibration_peak, vibration_kurtosis, vibration_skewness, crest_factor, impulse_factor, shape_factor, spectral_energy, spectral_centroid, spectral_spread, band_energy_low, band_energy_mid, band_energy_high
- data_loaders.py에서 이미 계산되어 있으므로 피처명만 추가
- 정상/이상 기준: rul_ratio 기반 (NORMAL > 0.5, ANOMALY ≤ 0.2) 변경 없음

## 결과

| 모델 | 기준선 (3피처) | 16피처 | 변화 |
|------|--------------|--------|------|
| IsolationForest | Det 30.6% / AUC 0.649 | Det **47.7%** / AUC **0.746** | Det +17.1%p, AUC +0.097 |
| Autoencoder | Det 37.7% / AUC 0.696 | Det **55.1%** / AUC **0.775** | Det +17.4%p, AUC +0.079 |
| OneClassSVM | Det 41.1% / AUC 0.737 | Det **53.2%** / AUC **0.789** | Det +12.1%p, AUC +0.052 |

## 판정
**성공.** 피처 확장만으로 전 모델 대폭 개선. FPR 5.1~5.3%로 오탐 변동 없이 탐지율만 상승.
