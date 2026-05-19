# CLAUDE.md

T 
python run_all.py          # 전체 실행 (RUL → 분류 → 이상탐지)
python run_all.py rul      # RUL 위험도 4등급 분류만
python run_all.py clf      # 고장모드 분류만
python run_all.py ad       # 이상 탐지만
```

## Architecture

3개의 독립적인 실험 파이프라인이 공통 데이터/피처 모듈을 공유한다:

```
data_loaders.py ──→ feature_engineering.py ──→ compare_rul_models.py      (RUL 4등급)
                                           ──→ compare_classification_models.py (고장모드)
                                           ──→ compare_anomaly_models.py  (이상탐지)
```

- **data_loaders.py**: 6개 데이터셋 로더 통합. RTF용 `load_all_rtf()` (22 bearings)와 분류용 `load_all_classification()` (47 bearings) 두 진입점 제공. RTF 로더는 파일별 RMS + 시간도메인 통계(7) + 주파수도메인(6) 피처를 계산하여 저장.
- **feature_engineering.py**: 추세 피처(short_trend, rolling_trend) + Health Index + 4등급 라벨 변환 (RUL 기반, HI 기반 두 가지)
- **weather_api.py**: KAIST 데이터용 기상청 API 연동 (캐시: `예지보전_v2/weather_cache/`)

## Data Flow

| 데이터셋 | Bearings | 용도 | 특이사항 |
|---------|----------|------|---------|
| FEMTO | 11 | RUL/이상탐지 | Failure Threshold 20g, fs=25.6kHz |
| KAIST | 1 | RUL/이상탐지 | 진동+온도+기상청 ambient, fs=25.6kHz |
| IMS | 4 | RUL/이상탐지 | 3세트, 고장 bearing만 사용, fs=20.48kHz |
| Zenodo | 6 | RUL/이상탐지 | B05 제외, V→g 변환(×10), fs=64kHz |
| XJTU-SY | 15 | 고장분류 | 후반 30%만 사용 (RTF 데이터이나 현재 분류 전용) |
| Paderborn | 32 | 고장분류 | 진동+전류+온도 |

## Key Constants

- 데이터셋 경로: `D:\project\데이터셋\` (hardcoded in `data_loaders.py`)
- 출력 경로: `D:\project\예지보전_v2\compare_*_v2\`
- `WINDOW_SIZE = 256`, `STEP = 128` (sliding window)

## RUL 실험 현재 설정 (최적 상태)

- **라벨링**: Health Index 기반 4등급 — LOW(>0.7), MEDIUM(0.4~0.7), HIGH(0.15~0.4), CRITICAL(≤0.15)
- **피처 (16개)**: vibration_rms, short_trend, rolling_trend, vibration_std, vibration_peak, vibration_kurtosis, vibration_skewness, crest_factor, impulse_factor, shape_factor, spectral_energy, spectral_centroid, spectral_spread, band_energy_low, band_energy_mid, band_energy_high
- **정규화**: 베어링별 z-score 정규화 (health_index 제외 — 라벨로 사용)
- **Optuna**: 과적합 억제 방향 탐색 범위 (max_depth 3~6, reg 0.1~50 등), RF도 Optuna 적용
- **클래스 균형**: compute_sample_weights로 소수 클래스 가중치 부여
- **최고 성능**: F1-Macro 50.79% (XGB+Env, 실험B), Gap 3.3%

## Dependencies

pandas, numpy, scipy, scikit-learn, xgboost, lightgbm, optuna, tensorflow, matplotlib, requests, tqdm

## Evaluation Methods

- **RUL**: LOBO-CV (Leave-One-Bearing-Out, GroupKFold 10-fold) + Optuna 60 trials × 3 모델
- **고장분류**: 5-Fold StratifiedKFold
- **이상탐지**: 80/20 정상 split, 5회 반복
- 모든 실험에서 train/test gap(overfit) 모니터링

## 작업 방식

**1. 코드 확인**
- 관련 코드를 깊이 있게 분석한 후 상세 보고
- 표면적 문제뿐 아니라 근본 원인까지 파악

**2. 코드 수정**
- 바로 수정하지 않음
- 수정할 코드를 먼저 보여주며 상세 설명
- 사용자 확인 후 적용

**3. 변수 통제**
- 한 번에 하나의 변수만 변경
- 각 변경마다 실행 → 결과 확인 → 비교 보고 → 다음 단계 논의

**4. 백업 버전 관리**
- `run_all.py` 전체 실행 시: `예지보전_v2/backup_vN/` (v3, v4...)
- 개별 변수 통제 실험 시: `예지보전_v2/증명_[요약명]/` (src/ + results/)
- `backup_v2/` = 최초 기준선

**5. 수정 순서**
- A(RUL) → C(이상탐지) → B(고장모드)
- 각 실험 내에서도 한 변수씩 순차 진행

## RUL 개선 이력

| 단계 | 변경 | 최고 F1 | Gap | 백업 |
|------|------|--------|-----|------|
| 기준선 | operating_hours 포함 (leakage) | 99.61% (무효) | 0.0% | backup_v2 |
| 1 | operating_hours 제거 | 29.43% | 19.0% | 증명_RUL_operating_hours제거 |
| 2 | 17피처 확장 (시간통계7+주파수6) | 35.80% | 22.1% | — |
| 3 | 베어링별 min-max 정규화 | 34.23% (실패) | 39.2% | 증명_RUL_17피처_minmax정규화 |
| 4 | 베어링별 z-score 정규화 | 40.89% | 24.9% | 증명_RUL_17피처_zscore정규화 |
| 5 | Optuna 과적합 억제 + RF Optuna | 41.09% | 14.7% | 증명_RUL_Optuna과적합억제 |
| 6 | HI 기반 라벨링 (피처에서 HI 제거) | 48.66% | 2.4% | 증명_RUL_HI기반라벨링 |
| 7 | balanced weight (sample_weight) | **50.79%** | **3.3%** | 증명_RUL_HI라벨_balanced |
| 8 | 사분위 기준 (실패, 되돌림) | 42.50% | 37.7% | — |

## 이상탐지 실험 현재 설정 (최적 상태)

- **피처 (16개)**: RUL과 동일한 16개 피처
- **정규화**: 베어링별 z-score
- **정상/이상 기준**: Health Index 기반 — 정상(HI > 0.7), 이상(HI ≤ 0.6)
- **HP 최적화**: 미적용 (비지도학습 — 테스트셋 과적합 위험)
- **최고 성능**: AUC 0.977, Det Rate 87.9% (Autoencoder, ≤0.6 기준)
- **Leakage 검증 완료**: HI 소스 피처 제거 후에도 AUC 0.984 유지
- **난이도 검증 완료**: 경계 좁힘(간격 0.1)에서도 AUC 0.97 유지

## 이상탐지 개선 이력

| 단계 | 변경 | 최고 AUC | 최고 Det Rate | 백업 |
|------|------|---------|-------------|------|
| 기준선 | 3피처, rul_ratio 기준 | 0.737 | 41.1% | backup_v2 |
| C-1 | 16피처 확장 | 0.789 | 55.1% | 증명_AD_16피처확장 |
| C-2 | 베어링별 z-score 정규화 | 0.852 | 55.4% | 증명_AD_zscore정규화 |
| C-3 | HI 기반 정상/이상 분리 | **0.988** | **97.2%** | 증명_AD_HI기반분리 |
| 검증 | HI 소스 피처 제거 (leakage 검증) | 0.984 | 96.4% | 증명_AD_HI분리_leakage검증 |

## 고장분류 실험 현재 설정 (기준선)

- **데이터**: XJTU-SY(15) + Paderborn(32) = 47 bearings
- **클래스**: ball, cage, inner_race, outer_race (healthy 제외)
- **균형화**: min_count 다운샘플링
- **모델**: CNN-1D(Raw), XGBoost+FFT, RF+FFT
- **평가**: 5-Fold StratifiedKFold
- **최고 성능**: CNN-1D Acc 86.1%, F1 86.0%, Gap 3.1%
- **핵심 병목**: inner_race vs outer_race 혼동

## Known Issues

- `run_all.py`에서 `root.handlers.clear()` 후 `logging.basicConfig`가 재설정되지 않아 로그 파일이 비어 있음
- XJTU-SY는 RTF 데이터이나 현재 고장분류 전용으로만 사용 — `load_xjtu_rtf()` 함수는 코드에 존재하나 미등록
- HI 분포가 극도로 편향 (98%가 >0.7) — 하드코딩 기준 + balanced weight 조합이 현재 최적
