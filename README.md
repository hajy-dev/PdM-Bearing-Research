# 🔧 PdM Bearing Research

> **베어링 진동 데이터 기반 예지보전 R&D 실험 기록**
> RUL · 이상탐지 · 고장분류 3가지 ML 파이프라인 통합 비교 실험.

[![Status](https://img.shields.io/badge/status-completed-success)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 🎯 프로젝트 개요

6개 공개 베어링 데이터셋(FEMTO·KAIST·IMS·Zenodo·XJTU-SY·Paderborn)을 통합하고, 3가지 ML 태스크를 변수 통제 실험으로 비교했다.

### 핵심 성과

| 파이프라인 | 최고 성능 | 평가 방식 |
|---|---|---|
| **RUL** (4등급 분류) | F1-Macro **50.79%** (Gap 3.3%) | LOBO-CV 10-fold + Optuna |
| **이상 탐지** | AUC **0.977**, Det Rate 87.9% | Autoencoder, 80/20 split × 5 |
| **고장 모드 분류** | Acc **86.1%**, F1 86.0% (Gap 3.1%) | 5-Fold Stratified |

### 주요 발견

1. **Data Leakage 검증의 중요성** — 초기 F1 99.61% → 검증 후 진정한 50.79%로 보정
2. **HI 기반 라벨링** — 시간 기반보다 8%p 향상 (개체차 흡수)
3. **LOBO-CV 필수성** — 같은 베어링의 train/test 중복 방지 → 일반화 평가

---

## 📁 폴더 구조

```
PdM-Bearing-Research/
├── README.md                # ← 현재 문서
├── CLAUDE.md                # 코드 가이드 (아키텍처·실험 흐름)
├── requirements.txt         # 의존성
├── .gitignore
│
├── src/                     # 핵심 코드 (운영용)
│   ├── data_loaders.py              # 6개 데이터셋 통합 로더
│   ├── feature_engineering.py       # 16피처 + HI + 4등급 라벨
│   ├── compare_rul_models.py        # RUL 4등급 분류
│   ├── compare_anomaly_models.py    # 이상 탐지
│   ├── compare_classification_models.py  # 고장 모드 분류
│   ├── extract_clf_features.py      # 분류용 피처 추출
│   ├── verify_3features.py          # 3피처 검증
│   ├── weather_api.py               # KAIST 기상 데이터
│   └── run_all.py                   # 통합 실행
│
├── experiments/             # 변수 통제 실험 (총 19개)
│   ├── README.md
│   ├── RUL/                 # RUL 10개 단계
│   │   ├── 00_baseline/
│   │   ├── 01_operating_hours제거/
│   │   ├── 02_minmax정규화_실패/
│   │   ├── 03_zscore정규화/
│   │   ├── 04_Optuna과적합억제/
│   │   ├── 05_HI기반라벨링/
│   │   ├── 06_HI라벨_balanced_FINAL/   ⭐ 최고 성능
│   │   ├── 07_사분위실패_되돌림/
│   │   ├── 08_XJTU_RTF추가/
│   │   └── 09_회귀전환/
│   ├── AD/                  # 이상탐지 6개
│   │   ├── 01_16피처확장/
│   │   ├── 02_zscore정규화/
│   │   ├── 03_HI기반분리_FINAL/         ⭐ 최고 성능
│   │   ├── 04_HI분리_leakage검증/
│   │   ├── 05_HI기반_16피처복원/
│   │   └── 06_경계좁힘검증/
│   └── CLF/                 # 고장분류 3개
│       ├── 00_기준선/
│       ├── 01_3피처_가능성검증/
│       └── 02_3피처ML학습_DEPLOY/      ⭐ 배포 모델
│
├── reports/                 # 통합 보고서
│   ├── SUMMARY.md                   # 전체 요약
│   ├── RUL_실험보고서.md
│   ├── 이상탐지_실험보고서.md
│   ├── 고장분류_실험보고서.md
│   ├── 검증보고서.md                # Leakage·난이도 검증
│   └── ML_API_스펙_보고서.md        # FastAPI 통합 스펙
│
└── _archive/                # 옛 시도 (참고)
    ├── 2026-03-19_1차ML시도/
    └── 2026-03-20_2차DL시도/
```

---

## 📊 데이터셋 (별도 보관)

> **데이터셋은 본 리포지토리에 포함되지 않는다.** 라이선스·용량 문제.

| 데이터셋 | Bearings | 용도 | 출처 |
|---|---|---|---|
| FEMTO | 11 | RUL/이상탐지 | [PRONOSTIA Challenge 2012](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset) |
| KAIST | 1 | RUL/이상탐지 (진동+온도+기상) | 자체 수집 |
| IMS (NASA) | 4 | RUL/이상탐지 | [NASA Bearing Dataset](https://data.nasa.gov/) |
| Zenodo | 6 | RUL/이상탐지 | [Zenodo](https://zenodo.org/) |
| XJTU-SY | 15 | 고장분류 (RTF) | [XJTU-SY](http://biaowang.tech/xjtu-sy-bearing-datasets/) |
| Paderborn | 32 | 고장분류 (진동+전류+온도) | [Paderborn KAt-Data](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/) |

데이터셋 경로는 `src/data_loaders.py`의 `DATASET_ROOT`에서 설정.

---

## 🚀 실행 방법

### 환경 설정

```bash
git clone https://github.com/hajy-dev/PdM-Bearing-Research.git
cd PdM-Bearing-Research

python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 데이터셋 준비

1. 위 표의 데이터셋 다운로드
2. `src/data_loaders.py`의 `DATASET_ROOT` 경로 수정

### 실행

```bash
cd src

# 전체 실행 (RUL → 분류 → 이상탐지)
python run_all.py

# 개별 실행
python run_all.py rul      # RUL 4등급 분류
python run_all.py clf      # 고장 모드 분류
python run_all.py ad       # 이상 탐지
```

---

## 📈 핵심 실험 흐름

### RUL 개선 단계 (F1-Macro 변화)

| 단계 | 변경 | F1 | Gap |
|---|---|---|---|
| 0 | 기준선 (operating_hours 포함 — **leakage**) | 99.61% (무효) | 0.0% |
| 1 | operating_hours 제거 | 29.43% | 19.0% |
| 2 | 17피처 확장 | 35.80% | 22.1% |
| 3 | min-max 정규화 (실패) | 34.23% | 39.2% |
| 4 | z-score 정규화 | 40.89% | 24.9% |
| 5 | Optuna 과적합 억제 | 41.09% | 14.7% |
| 6 | **HI 기반 라벨링** | 48.66% | 2.4% |
| 7 | **balanced weight** | **50.79%** | **3.3%** |

### 이상탐지 개선 단계 (AUC 변화)

| 단계 | 변경 | AUC | Det Rate |
|---|---|---|---|
| 기준선 | 3피처 | 0.737 | 41.1% |
| 1 | 16피처 확장 | 0.789 | 55.1% |
| 2 | z-score 정규화 | 0.852 | 55.4% |
| 3 | **HI 기반 분리** | **0.988** | **97.2%** |
| 검증 | HI 소스 피처 제거 | 0.984 | 96.4% |

---

## 🔬 방법론

### 평가
- **LOBO-CV** (Leave-One-Bearing-Out, `GroupKFold`) — 베어링 단위 분리로 일반화 검증
- 모든 단계에서 train/test gap (overfitting) 모니터링
- Optuna 60 trials × 3 모델

### 피처 (16개)
- **시간 도메인 (10)**: vibration_rms, std, peak, kurtosis, skewness, crest_factor, impulse_factor, shape_factor, short_trend, rolling_trend
- **주파수 도메인 (6)**: spectral_energy, spectral_centroid, spectral_spread, band_energy_low/mid/high

### 정규화
- **베어링별 z-score** — 개체차 흡수

### 라벨링 (HI 기반 4등급)
| 등급 | HI 범위 |
|---|---|
| LOW | > 0.7 |
| MEDIUM | 0.4 ~ 0.7 |
| HIGH | 0.15 ~ 0.4 |
| CRITICAL | ≤ 0.15 |

---

## 🔗 관련 프로젝트

- **AIpass (S-Traffic Final Project)** — 본 R&D 결과를 FastAPI 추론 서비스로 통합한 풀스택 시스템: https://github.com/hajy-dev/STrafficFinalProject

---

## 📜 라이선스

MIT License — 자유롭게 사용 가능. 데이터셋은 각 출처의 라이선스 따름.

---

## 👤 작성자

**ha.jy** ([@hajy-dev](https://github.com/hajy-dev))
- Tistory: https://connect-tech.tistory.com/
