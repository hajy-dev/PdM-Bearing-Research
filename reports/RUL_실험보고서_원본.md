# RUL 위험도 4등급 분류 — 실험 보고서

## 1. 실험 개요

- **목적**: 베어링 진동 데이터로 현재 위험도 등급(LOW/MEDIUM/HIGH/CRITICAL)을 예측
- **데이터**: FEMTO(11) + KAIST(1) + IMS(4) + Zenodo(6) = 22 bearings, ~49,700 rows
- **평가**: LOBO-CV (Leave-One-Bearing-Out, GroupKFold 10-fold)
- **모델**: XGBoost, LightGBM, RandomForest (모두 Optuna 최적화)

---

## 2. 실험 이력

### 기준선 (backup_v2)

- 피처: vibration_rms, short_trend, rolling_trend, health_index, **operating_hours** (5개)
- 라벨: rul_ratio 기반 4등급
- 결과: RandomForest Acc **99.99%**, F1 **99.99%**
- **판정: Data Leakage** — operating_hours = 1 - rul_ratio (정답의 역함수)

### 단계별 개선

| 단계 | 변경 내용 | 최고 F1 | Gap | 판정 | 백업 |
|------|----------|--------|-----|------|------|
| 1 | operating_hours 제거 (4피처) | 29.43% | 19.0% | leakage 증명 | 증명_RUL_operating_hours제거 |
| 2 | 17피처 확장 (시간통계7+주파수6 추가) | 35.80% | 22.1% | +6.4%p 개선 | — |
| 3 | 베어링별 min-max 정규화 | 34.23% | 39.2% | **실패** (이상치에 취약) | 증명_RUL_17피처_minmax정규화 |
| 4 | 베어링별 z-score 정규화 | **40.89%** | 24.9% | +5.1%p 개선 | 증명_RUL_17피처_zscore정규화 |
| 5 | Optuna 과적합 억제 + RF Optuna | 41.09% | **14.7%** | Gap 대폭 개선 | 증명_RUL_Optuna과적합억제 |
| 6 | HI 기반 라벨링 (피처에서 HI 제거, 16피처) | 48.66% | 2.4% | 대전환 | 증명_RUL_HI기반라벨링 |
| 7 | balanced weight (sample_weight) | **50.79%** | **3.3%** | **최고 기록** | 증명_RUL_HI라벨_balanced |
| 8 | 사분위 기반 라벨 기준 | 42.50% | 37.7% | **실패** (노이즈 학습) | 증명_RUL_사분위실패_되돌림 |
| 9 | XJTU-SY 15 bearings RTF 추가 (37 bearings) | 46.29% | 8.2% | **실패** (데이터 이질성) | 증명_RUL_XJTU_RTF추가 |

---

## 3. 최종 최적 설정 (7단계)

### 구성

| 항목 | 설정 |
|------|------|
| 데이터 | 22 bearings (FEMTO+KAIST+IMS+Zenodo) |
| 라벨 | Health Index 기반 4등급: LOW(>0.7), MEDIUM(0.4~0.7), HIGH(0.15~0.4), CRITICAL(≤0.15) |
| 피처 (16개) | vibration_rms, short_trend, rolling_trend, vibration_std, vibration_peak, vibration_kurtosis, vibration_skewness, crest_factor, impulse_factor, shape_factor, spectral_energy, spectral_centroid, spectral_spread, band_energy_low, band_energy_mid, band_energy_high |
| 정규화 | 베어링별 z-score (health_index 제외 — 라벨로 사용) |
| Optuna | 과적합 억제 방향 (max_depth 3~6, reg 0.1~50, subsample 0.5~0.8, colsample 0.3~0.7) |
| 클래스 균형 | compute_sample_weights (클래스별 역빈도 가중치) |

### 최종 성능

**실험 A: 전체 22 bearings**

| 모델 | Accuracy | F1-Macro | Overfit Gap |
|------|----------|----------|-------------|
| XGBoost | 92.21% | 49.42% | 5.7% |
| LightGBM | 92.30% | 46.43% | 6.3% |
| RandomForest | 93.43% | 48.80% | 3.3% |

**실험 B: 환경보정 (Zenodo+KAIST, 7 bearings)**

| 모델 | Accuracy | F1-Macro | Overfit Gap |
|------|----------|----------|-------------|
| XGB+Env | 85.66% | **50.79%** | 12.2% |
| XGB VibOnly | 85.40% | 44.84% | 13.5% |
| 환경보정 효과 | — | **+5.95%p** | — |

---

## 4. 핵심 발견

### Data Leakage 발견 및 수정
- `operating_hours = file_idx / max(file_idx)` = `1 - rul_ratio` (정답의 역함수)
- 모든 데이터셋에서 수학적으로 증명, 제거 후 99.99% → 40%로 하락하여 확인

### HI 기반 라벨링의 효과
- rul_ratio(시간 기반) → health_index(상태 기반)로 전환
- 피처와 라벨이 같은 물리적 도메인에 있어 베어링 간 일반화 성공
- Overfit Gap: 19~28% → 2~6%로 극적 개선

### 환경보정 효과
- leakage 있을 때: -0.05%p (효과 없음으로 보임)
- leakage 제거 후: +5.95%p (진짜 효과 드러남)

### HI 분포 편향
- 전체 데이터의 98.1%가 LOW(HI > 0.7)
- MEDIUM 1.4%, HIGH 0.4%, CRITICAL 0.1%
- balanced weight로 보정했으나 소수 클래스 데이터 절대 부족이 F1 50% 한계의 근본 원인

---

## 5. 한계 및 향후 과제

- F1 50%는 4등급 랜덤(25%) 대비 2배이나 실용 기준(70%+)에 미달
- HI 분포 극도 편향으로 소수 클래스 학습 데이터 부족 (946개 / 49,700)
- XJTU-SY 추가 시도했으나 데이터 이질성으로 실패 — 도메인 적응 필요
- Vue 프론트엔드 표시 시 이상탐지/고장분류와 결합하여 보완 가능성 검토 필요
