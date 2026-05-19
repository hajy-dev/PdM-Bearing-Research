# 증명: operating_hours 제거 (Data Leakage 검증)

## 변경 내용
- `compare_rul_models.py`에서 `operating_hours` 피처 제거
- 피처: vibration_rms, short_trend, rolling_trend, health_index (4개)
- 라벨: rul_ratio 기반, 정규화/Optuna 변경 없음

## 증명 목적
`operating_hours = file_idx / max(file_idx) = 1 - rul_ratio` 수학적으로 정답의 역함수임을 실험으로 증명

## 결과

### 실험 A (22 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 39.73% | 28.62% | 27.7% |
| LightGBM | 42.18% | 29.43% | 19.0% |
| RandomForest | 40.69% | 27.39% | 25.9% |

### 실험 B (환경보정)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 43.58% | 30.89% | 36.7% |
| XGB VibOnly | 41.83% | 33.17% | 26.6% |
| 환경보정 효과 | — | +1.75%p | — |

## 판정
**Leakage 확정.** 기준선 99.99% → 40%로 약 60%p 하락. operating_hours 하나가 사실상 정답지 역할.

## 부수 발견
- 환경보정 효과가 +1.75%p로 양수 전환 (기준선에서는 -0.05%p)
- leakage가 환경보정의 진짜 효과를 가리고 있었음
