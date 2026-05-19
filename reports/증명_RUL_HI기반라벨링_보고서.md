# 증명: Health Index 기반 라벨링

## 변경 내용
- 라벨 기준 변경: rul_ratio(시간) → health_index(상태)
  - LOW: HI > 0.7, MEDIUM: 0.4~0.7, HIGH: 0.15~0.4, CRITICAL: ≤0.15
- health_index를 피처에서 제거 (leakage 방지) → 17→16개 피처
- `feature_engineering.py`에 `hi_to_severity`, `add_severity_labels_hi` 추가
- `compare_rul_models.py` import 및 라벨 생성 변경
- z-score 정규화, Optuna 억제 유지

## 결과

### 실험 A (22 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 94.32% | 46.22% | **5.6%** |
| LightGBM | 93.76% | **48.66%** | 6.2% |
| RandomForest | 96.43% | 31.60% | **2.4%** |

### 실험 B (환경보정)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 85.89% | 42.85% | 13.9% |
| XGB VibOnly | 85.26% | 41.57% | 14.7% |
| 환경보정 효과 | — | +0.63%p | — |

## 판정
**대전환.** Accuracy 47%→94%, Gap 19%→5.6%. 피처와 라벨이 같은 물리적 도메인(진동 상태)에 있어 베어링 간 일반화 성공.

## 주의사항
- Accuracy 94~96%는 HI 분포 편향의 영향 (98.1%가 LOW)
- RandomForest F1 31.6% — Acc는 높지만 소수 클래스를 전혀 못 맞힘
- F1-Macro가 진짜 성능 지표, Accuracy는 과대평가
