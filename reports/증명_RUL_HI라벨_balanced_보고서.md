# 증명: HI 라벨 + balanced weight (최적 상태)

## 변경 내용
- `compute_sample_weights` 함수 추가: 클래스별 역빈도 가중치 계산
- Optuna 3개 함수(XGBoost, LightGBM, RF) + evaluate_model에서 `sample_weight` 적용
- HI 기반 라벨, 16피처, z-score 정규화, Optuna 억제 모두 유지

## 결과

### 실험 A (22 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 92.21% | **49.42%** | 5.7% |
| LightGBM | 92.30% | 46.43% | 6.3% |
| RandomForest | 93.43% | 48.80% | **3.3%** |

### 실험 B (환경보정)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 85.66% | **50.79%** | 12.2% |
| XGB VibOnly | 85.40% | 44.84% | 13.5% |
| 환경보정 효과 | — | **+5.95%p** | — |

## 판정
**전체 실험 최고 기록.** F1 50.79% (XGB+Env), Gap 3.3% (RF). 이 설정이 RUL 최적 상태.

## 핵심 개선
- RandomForest F1: 31.60% → **48.80%** (+17.2%p) — balanced weight 효과 가장 극적
- 환경보정 효과: +0.63%p → **+5.95%p** — balanced로 소수 클래스가 살아나면서 환경변수 기여도 증가
- Accuracy 소폭 하락(-1~3%p)은 소수 클래스 예측 강화의 자연스러운 트레이드오프

## HI 분포 참고
| 등급 | 범위 | 데이터 비율 |
|------|------|-----------|
| LOW | HI > 0.7 | 98.1% (48,721개) |
| MEDIUM | 0.4~0.7 | 1.4% (685개) |
| HIGH | 0.15~0.4 | 0.4% (200개) |
| CRITICAL | ≤0.15 | 0.1% (61개) |
