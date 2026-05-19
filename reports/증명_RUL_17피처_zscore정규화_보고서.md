# 증명: 17피처 + 베어링별 z-score 정규화

## 변경 내용
- min-max → z-score 변경: `(x - x.mean()) / x.std()`
- health_index 제외 16개 피처 정규화
- 라벨: rul_ratio 기반

## 결과

### 실험 A (22 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 46.17% | 40.40% | 32.8% |
| LightGBM | 47.43% | 40.89% | 24.9% |
| RandomForest | 47.23% | 40.19% | 34.1% |

### 실험 B (환경보정)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 45.43% | 36.07% | 39.6% |
| XGB VibOnly | 48.23% | 41.35% | 28.0% |
| 환경보정 효과 | — | -2.80%p | — |

## 판정
**성공.** min-max 대비 F1 +6~7%p 회복, Gap 대폭 개선. 이상치에 강건한 z-score가 RTF 데이터에 적합.

## 핵심 수치
- F1 40.89% — 당시 최고 기록
- Accuracy는 17피처(정규화 없음) 수준 유지하면서 F1만 대폭 상승
- 소수 클래스(HIGH, CRITICAL) 분류 개선 확인
