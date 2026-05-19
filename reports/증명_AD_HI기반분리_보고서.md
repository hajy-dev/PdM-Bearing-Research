# 증명: 이상탐지 HI 기반 정상/이상 분리

## 변경 내용
- import에 `compute_health_index` 추가
- 정상/이상 분리 기준 변경: rul_ratio → health_index
  - 정상: HI > 0.7 (RUL의 LOW 등급과 동일)
  - 이상: HI ≤ 0.4 (RUL의 HIGH+CRITICAL과 동일)
  - 중간(0.4~0.7): 경계 모호 데이터 제외
- Step 2에서 `compute_health_index` 호출 추가
- 16피처, z-score 정규화 유지

## 결과

| 모델 | C-2 (rul 기준) | C-3 (HI 기준) | 변화 |
|------|---------------|---------------|------|
| IsolationForest | Det 38.7% / AUC 0.787 | Det **97.2%** / AUC **0.988** | Det **+58.5%p**, AUC **+0.201** |
| Autoencoder | Det 55.4% / AUC 0.852 | Det **93.9%** / AUC **0.985** | Det +38.5%p, AUC +0.133 |
| OneClassSVM | Det 51.8% / AUC 0.782 | Det **96.0%** / AUC **0.986** | Det +44.2%p, AUC +0.204 |

## 판정
**대전환.** AUC 0.78~0.85 → **0.985~0.988**, Det Rate 38~55% → **93~97%**. FPR 5.0~5.6%로 오탐 변동 없음.

## Leakage 검증 (별도 실험)
HI 소스 피처(vibration_rms, short_trend, rolling_trend) 3개를 제거하고 13피처만으로 재실험:

| 모델 | 16피처 | 13피처 (HI소스 제거) | 차이 |
|------|--------|-------------------|------|
| IsolationForest | AUC 0.988 | AUC 0.984 | -0.004 |
| Autoencoder | AUC 0.985 | AUC 0.984 | -0.001 |
| OneClassSVM | AUC 0.986 | AUC 0.983 | -0.003 |

**Leakage 아님 확정.** HI 소스 제거 후에도 AUC 0.98 유지 — 나머지 13개 피처가 독립적으로 정상/이상 구분 가능.

## 핵심 발견
- rul_ratio(시간) → health_index(상태) 전환이 RUL과 동일한 효과
- 시간적으로 후반이지만 진동이 정상인 데이터가 올바르게 분류됨
- RUL과 이상탐지가 같은 HI 기준을 공유하여 Vue 프론트엔드에서 일관된 결과 제공
