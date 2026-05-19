# 증명: 이상탐지 HI 분리 Leakage 검증

## 변경 내용
- 피처에서 HI 소스 3개 제거: vibration_rms, short_trend, rolling_trend
- 나머지 13개 피처만으로 실험
- HI 기반 분리, z-score 정규화 유지

## 검증 목적
HI = f(vibration_rms, short_trend, rolling_trend)이므로, 이 3개가 피처에 포함되면 모델이 라벨의 정의를 직접 학습하는 순환 구조(leakage) 가능성 검증

## 결과

| 모델 | 16피처 | 13피처 (HI소스 제거) | 차이 |
|------|--------|-------------------|------|
| IsolationForest | Det 97.2% / AUC 0.988 | Det **96.4%** / AUC **0.984** | -0.8%p / -0.004 |
| Autoencoder | Det 93.9% / AUC 0.985 | Det **92.7%** / AUC **0.984** | -1.2%p / -0.001 |
| OneClassSVM | Det 96.0% / AUC 0.986 | Det **94.8%** / AUC **0.983** | -1.2%p / -0.003 |

## 판정
**Leakage 아님.** HI 소스 피처 제거 후에도 AUC 0.983~0.984, Det Rate 92~96%로 거의 동일. 하락폭 0.1~1.2%p에 불과.

## 결론
- 나머지 13개 피처(std, peak, kurtosis, spectral_energy 등)가 독립적으로 정상/이상을 구분하는 충분한 정보 보유
- HI 기반 분리는 물리적으로 정확한 정상/이상 경계를 제공하는 것이지 leakage가 아님
- **16개 피처로 복원하여 RUL과 동일한 피처 구성으로 최종 확정**
