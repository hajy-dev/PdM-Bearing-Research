# 증명: 이상탐지 베어링별 z-score 정규화

## 변경 내용
- 피처 정의 후, 정상/이상 분리 전에 베어링별 z-score 정규화 추가
- `(x - x.mean()) / max(x.std(), 1e-10)` — bearing_id 그룹별 적용
- 16개 피처 전체 정규화
- 정상/이상 기준: rul_ratio 기반 변경 없음

## 결과

| 모델 | C-1 (16피처) | C-2 (z-score) | 변화 |
|------|-------------|---------------|------|
| IsolationForest | Det 47.7% / AUC 0.746 | Det 38.7% / AUC **0.787** | Det -9.0%p, AUC +0.041 |
| Autoencoder | Det 55.1% / AUC 0.775 | Det **55.4%** / AUC **0.852** | Det +0.3%p, AUC **+0.077** |
| OneClassSVM | Det 53.2% / AUC 0.789 | Det 51.8% / AUC 0.782 | Det -1.4%p, AUC -0.007 |

## 판정
**Autoencoder에서 효과적.** AUC 0.775→0.852 대폭 상승. IsolationForest는 AUC 상승했으나 Det Rate 하락 — contamination 파라미터(0.05 고정)가 z-score 분포와 미스매치. OneClassSVM은 변동 미미.

## 핵심 발견
- AUC(종합 판별력) 상승 + Det Rate(특정 임계값) 하락은 모순이 아님
- AUC가 높으면 임계값 조정으로 Det Rate를 끌어올릴 수 있음
- Autoencoder의 재구성 오차 기반 탐지가 z-score 정규화와 시너지
