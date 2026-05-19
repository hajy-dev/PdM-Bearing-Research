# 증명: 3피처 가능성 검증 (Paderborn)

## 목적
DB의 3개 센서값(vibration_rms, temperature, current_rms)만으로
이상탐지 + 고장분류가 가능한지 검증

## 배경
- 기존 실험: raw 진동 신호에서 추출한 16피처로 학습 (AUC 0.977, CNN 86.1%)
- DB 현실: sensor_log에 vibration(rms 1개), temperature, motor_current만 존재
- 16피처 중 13개는 raw 신호 없이 계산 불가 → 학습-배포 괴리 발견
- 사용자 제안: DB 구조에 맞게 3피처로 학습-배포 일치시키자

## 데이터
- Paderborn 32 bearings (유일하게 vibration + temperature + current 전부 보유)
- healthy: 6개(961,517 rows), outer_race: 12개(1,922,124), inner_race: 11개(1,764,700), ball: 3개(481,268)
- 총 5,129,609 rows
- 평가: StratifiedGroupKFold 5-fold (bearing 단위 분할, 라벨 균등 배치)
- 정규화 미적용 (XGBoost 트리 모델은 스케일 불변)

## 변경 내용
- `verify_3features.py` 신규 작성
- Paderborn 데이터에서 3피처(vibration_rms, temperature, current_rms) 추출
- 실험 1: 이상탐지 (healthy vs fault, 2클래스)
- 실험 2: 세부 고장분류 (inner_race / outer_race / ball, 3클래스)

## 결과

### 피처 분포 (고장 유형별)
| 피처 | healthy | inner_race | outer_race | ball |
|------|---------|-----------|-----------|------|
| vibration_rms | 0.258±0.143 | 0.270±0.159 | 0.271±0.164 | 0.472±0.355 |
| temperature | 46.6±2.6 | 45.4±6.5 | 48.2±3.1 | 48.8±3.5 |
| current_rms | 1.512±0.427 | 1.509±0.431 | 1.507±0.434 | 1.514±0.430 |

**핵심 발견**: inner_race(0.270)와 outer_race(0.271)의 rms 차이 0.001. current_rms는 4클래스 전부 동일(~1.51).

### 실험 1: 이상탐지 (healthy vs fault)
| 지표 | 값 |
|------|-----|
| Mean Acc | 0.7736 ± 0.0573 |
| Mean F1 | 0.8692 ± 0.0374 |
| Mean AUC | **0.5642 ± 0.1335** |

- AUC 0.56 = 랜덤(0.5) 수준. 사실상 구분 능력 거의 없음
- Acc/F1이 높아 보이는 이유: 데이터 81%가 fault → 전부 fault로 찍어도 Acc 81%
- Feature Importance: temperature(0.53) > vibration_rms(0.30) > current_rms(0.17)

### 실험 2: 세부 고장분류 (inner/outer/ball)
| 지표 | 값 |
|------|-----|
| Mean Acc | 0.3503 ± 0.0998 |
| Mean F1-Macro | **0.2478 ± 0.0821** |

| 클래스 | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| inner_race | 0.3205 | 0.3135 | 0.3170 |
| outer_race | 0.4005 | 0.4809 | 0.4370 |
| ball | 0.0056 | 0.0016 | **0.0024** |

- Acc 35% < 랜덤 33% 수준. 모델이 아무것도 배우지 못함
- ball F1 = 0.0024 (사실상 0)
- Feature Importance: temperature(0.52) > vibration_rms(0.39) > current_rms(0.09)

### 실험 3: Autoencoder 이상탐지 (정상 데이터만 학습)
- 방식: 정상(healthy) 베어링으로만 학습 → 복원 오차로 이상 판단
- 평가: Leave-One-Healthy-Bearing-Out (6 folds)

| 지표 | 값 |
|------|-----|
| Mean AUC | **0.6105 ± 0.0758** |
| Mean Acc | 0.3839 ± 0.0992 |
| Mean F1 | 0.4093 ± 0.1516 |
| Mean Precision | 0.9588 ± 0.0309 |
| Mean Recall | 0.2719 ± 0.1208 |

- AUC 0.61: XGBoost(0.56)보다 +0.046 향상, 여전히 랜덤(0.5)에 가까움
- Precision 0.96: 이상이라고 한 건 거의 맞지만, Recall 0.27로 실제 이상의 73%를 놓침

### 3피처 이상탐지 모델 비교
| 모델 | AUC | 판정 |
|------|-----|------|
| XGBoost | 0.5642 | 랜덤 수준 |
| Autoencoder | 0.6105 | 약간 나음, 여전히 부족 |
| **16피처 Autoencoder (기존)** | **0.977** | 우수 |

**결론: XGBoost → Autoencoder로 바꿔도 3피처 한계를 극복할 수 없음**

---

## 근본 원인 분석

### 1. 피처 정보 부족 (가장 큰 원인)
- inner_race와 outer_race의 rms가 0.001 차이 → rms 하나로 구분 불가능
- current_rms는 4클래스 전부 동일 → 분류에 기여 0
- 고장 유형 구분에는 주파수 도메인 정보(FFT, envelope)가 필수

### 2. 베어링 다양성 부족
- 32개 베어링 (healthy 6개뿐)
- 한 베어링에서 수만 row가 나오지만 같은 베어링 내 데이터는 거의 동일
- GroupKFold에서 unseen bearing 예측이 어려움

### 3. 데이터 양 자체는 충분
- 5백만 rows는 학습에 충분
- 양이 아니라 피처의 정보량이 문제

## 판정
- **이상탐지**: AUC 0.56 — 3피처 ML 학습으로는 제한적. 규칙 기반(임계값)이 더 적합
- **세부 고장분류**: Acc 35%, F1 25% — **3피처로 불가능 확정**
- **대분류(베어링/모터/환경)**: 피처 패턴 조합 규칙으로 가능 (ML 불필요)

## 결론
DB의 3개 값으로는 **세부 고장분류(inner/outer/ball)가 물리적으로 불가능**함을 실험으로 증명.
사용자가 제안한 규칙 기반 대분류(베어링이상/모터이상/환경변수)가 3피처의 현실적 최대치.
이 실험은 "피처 엔지니어링의 중요성"을 숫자로 입증한 결과로, 발표 자료에 활용 가능.

## 이전 실험 대비
| 실험 | 피처 | 이상탐지 | 고장분류 |
|------|------|---------|---------|
| 기존 16피처 | raw 신호 기반 16개 | AUC 0.977 | CNN 86.1% |
| **이번 3피처** | **DB 값 3개** | **AUC 0.56** | **Acc 35%** |
| 차이 | 13개 피처 손실 | **-0.41** | **-51%p** |
