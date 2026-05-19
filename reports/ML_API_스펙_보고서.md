# ML 모델 API 스펙 보고서
> finalproject 팀 공유용 | 작성일: 2026-04-01

---

## 1. 시스템 통신 흐름

```
Spring Boot (15초 주기)
  → GET /api/equipment/sensor-data → sensor_log에서 vibration, temperature 조회
  → POST (FastAPI prediction_loop이 pull)
  
FastAPI (내부 prediction_loop)
  → sensor_log.vibration(rms) 수신
  → 피처 파생 (rms → 16개)
  → 3개 모델 순차 실행
  → POST /api/equipment/predict-result → Webhook push

Spring Boot
  → 결과 DB 저장 + T분 지속 판단 + 상태 전환
```

---

## 2. RUL 예측 모델 (위험도 4등급)

### 입력 (Spring Boot → FastAPI)
```json
{
  "deviceId": "CAM05",
  "vibration": 3.5,
  "temperature": 58.5,
  "recordedAt": "2026-04-01T15:30:00"
}
```
- `vibration`: sensor_log.vibration (RMS 스칼라, 단위: g)
- `temperature`: sensor_log.temperature (장비 온도, 단위: 도)
- FastAPI 내부에서 rms 기반 16개 피처 파생 후 모델 입력

### 출력 (FastAPI → Spring Boot Webhook)
```json
{
  "deviceId": "CAM05",
  "riskLevel": "HIGH",
  "healthIndex": 0.28,
  "confidence": 0.82,
  "timestamp": "2026-04-01T15:30:15"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| riskLevel | String | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| healthIndex | Float (0~1) | 건강 지수 (1=정상, 0=고장) |
| confidence | Float (0~1) | 예측 확신도 |

### 등급 기준
| 등급 | Health Index | 의미 |
|------|-------------|------|
| LOW | > 0.7 | 정상 |
| MEDIUM | 0.4 ~ 0.7 | 주의 관찰 |
| HIGH | 0.15 ~ 0.4 | 점검 권고 |
| CRITICAL | <= 0.15 | 즉시 교체 |

### 모델 정보
- 알고리즘: XGBoost
- 검증 성능: F1-Macro 50.79%, Overfit Gap 3.3%
- 평가: LOBO-CV (Leave-One-Bearing-Out, 10-fold)

---

## 3. 이상탐지 모델

### 입력
RUL과 동일 (같은 prediction_loop 내에서 같은 피처 사용)

### 출력 (통합 Webhook에 포함)
```json
{
  "isAnomaly": true,
  "anomalyScore": 0.087,
  "anomalyThreshold": 0.065
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| isAnomaly | Boolean | 이상 여부 |
| anomalyScore | Float | Autoencoder 재구성 오차 |
| anomalyThreshold | Float | 판정 임계값 |

### 판정 기준
- 정상: Health Index > 0.7
- 이상: Health Index <= 0.6
- T분 연속 이상 시 상태 전환 (Spring Boot에서 판단)

### 모델 정보
- 알고리즘: Autoencoder (Keras)
- 검증 성능: AUC 0.977, Detection Rate 87.9%, FPR 5.3%

---

## 4. 고장분류 모델 (Colab 검증 중)

### 입력
별도 논의 필요 (FFT 기반 135피처 — rms 파생과 다른 구조)

### 출력 (예상)
```json
{
  "faultType": "inner_race",
  "faultConfidence": 0.78,
  "faultProbabilities": {
    "ball": 0.05,
    "cage": 0.08,
    "inner_race": 0.78,
    "outer_race": 0.09
  }
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| faultType | String | `ball` / `cage` / `inner_race` / `outer_race` |
| faultConfidence | Float (0~1) | 최고 확률 클래스의 확신도 |
| faultProbabilities | Object | 클래스별 확률 |

### 모델 정보 (기준선)
- 알고리즘: CNN-1D (Raw signal 기반)
- 검증 성능: Accuracy 86.1%, F1-Macro 86.0%, Gap 3.1%
- Colab Optuna 최적화 진행 중

---

## 5. 통합 Webhook 응답 (최종)

FastAPI → Spring Boot `POST /api/equipment/predict-result`

```json
{
  "deviceId": "CAM05",
  "riskLevel": "HIGH",
  "healthIndex": 0.28,
  "confidence": 0.82,
  "isAnomaly": true,
  "anomalyScore": 0.087,
  "anomalyThreshold": 0.065,
  "faultType": "inner_race",
  "faultConfidence": 0.78,
  "tempResidual": 3.2,
  "timestamp": "2026-04-01T15:30:15"
}
```

### 환경보정 참고 지표 (대시보드 표시용)
| 필드 | 설명 | 계산 |
|------|------|------|
| tempResidual | 외기 대비 장비 온도 잔차 | device_temp - (ambient_temp + baseline_heat) |

- tempResidual ≈ 0: 정상 발열
- tempResidual > 5: 비정상 발열 의심

---

## 6. 대시보드에서 필요한 데이터 흐름

```
Vue 화면 갱신 (15초 Polling):
  GET /api/equipment/list → Spring Boot → DB 조회
  
응답 예시:
[
  {
    "equipmentId": 5,
    "intersectionName": "신당교차로",
    "status": "점검요망",
    "vibration": 3.5,
    "temperature": 58.5,
    "riskLevel": "HIGH",
    "healthIndex": 0.28,
    "isAnomaly": true,
    "faultType": "inner_race",
    "tempResidual": 3.2,
    "lastUpdated": "2026-04-01T15:30:15"
  },
  ...
]
```

---

## 7. 미확정 사항

| 항목 | 상태 | 비고 |
|------|------|------|
| 고장분류 배포 방식 | 미결정 | Colab 결과 후 논의 |
| prediction_loop 주기 | 15초 (잠정) | 조정 가능 |
| T분 지속 시간 | 미결정 | Spring Boot에서 설정 |
| RUL 일 단위 표시 | 미결정 | healthIndex → D-N 변환 로직 필요 |

---

*AI-PASS 예지보전 ML팀 | 방안 B (rms 기반 피처 파생) 확정*
