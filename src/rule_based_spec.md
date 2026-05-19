# 예지보전 시스템 — 최종 설계 명세
## Rule-based Bootstrapping → ML 전환 전략

---

## 0. 전체 전략

```
[1단계] 규칙 기반으로 시스템 가동 → sensor_log에 데이터 축적 (1~2일)
[2단계] 축적된 데이터(3피처 + 라벨) export → 경량 ML 모델 학습
[3단계] FastAPI에 ML 모델 배포 → model.predict()로 전환
[4단계] 실제 센서 데이터가 쌓이면 재학습으로 개선
```

**패턴명**: Rule-based Bootstrapping (규칙 기반 부트스트래핑)
- 규칙으로 초기 라벨 생성 (Cold Start 해결)
- 라벨된 데이터로 ML 학습 → 배포
- 학습 데이터 = 배포 데이터 (같은 DB, 분포 일치)

---

## 1. 이상탐지 (Spring Boot → 이후 ML 전환)

### 입력: sensor_log의 3개 값
- vibration (rms)
- temperature
- motor_current

### 임계값 기준
| 센서 | 정상 범위 | 주의(↑) | 위험(↑↑) |
|------|----------|--------|---------|
| vibration | < 0.8 | 0.8 ~ 2.0 | >= 2.0 |
| temperature | < 60 | 60 ~ 80 | >= 80 |
| motor_current | < 20 | 20 ~ 35 | >= 35 |

※ DB 실데이터 참고: 정상(0.45, 42.5, 15.2) ~ 위험(8.20, 88.0, 45.1)

### 이상 판정
- 3개 값 모두 정상 범위 → 정상 (is_anomaly=false)
- 1개 이상 주의/위험 → 이상 (is_anomaly=true, 고장분류로 넘김)

### anomaly_score 계산
```
각 센서의 정상 범위 대비 벗어난 정도를 0~1로 종합
예: vibration=1.85 → (1.85-0.8)/(2.0-0.8) = 0.875
    temperature=55 → 정상 범위 내 = 0.0
    → anomaly_score = max(0.875, 0.0, ...) 또는 가중 평균
```

---

## 2. 고장분류 (Spring Boot → 이후 ML 전환)

### 패턴 조합 규칙

| # | vibration | temperature | motor_current | 진단 | 코드 |
|---|-----------|------------|---------------|------|------|
| 1 | 정상 | 정상 | 정상 | 정상 | NORMAL |
| 2 | 정상(-) | ↑↑ | ↑↑ | 모터 이상 | MOTOR_FAULT |
| 3 | ↑ | ↑ | ↑↑ | 베어링+모터 이상 | COMPOUND_FAULT |
| 4 | ↑↑ | 정상/↑ | ↑ | 베어링 이상 | BEARING_FAULT |
| 5 | 정상(-) | ↑↑ | 정상(-) | 환경변수 (정상) | NORMAL_ENV |
| 6 | ↑↑ | ↑↑ | ↑↑ | 전체 이상 (즉시 점검) | CRITICAL |

---

## 3. RUL (Spring Boot — 계산 기반, ML 불필요)

### 계산 방식
```
정상:     remaining_days = 300 - (현재일 - installation_date).days
고장확정:  remaining_days = 2 (즉시 D-2로 세팅)
정비완료:  remaining_days = 300 (리셋)
```

### risk_level 매핑 (RUL 기반)
```
remaining_days > 200  → LOW
remaining_days > 100  → MEDIUM
remaining_days > 2    → HIGH
remaining_days <= 2   → CRITICAL
```

---

## 4. DB 변경 (최종 확정)

### 신규 테이블: 없음
### equipment — 컬럼 2개 추가
```sql
ALTER TABLE equipment
ADD COLUMN risk_level VARCHAR(10) DEFAULT 'LOW',
ADD COLUMN fault_type VARCHAR(30);
```

| 컬럼 | 용도 |
|------|------|
| risk_level | 장비의 현재 확정 위험등급 (T분 조건 반영) |
| fault_type | 현재 고장 유형 (정상이면 NULL) |

### sensor_log — 컬럼 4개 추가
```sql
ALTER TABLE sensor_log
ADD COLUMN is_anomaly BOOLEAN DEFAULT FALSE,
ADD COLUMN anomaly_score DOUBLE PRECISION,
ADD COLUMN fault_type VARCHAR(30),
ADD COLUMN risk_level VARCHAR(10);
```

| 컬럼 | 용도 |
|------|------|
| is_anomaly | 이 시점에 이상 감지 여부 (true/false) |
| anomaly_score | 이상도 점수 (0.0~1.0) |
| fault_type | 고장 유형 (NORMAL/BEARING_FAULT/MOTOR_FAULT 등) |
| risk_level | 이 시점의 위험 등급 (LOW/MEDIUM/HIGH/CRITICAL) |

### 기존 컬럼과의 관계
- risk_score (기존): 설비 종합 위험도 (RUL + 센서 종합, 0~9.99)
- anomaly_score (신규): 현재 순간의 센서 이상도 (0.0~1.0)
- 역할 분리: risk_score는 "전체적", anomaly_score는 "지금 이 순간"

---

## 5. 전체 흐름

### 1단계: 규칙 기반 운영 (초기)
```
Spring Boot 스케줄러 (15초 주기):
  1. 더미 센서값 생성 (vibration, temperature, motor_current)
  2. 규칙 기반 판단:
     - is_anomaly, anomaly_score 계산
     - fault_type 분류
     - risk_level 결정
  3. sensor_log INSERT (센서값 + 판단 결과 동시)
  4. T분 조건 확인 (메모리에서 연속 이상 횟수 관리):
     → 20회 미달 → 대기
     → 20회 도달 → 고장 확정
  5. 고장 확정 시:
     → equipment.risk_level = 'CRITICAL'
     → equipment.fault_type = 해당 유형
     → equipment.status = '점검요망'
     → remaining_days = 2 (D-2)
     → notification INSERT (정비 알림)
  6. RUL 계산: 300 - (현재일 - installation_date)
  7. 정비 완료 시 (관리자 수동):
     → equipment.status = '정상가동'
     → equipment.risk_level = 'LOW'
     → equipment.fault_type = NULL
```

### 2단계: ML 전환 (1~2일 데이터 축적 후)
```
데이터 축적:
  sensor_log에 1~2일 × 4회/분 × 60분 × 24시간 × 12장비 ≈ 69,000~138,000건

ML 학습:
  SELECT vibration, temperature, motor_current,  -- X (입력)
         is_anomaly, fault_type                  -- y (라벨)
  FROM sensor_log;
  → XGBoost(3피처) 학습 → .joblib 저장

FastAPI 배포:
  model = joblib.load("anomaly_model.joblib")
  → Spring Boot가 호출: POST /predict {vibration, temperature, motor_current}
  → FastAPI 응답: {is_anomaly, anomaly_score, fault_type}
  → Spring Boot가 sensor_log INSERT + equipment UPDATE
```

### 역할 분담

| 역할 | 1단계 (규칙) | 2단계 (ML) |
|------|------------|-----------|
| 더미 데이터 생성 | Spring Boot | Spring Boot |
| 이상탐지 | Spring Boot (if/else) | **FastAPI (model.predict)** |
| 고장분류 | Spring Boot (if/else) | **FastAPI (model.predict)** |
| RUL 계산 | Spring Boot (계산) | Spring Boot (계산) |
| DB INSERT | Spring Boot | Spring Boot |
| T분 조건 | Spring Boot | Spring Boot |
| Vue API | Spring Boot | Spring Boot |

---

## 6. Java 코드 (Spring Boot — 1단계 규칙 기반)

```java
@Service
public class PredictiveService {

    // 임계값 상수
    private static final double VIB_WARN = 0.8;
    private static final double VIB_DANGER = 2.0;
    private static final double TEMP_WARN = 60.0;
    private static final double TEMP_DANGER = 80.0;
    private static final double CUR_WARN = 20.0;
    private static final double CUR_DANGER = 35.0;

    // 이상탐지
    public boolean isAnomaly(double vibration, double temperature, double current) {
        return vibration >= VIB_WARN
            || temperature >= TEMP_WARN
            || current >= CUR_WARN;
    }

    // anomaly_score 계산
    public double calculateAnomalyScore(double vibration, double temperature, double current) {
        double vibScore = Math.max(0, (vibration - VIB_WARN) / (VIB_DANGER - VIB_WARN));
        double tempScore = Math.max(0, (temperature - TEMP_WARN) / (TEMP_DANGER - TEMP_WARN));
        double curScore = Math.max(0, (current - CUR_WARN) / (CUR_DANGER - CUR_WARN));
        return Math.min(1.0, Math.max(vibScore, Math.max(tempScore, curScore)));
    }

    // 고장분류
    public String classifyFault(double vibration, double temperature, double current) {
        boolean vibWarn = vibration >= VIB_WARN;
        boolean vibDanger = vibration >= VIB_DANGER;
        boolean tempWarn = temperature >= TEMP_WARN;
        boolean tempDanger = temperature >= TEMP_DANGER;
        boolean curWarn = current >= CUR_WARN;
        boolean curDanger = current >= CUR_DANGER;

        if (vibDanger && tempDanger && curDanger) return "CRITICAL";
        if (!vibWarn && tempDanger && curDanger) return "MOTOR_FAULT";
        if (vibWarn && tempWarn && curDanger) return "COMPOUND_FAULT";
        if (vibDanger && curWarn) return "BEARING_FAULT";
        if (!vibWarn && tempDanger && !curWarn) return "NORMAL_ENV";
        if (!vibWarn && !tempWarn && !curWarn) return "NORMAL";
        return "MONITORING";
    }

    // RUL 계산
    public int calculateRemainingDays(LocalDate installDate, boolean isCritical) {
        if (isCritical) return 2;
        int elapsed = (int) ChronoUnit.DAYS.between(installDate, LocalDate.now());
        return Math.max(300 - elapsed, 0);
    }

    // risk_level 매핑
    public String getRiskLevel(int remainingDays) {
        if (remainingDays > 200) return "LOW";
        if (remainingDays > 100) return "MEDIUM";
        if (remainingDays > 2) return "HIGH";
        return "CRITICAL";
    }
}
```

---

## 7. 발표 전략

```
[파트 1: AI 연구 역량]
  "공개 데이터셋 6종으로 ML 모델 검증"
  - 이상탐지 AUC 0.977 (16피처 Autoencoder)
  - 고장분류 CNN 86.1%, XGB 81.5%
  - 14개 증명 보고서 (변수 통제 실험)
  - 3피처 한계 실험으로 피처 엔지니어링 중요성 입증

[파트 2: 시스템 구현]
  "Rule-based Bootstrapping 전략"
  - 1단계: 규칙 기반으로 초기 데이터 수집 + 라벨링
  - 2단계: 수집 데이터로 경량 ML 모델 학습 → 배포
  - 실시간 대시보드 데모 (Vue + Spring Boot + FastAPI)

[파트 3: 확장성]
  "실제 센서 연결 시 즉시 확장 가능한 구조"
  - 3피처 ML → 16피처 ML (센서 업그레이드 시)
  - 데이터 축적 → 재학습 → 성능 개선 사이클
```

---

## 8. 변경 없는 것들 (확인)

| 항목 | 상태 |
|------|------|
| equipment_type = 'CCTV' | 유지 |
| status 한글 (정상가동/점검중/점검요망/통신오류) | 유지 |
| maintenance_log | 변경 없음 |
| notification | 변경 없음 |
| sensor_log 기존 컬럼 | 변경 없음 |
| prediction_log 신규 테이블 | 불필요 (sensor_log 확장) |
