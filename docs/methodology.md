# 방법론

> 본 R&D의 평가·검증 방법론 종합.

---

## 1. LOBO-CV (Leave-One-Bearing-Out)

### 왜 필요한가
- 일반 K-Fold: 같은 베어링의 윈도우 데이터가 train·test에 동시 존재 → **그룹 누출(group leakage)**
- 결과: F1 99.61% 같은 비현실적 성능

### 어떻게 적용
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=10)
for train_idx, test_idx in gkf.split(X, y, groups=bearing_ids):
    # 같은 베어링이 한 fold에만 들어감
```

### 영향
- baseline F1 99.61% → 29.43% (진짜)
- 차이 70%p는 leakage 환상

---

## 2. Health Index (HI) 기반 라벨링

### 시간 기반 라벨의 한계
- `operating_hours` 기준은 베어링 개체차 무시
- 같은 시간 학습 = 같은 상태 ? **❌** (어떤 베어링은 100h만에 고장, 어떤 건 300h)

### HI 4등급 분리
| 등급 | HI 범위 | 의미 |
|---|---|---|
| LOW | > 0.7 | 정상 |
| MEDIUM | 0.4 ~ 0.7 | 주의 |
| HIGH | 0.15 ~ 0.4 | 위험 |
| CRITICAL | ≤ 0.15 | 임박 |

### Leakage 회피
HI를 라벨로 쓰면서 **HI 산출 피처(rms 등)를 모델 입력에 두면 leakage** → 입력에서 HI 소스 제거.

---

## 3. 피처 엔지니어링 (16개)

### 시간 도메인 (10)
- `vibration_rms` — 에너지 크기
- `std`, `peak` — 변동성·최댓값
- `kurtosis`, `skewness` — 분포 형태
- `crest_factor`, `impulse_factor`, `shape_factor` — 무차원 비율
- `short_trend`, `rolling_trend` — 추세

### 주파수 도메인 (6)
- `spectral_energy` — 전체 에너지
- `spectral_centroid` — 무게중심 주파수
- `spectral_spread` — 분포 폭
- `band_energy_low/mid/high` — 3대역 에너지

### 슬라이딩 윈도우
- `WINDOW_SIZE = 256`, `STEP = 128` (50% overlap)

---

## 4. 정규화

### 베어링별 z-score
- 베어링마다 진동 특성이 다름 → 글로벌 정규화는 정보 손실
- 베어링 ID로 그룹화 후 각 그룹별 `fit_transform`

```python
for bearing_id in df['bearing_id'].unique():
    mask = df['bearing_id'] == bearing_id
    df.loc[mask, feature_cols] = StandardScaler().fit_transform(
        df.loc[mask, feature_cols]
    )
```

### Min-Max는 왜 실패했나
- 이상치에 매우 민감 → 진동 spike가 전체 분포를 왜곡

---

## 5. 하이퍼파라미터 최적화

### Optuna (TPE)
- 60 trials × 3 모델 (XGBoost, LightGBM, Random Forest)
- 탐색 범위 — 과적합 억제 방향:
  - `max_depth`: 3~6 (얕게)
  - `reg_alpha`, `reg_lambda`: 0.1~50 (정규화 강하게)
- Pruning으로 가망 없는 trial 조기 종료

---

## 6. 불균형 데이터 처리

### compute_sample_weights
```python
from sklearn.utils.class_weight import compute_sample_weight

sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)
```

- 소수 클래스 (CRITICAL, HIGH)에 큰 가중치
- 7단계 적용 후 F1 48.66% → 50.79%

---

## 7. 평가 지표

### RUL (분류)
- **F1-Macro** — 클래스별 F1의 단순 평균 (불균형에 강함)
- **Train/Test Gap** — 과적합 모니터링

### 이상탐지
- **AUC (ROC-AUC)** — 임계값 무관 종합 성능
- **Detection Rate** — 진짜 이상 중 잡은 비율
- **False Alarm Rate** — 정상 중 잘못 본 비율

### 고장 분류
- **Accuracy**, **F1-Macro**
- **혼동행렬** — 어떤 클래스 혼동되는지 확인

---

## 8. 변수 통제 실험 원칙

1. **한 번에 한 변수만** 변경
2. 각 변경마다 **별도 폴더** (`증명_*` 또는 `experiments/`)
3. **report.md** 에 가설·변경·결과·결론 명시
4. **이전 단계와 비교** (F1·Gap 변화)
5. 실패도 보존 (`07_사분위실패_되돌림`처럼)
