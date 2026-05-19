# 증명: XJTU-SY 15 bearings RTF 추가 (22→37 bearings)

## 변경 내용
- `data_loaders.py`에 `load_xjtu_rtf()` 함수 추가
- XJTU-SY 15개 베어링의 전체 수명 데이터를 파일별 피처로 변환
- Horizontal + Vertical 축 결합 (combined = sqrt(H²+V²)), fs=25.6kHz
- `load_all_rtf()`에 등록하여 22→37 bearings 확장
- 기존 `load_xjtu()`(분류용)는 변경 없음

## 결과

### 실험 A (37 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 91.30% | 46.29% | 8.2% |
| LightGBM | 91.84% | 45.67% | 7.6% |
| RandomForest | 89.65% | 42.09% | 5.8% |

### 실험 B (환경보정, 7 bearings — 변동 없음)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 85.62% | 51.13% | 12.4% |
| XGB VibOnly | 85.49% | 47.50% | 14.1% |
| 환경보정 효과 | — | +0.13%p | — |

### 이전 대비 (22 bearings → 37 bearings)
| 모델 | F1 변화 | Gap 변화 |
|------|--------|---------|
| XGBoost | 49.42% → 46.29% (**-3.1%p**) | 5.7% → 8.2% |
| LightGBM | 46.43% → 45.67% (-0.8%p) | 6.3% → 7.6% |
| RandomForest | 48.80% → 42.09% (**-6.7%p**) | 3.3% → 5.8% |

## 판정
**실패.** 데이터를 68% 늘렸으나 성능 오히려 하락. 되돌림 완료 (load_all_rtf에서 제거).

## 실패 원인
- XJTU-SY는 가속 수명 실험(2100~2400rpm, 10~12kN)으로 기존 4개 데이터셋과 운전 조건이 이질적
- LOBO-CV에서 이질적 베어링이 test fold에 빠지면 예측 불가
- 단순 데이터 확장은 효과 없음 — 도메인 적응(domain adaptation) 필요
- `load_xjtu_rtf()` 함수는 코드에 잔존 (향후 활용 가능)
