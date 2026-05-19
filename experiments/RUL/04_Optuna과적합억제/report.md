# 증명: Optuna 과적합 억제 + RandomForest Optuna

## 변경 내용
- XGBoost Optuna 탐색 범위 제한: max_depth 3~6, reg 0.1~50, subsample 0.5~0.8, colsample 0.3~0.7, min_child_weight 5~30
- LightGBM Optuna 탐색 범위 제한: max_depth 3~6, num_leaves 7~31, min_child_samples 10~50
- RandomForest Optuna 신규 추가 (기존 하드코딩 제거): max_depth 3~8, min_leaf 10~50, max_features 0.3~0.7
- z-score 정규화, rul_ratio 기반 라벨 유지

## 결과

### 실험 A (22 bearings)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGBoost | 47.92% | 41.09% | **19.3%** |
| LightGBM | 46.81% | 40.58% | 27.3% |
| RandomForest | 48.72% | 38.79% | **14.7%** |

### 실험 B (환경보정)
| 모델 | Accuracy | F1-Macro | Gap |
|------|----------|----------|-----|
| XGB+Env | 46.31% | 36.76% | 34.6% |
| XGB VibOnly | 47.75% | 41.08% | **13.6%** |
| 환경보정 효과 | — | -1.44%p | — |

## 판정
**Gap 대폭 개선.** XGBoost Gap 32.8%→19.3%, RandomForest Gap 34.1%→14.7%. F1은 미세 변동, Acc는 유지/소폭 상승.

## 핵심 발견
- RandomForest: 하드코딩(max_depth=12, min_leaf=5) → Optuna로 교체한 효과가 가장 큼 (Gap -19.4%p)
- LightGBM: 이미 내부 정규화가 강해서 탐색 범위 제한이 오히려 역효과
- VibOnly Gap 13.6%가 전체 실험 최저 — 과적합 없이 일반화 달성
