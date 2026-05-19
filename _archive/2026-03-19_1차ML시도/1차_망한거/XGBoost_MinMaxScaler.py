"""
AI-Pass 예지보전 XGBoost 모델 학습
====================================
입력 : combined_train.csv (SMOTE 균등 분포)
검증 : combined_test.csv  (원본 분포, 처음 보는 베어링)
출력 :
  D:/project/예지보전/output_v5/
    ├── xgboost_rul.json
    ├── training_report.xlsx
    └── evaluation_report.xlsx

실행: python train_xgboost.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import time

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path(r"D:\project\예지보전\output_v6")
TRAIN_CSV    = OUTPUT_DIR / "combined_train.csv"
TEST_CSV    = OUTPUT_DIR / "kaist_test.csv"
RUL_SCALER   = OUTPUT_DIR / "rul_scaler.pkl"
MODEL_PATH   = OUTPUT_DIR / "xgboost_rul.json"
REPORT_PATH  = OUTPUT_DIR / "training_report.xlsx"
EVAL_PATH    = OUTPUT_DIR / "evaluation_report.xlsx"

BASE_LIFE_DAYS = 180

FEATURE_COLS = [
    "vibration_rms",
    "vibration_kurtosis",
    "vibration_crest",
    "vibration_peak",
    "vibration_skewness",
    "temperature",
    "temp_residual",
    "motor_current",
    "operating_hours",
    "ambient_temp",
    "wind_speed",
    "humidity",
    "season",
]

XGBOOST_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 6,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "tree_method"      : "hist",   # CPU 빠른 학습
}

# ══════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════

def rul_to_risk(rul: float) -> str:
    if rul >= 31: return "LOW"
    if rul >= 16: return "MEDIUM"
    if rul >=  3: return "HIGH"
    return "CRITICAL"


def print_dist(label: str, risks: list):
    dist  = Counter(risks)
    total = len(risks)
    print(f"\n{label}: {total:,}개")
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>7,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 1. 데이터 로딩
# ══════════════════════════════════════════════════════════════

def load_data():
    print("\n" + "=" * 60)
    print("STEP 1. 데이터 로딩")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)
    test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].fillna(0)

    # 학습 대상: rul_norm (0~1)
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["rul_norm"].values

    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df["rul_norm"].values

    # 일 단위도 보관 (평가용)
    y_test_days = test_df["rul_days"].values

    print(f"train: {len(train_df):,}개")
    print_dist("train 분포", train_df["risk_level"].tolist())

    print(f"\ntest : {len(test_df):,}개")
    print_dist("test 분포 (원본)", test_df["risk_level"].tolist())

    print(f"\ntrain rul_norm: {y_train.min():.4f} ~ {y_train.max():.4f}")
    print(f"test  rul_norm: {y_test.min():.4f} ~ {y_test.max():.4f}")

    return X_train, y_train, X_test, y_test, y_test_days


# ══════════════════════════════════════════════════════════════
# STEP 2. 모델 학습
# ══════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 2. XGBoost 모델 학습")
    print(f"  n_estimators : {XGBOOST_PARAMS['n_estimators']}")
    print(f"  max_depth    : {XGBOOST_PARAMS['max_depth']}")
    print(f"  learning_rate: {XGBOOST_PARAMS['learning_rate']}")
    print("=" * 60)

    model = XGBRegressor(**XGBOOST_PARAMS)

    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=50,
    )
    elapsed = time.time() - start
    print(f"\n학습 완료: {elapsed:.1f}초")

    # Cross Validation (train 기준)
    print("\nCross Validation (5-fold) 중...")
    cv_scores = cross_val_score(
        XGBRegressor(**XGBOOST_PARAMS),
        X_train, y_train,
        cv=5, scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()
    cv_std = cv_scores.std()
    print(f"CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")

    # Train 성능
    y_pred_train = np.clip(model.predict(X_train), 0, 1)
    train_mae    = mean_absolute_error(y_train, y_pred_train)
    train_r2     = r2_score(y_train, y_pred_train)

    print(f"\n[Train] MAE: {train_mae:.4f} | R²: {train_r2:.4f}")

    return model, cv_mae, cv_std, train_mae, train_r2


# ══════════════════════════════════════════════════════════════
# STEP 3. 평가
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test, y_test_days):
    print("\n" + "=" * 60)
    print("STEP 3. 모델 평가 (combined_test, 원본 분포)")
    print("=" * 60)

    rul_sc = joblib.load(RUL_SCALER)

    # 예측 (0~1)
    y_pred_norm = np.clip(model.predict(X_test), 0, 1)

    # 역변환 → 일 단위
    y_pred_days = np.maximum(
        rul_sc.inverse_transform(
            y_pred_norm.reshape(-1, 1)
        ).flatten(), 0
    )

    # 평가 지표
    mae  = mean_absolute_error(y_test_days, y_pred_days)
    rmse = np.sqrt(mean_squared_error(y_test_days, y_pred_days))
    r2   = r2_score(y_test_days, y_pred_days)

    actual_risk = [rul_to_risk(v) for v in y_test_days]
    pred_risk   = [rul_to_risk(v) for v in y_pred_days]
    risk_acc    = sum(
        a == p for a, p in zip(actual_risk, pred_risk)
    ) / len(y_test_days)

    print(f"Test MAE    : {mae:.4f}일")
    print(f"Test RMSE   : {rmse:.4f}일")
    print(f"Test R²     : {r2:.4f}")
    print(f"등급 정확도 : {risk_acc*100:.2f}%")

    # 과적합 판정
    print(f"\n과적합 판정:")
    print(f"  R² > 0.7    : {'✅' if r2 > 0.7 else '❌'} ({r2:.4f})")
    print(f"  등급 > 70%  : {'✅' if risk_acc > 0.7 else '❌'} ({risk_acc*100:.2f}%)")

    # 등급별 정확도
    print(f"\n등급별 정확도:")
    grade_acc = {}
    for g in ["LOW","MEDIUM","HIGH","CRITICAL"]:
        idx = [i for i, r in enumerate(actual_risk) if r == g]
        if not idx:
            continue
        correct = sum(actual_risk[i] == pred_risk[i] for i in idx)
        acc     = correct / len(idx)
        grade_acc[g] = acc
        print(f"  {g:10s}: {correct:5,}/{len(idx):5,} ({acc*100:.2f}%)")

    # 피처 중요도
    importance = model.feature_importances_
    feat_imp   = sorted(
        zip(FEATURE_COLS, importance),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n피처 중요도 (상위 5개):")
    for name, imp in feat_imp[:5]:
        print(f"  {name:25s}: {imp:.4f}")

    # LOW → CRITICAL 전환 감지
    print(f"\n[핵심] LOW→CRITICAL 전환 감지:")
    transitions = []
    for i in range(1, len(actual_risk)):
        if actual_risk[i-1] == "LOW" and actual_risk[i] == "CRITICAL":
            transitions.append({
                "index"    : i,
                "rul_actual": round(float(y_test_days[i]), 2),
                "rul_pred"  : round(float(y_pred_days[i]), 2),
                "pred_risk" : pred_risk[i],
                "감지성공"  : pred_risk[i] in ["HIGH","CRITICAL"],
            })

    if transitions:
        success = sum(t["감지성공"] for t in transitions)
        print(f"  전환 구간: {len(transitions)}개")
        print(f"  감지 성공: {success}/{len(transitions)} ({success/len(transitions)*100:.1f}%)")
    else:
        print("  전환 구간 없음")

    detail_df = pd.DataFrame({
        "actual_rul_days": np.round(y_test_days, 4),
        "pred_rul_days"  : np.round(y_pred_days, 4),
        "actual_risk"    : actual_risk,
        "pred_risk"      : pred_risk,
        "error_days"     : np.round(np.abs(y_test_days - y_pred_days), 4),
        "correct"        : [a == p for a, p in zip(actual_risk, pred_risk)],
    })

    feat_imp_df = pd.DataFrame(feat_imp, columns=["feature","importance"])

    metrics = {
        "mae"         : round(mae, 4),
        "rmse"        : round(rmse, 4),
        "r2"          : round(r2, 4),
        "risk_acc"    : round(risk_acc, 4),
        "low_acc"     : round(grade_acc.get("LOW",      0), 4),
        "medium_acc"  : round(grade_acc.get("MEDIUM",   0), 4),
        "high_acc"    : round(grade_acc.get("HIGH",     0), 4),
        "critical_acc": round(grade_acc.get("CRITICAL", 0), 4),
    }

    return detail_df, feat_imp_df, metrics


# ══════════════════════════════════════════════════════════════
# STEP 4. 저장
# ══════════════════════════════════════════════════════════════

def save_results(
    model      ,
    cv_mae     : float,
    cv_std     : float,
    train_mae  : float,
    train_r2   : float,
    detail_df  : pd.DataFrame,
    feat_imp_df: pd.DataFrame,
    metrics    : dict,
):
    print("\n" + "=" * 60)
    print("STEP 4. 저장")
    print("=" * 60)

    # 모델 저장
    model.save_model(str(MODEL_PATH))
    print(f"  xgboost_rul.json: 저장 완료")

    # training_report.xlsx
    overfit_r2  = train_r2 - metrics["r2"]
    overfit_mae = metrics["mae"] - train_mae

    if overfit_r2 > 0.15:
        verdict = "과적합 — max_depth 줄이기 / reg 높이기 권장"
    elif train_r2 < 0.5:
        verdict = "과소적합 — n_estimators 늘리기 / 피처 추가 권장"
    else:
        verdict = "양호 — Train/Test 균형 좋음"

    train_summary = pd.DataFrame([{
        "Train_MAE"    : train_mae,
        "Train_R²"     : train_r2,
        "Test_MAE(일)" : metrics["mae"],
        "Test_RMSE(일)": metrics["rmse"],
        "Test_R²"      : metrics["r2"],
        "CV_MAE"       : round(cv_mae, 4),
        "CV_std"       : round(cv_std, 4),
        "과적합갭_R²"  : round(overfit_r2, 4),
        "과적합갭_MAE" : round(overfit_mae, 4),
        "판정"         : verdict,
        "등급정확도"   : metrics["risk_acc"],
        "LOW정확도"    : metrics["low_acc"],
        "MEDIUM정확도" : metrics["medium_acc"],
        "HIGH정확도"   : metrics["high_acc"],
        "CRITICAL정확도": metrics["critical_acc"],
        "n_estimators" : XGBOOST_PARAMS["n_estimators"],
        "max_depth"    : XGBOOST_PARAMS["max_depth"],
        "learning_rate": XGBOOST_PARAMS["learning_rate"],
        "피처수"       : len(FEATURE_COLS),
    }])

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        train_summary.to_excel(writer, sheet_name="학습요약",    index=False)
        feat_imp_df.to_excel(  writer, sheet_name="피처중요도",  index=False)

    with pd.ExcelWriter(EVAL_PATH, engine="openpyxl") as writer:
        train_summary.to_excel(writer, sheet_name="평가요약",   index=False)
        detail_df.to_excel(    writer, sheet_name="샘플별결과", index=False)
        feat_imp_df.to_excel(  writer, sheet_name="피처중요도", index=False)

    print(f"  training_report.xlsx : 저장 완료")
    print(f"  evaluation_report.xlsx: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    X_train, y_train, X_test, y_test, y_test_days = load_data()

    model, cv_mae, cv_std, train_mae, train_r2 = train_model(
        X_train, y_train, X_test, y_test
    )

    detail_df, feat_imp_df, metrics = evaluate_model(
        model, X_test, y_test, y_test_days
    )

    save_results(
        model, cv_mae, cv_std, train_mae, train_r2,
        detail_df, feat_imp_df, metrics
    )

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("완료.")