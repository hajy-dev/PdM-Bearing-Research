"""
AI-Pass 예지보전 XGBoost 모델 학습 v2.0
========================================
v1 → v2 변경사항:
  - MinMaxScaler → StandardScaler (피처별 분산 통일)
  - 온도 범위(30~99°C) vs 진동(0.5~2.1g) 스케일 차이 해소
  - 진동 피처 기여도 상승 기대
  - combined_train.csv 원본 RUL 분포 기반 재스케일링

실행: python train_xgboost_v2.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from collections import Counter
import time

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ══════════════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR   = Path(r"D:\project\예지보전\output_v6")
TRAIN_CSV    = OUTPUT_DIR / "combined_train.csv"
TEST_CSV     = OUTPUT_DIR / "kaist_test.csv"
KAIST_TRAIN  = OUTPUT_DIR / "kaist_train.csv"   # 실제 RUL 분포 참조용
MODEL_PATH   = OUTPUT_DIR / "xgboost_rul_v2.json"
SCALER_PATH  = OUTPUT_DIR / "scaler_v2.pkl"
RUL_SCALER   = OUTPUT_DIR / "rul_scaler.pkl"
REPORT_PATH  = OUTPUT_DIR / "training_report_v2.xlsx"
EVAL_PATH    = OUTPUT_DIR / "evaluation_report_v2.xlsx"

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
    "n_estimators"     : 500,
    "max_depth"        : 6,
    "learning_rate"    : 0.03,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.0,
    "min_child_weight" : 3,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "tree_method"      : "hist",
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
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        print(f"  {g:10s}: {dist[g]:>8,}개 ({dist[g]/total*100:.1f}%)")


# ══════════════════════════════════════════════════════════════
# STEP 1. 데이터 로딩 + StandardScaler 재적용
# ══════════════════════════════════════════════════════════════

def load_and_rescale():
    print("\n" + "=" * 60)
    print("STEP 1. 데이터 로딩 + StandardScaler 재적용")
    print("=" * 60)

    train_df      = pd.read_csv(TRAIN_CSV)
    test_df       = pd.read_csv(TEST_CSV)
    kaist_train   = pd.read_csv(KAIST_TRAIN)

    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(0)
    test_df[FEATURE_COLS]  = test_df[FEATURE_COLS].fillna(0)

    print(f"combined_train : {len(train_df):,}행")
    print(f"kaist_test     : {len(test_df):,}행")
    print_dist("train 분포", train_df["risk_level"].tolist())
    print_dist("test 분포 (원본)", test_df["risk_level"].tolist())

    # ── StandardScaler fit (kaist_train 원본 기준) ──
    # SMOTE 합성 데이터가 아닌 실제 데이터 기준으로 fit
    print(f"\n[v2 변경] StandardScaler fit: kaist_train {len(kaist_train):,}개 기준")
    print(f"  → 피처별 평균=0, 분산=1 정규화")
    print(f"  → 온도/진동 스케일 차이 해소")

    scaler = StandardScaler()
    scaler.fit(kaist_train[FEATURE_COLS].fillna(0))

    # 재스케일링
    train_scaled = train_df.copy()
    test_scaled  = test_df.copy()
    train_scaled[FEATURE_COLS] = scaler.transform(train_df[FEATURE_COLS])
    test_scaled[FEATURE_COLS]  = scaler.transform(test_df[FEATURE_COLS])

    # 스케일러 저장
    joblib.dump(scaler, SCALER_PATH)
    print(f"  scaler_v2.pkl 저장 완료")

    # 피처별 스케일 정보 출력
    print(f"\n피처별 mean/std (상위 5개 중요 피처):")
    feat_stats = list(zip(FEATURE_COLS, scaler.mean_, scaler.scale_))
    for name, mean, std in sorted(feat_stats, key=lambda x: x[2], reverse=True)[:5]:
        print(f"  {name:25s}: mean={mean:8.4f}, std={std:8.4f}")

    # RUL: kaist_train 실제 값 기반 (SMOTE 중간값 아님)
    # → combined_train의 rul_days가 SMOTE 중간값이므로
    #   kaist_train의 실제 분포로 보정
    rul_sc = joblib.load(RUL_SCALER)

    # train: rul_norm 그대로 사용 (SMOTE 균등 분포 유지)
    X_train = train_scaled[FEATURE_COLS].values
    y_train = train_scaled["rul_norm"].values

    # test: 실제 rul_days 역변환
    X_test      = test_scaled[FEATURE_COLS].values
    y_test_norm = test_scaled["rul_norm"].values
    y_test_days = test_df["rul_days"].values

    print(f"\ntrain rul_norm: {y_train.min():.4f} ~ {y_train.max():.4f}")
    print(f"test  rul_days: {y_test_days.min():.2f} ~ {y_test_days.max():.2f}일")

    return X_train, y_train, X_test, y_test_norm, y_test_days, scaler


# ══════════════════════════════════════════════════════════════
# STEP 2. 모델 학습
# ══════════════════════════════════════════════════════════════

def train_model(X_train, y_train, X_test, y_test_norm):
    print("\n" + "=" * 60)
    print("STEP 2. XGBoost 모델 학습")
    print(f"  n_estimators  : {XGBOOST_PARAMS['n_estimators']}")
    print(f"  max_depth     : {XGBOOST_PARAMS['max_depth']}")
    print(f"  learning_rate : {XGBOOST_PARAMS['learning_rate']}")
    print("=" * 60)

    model = XGBRegressor(**XGBOOST_PARAMS)

    start = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test_norm)],
        verbose=50,
    )
    elapsed = time.time() - start
    print(f"\n학습 완료: {elapsed:.1f}초")

    # Train 성능
    y_pred_train = np.clip(model.predict(X_train), 0, 1)
    train_mae    = mean_absolute_error(y_train, y_pred_train)
    train_r2     = r2_score(y_train, y_pred_train)
    print(f"[Train] MAE: {train_mae:.4f} | R²: {train_r2:.4f}")

    # Cross Validation
    print("\nCross Validation (5-fold) 중...")
    cv_scores = cross_val_score(
        XGBRegressor(**XGBOOST_PARAMS),
        X_train, y_train,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    cv_mae = -cv_scores.mean()
    cv_std =  cv_scores.std()
    print(f"CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")

    return model, cv_mae, cv_std, train_mae, train_r2


# ══════════════════════════════════════════════════════════════
# STEP 3. 평가
# ══════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test_norm, y_test_days):
    print("\n" + "=" * 60)
    print("STEP 3. 모델 평가 (kaist_test, 원본 분포)")
    print("=" * 60)

    rul_sc = joblib.load(RUL_SCALER)

    y_pred_norm = np.clip(model.predict(X_test), 0, 1)
    y_pred_days = np.maximum(
        rul_sc.inverse_transform(
            y_pred_norm.reshape(-1, 1)
        ).flatten(), 0
    )

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

    print(f"\n과적합 판정:")
    print(f"  R² > 0.7   : {'✅' if r2 > 0.7 else '❌'} ({r2:.4f})")
    print(f"  등급 > 70% : {'✅' if risk_acc > 0.7 else '❌'} ({risk_acc*100:.2f}%)")

    print(f"\n등급별 정확도:")
    grade_acc = {}
    for g in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        idx = [i for i, r in enumerate(actual_risk) if r == g]
        if not idx:
            continue
        correct = sum(actual_risk[i] == pred_risk[i] for i in idx)
        acc     = correct / len(idx)
        grade_acc[g] = acc
        print(f"  {g:10s}: {correct:6,}/{len(idx):6,} ({acc*100:.2f}%)")

    # 피처 중요도
    importance  = model.feature_importances_
    feat_imp    = sorted(
        zip(FEATURE_COLS, importance),
        key=lambda x: x[1], reverse=True
    )
    print(f"\n피처 중요도 (전체):")
    for name, imp in feat_imp:
        bar = "█" * int(imp * 50)
        print(f"  {name:25s}: {imp:.4f} {bar}")

    # LOW→CRITICAL 전환 감지
    print(f"\n[핵심] LOW→CRITICAL 전환 감지:")
    transitions = []
    for i in range(1, len(actual_risk)):
        if actual_risk[i-1] == "LOW" and actual_risk[i] == "CRITICAL":
            transitions.append({
                "index"     : i,
                "rul_actual": round(float(y_test_days[i]), 2),
                "rul_pred"  : round(float(y_pred_days[i]), 2),
                "pred_risk" : pred_risk[i],
                "감지성공"  : pred_risk[i] in ["HIGH", "CRITICAL"],
            })

    if transitions:
        success = sum(t["감지성공"] for t in transitions)
        print(f"  전환 구간: {len(transitions)}개")
        print(f"  감지 성공: {success}/{len(transitions)}"
              f" ({success/len(transitions)*100:.1f}%)")
    else:
        print("  전환 구간 없음")

    detail_df   = pd.DataFrame({
        "actual_rul_days": np.round(y_test_days, 4),
        "pred_rul_days"  : np.round(y_pred_days, 4),
        "actual_risk"    : actual_risk,
        "pred_risk"      : pred_risk,
        "error_days"     : np.round(np.abs(y_test_days - y_pred_days), 4),
        "correct"        : [a == p for a, p in zip(actual_risk, pred_risk)],
    })
    feat_imp_df = pd.DataFrame(feat_imp, columns=["feature", "importance"])

    metrics = {
        "mae"         : round(mae,      4),
        "rmse"        : round(rmse,     4),
        "r2"          : round(r2,       4),
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

    model.save_model(str(MODEL_PATH))
    print(f"  xgboost_rul_v2.json: 저장 완료")

    overfit_r2  = train_r2 - metrics["r2"]
    overfit_mae = metrics["mae"] - train_mae

    if overfit_r2 > 0.15:
        verdict = "과적합 — max_depth 줄이기 권장"
    elif train_r2 < 0.5:
        verdict = "과소적합 — n_estimators 늘리기 권장"
    else:
        verdict = "양호"

    summary_df = pd.DataFrame([{
        "버전"          : "v2 (StandardScaler)",
        "Train_MAE"     : train_mae,
        "Train_R²"      : train_r2,
        "Test_MAE(일)"  : metrics["mae"],
        "Test_RMSE(일)" : metrics["rmse"],
        "Test_R²"       : metrics["r2"],
        "CV_MAE"        : round(cv_mae, 4),
        "CV_std"        : round(cv_std, 4),
        "과적합갭_R²"   : round(overfit_r2, 4),
        "과적합갭_MAE"  : round(overfit_mae, 4),
        "판정"          : verdict,
        "등급정확도"    : metrics["risk_acc"],
        "LOW정확도"     : metrics["low_acc"],
        "MEDIUM정확도"  : metrics["medium_acc"],
        "HIGH정확도"    : metrics["high_acc"],
        "CRITICAL정확도": metrics["critical_acc"],
        "정규화"        : "StandardScaler (kaist_train 기준 fit)",
        "n_estimators"  : XGBOOST_PARAMS["n_estimators"],
        "max_depth"     : XGBOOST_PARAMS["max_depth"],
        "learning_rate" : XGBOOST_PARAMS["learning_rate"],
    }])

    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        summary_df.to_excel( writer, sheet_name="학습요약",   index=False)
        feat_imp_df.to_excel(writer, sheet_name="피처중요도", index=False)

    with pd.ExcelWriter(EVAL_PATH, engine="openpyxl") as writer:
        summary_df.to_excel( writer, sheet_name="평가요약",   index=False)
        detail_df.to_excel(  writer, sheet_name="샘플별결과", index=False)
        feat_imp_df.to_excel(writer, sheet_name="피처중요도", index=False)

    print(f"  training_report_v2.xlsx : 저장 완료")
    print(f"  evaluation_report_v2.xlsx: 저장 완료")
    print(f"\n출력 경로: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()

    X_train, y_train, X_test, y_test_norm, y_test_days, scaler = (
        load_and_rescale()
    )

    model, cv_mae, cv_std, train_mae, train_r2 = train_model(
        X_train, y_train, X_test, y_test_norm
    )

    detail_df, feat_imp_df, metrics = evaluate_model(
        model, X_test, y_test_norm, y_test_days
    )

    save_results(
        model, cv_mae, cv_std, train_mae, train_r2,
        detail_df, feat_imp_df, metrics
    )

    elapsed = time.time() - total_start
    print(f"\n전체 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    print("완료.")