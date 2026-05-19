import pandas as pd
import numpy as np
from pathlib import Path

input_path  = Path(r"D:\project\예지보전\kaist_analysis.xlsx")
output_path = Path(r"D:\project\예지보전\kaist_rul_labeled.xlsx")

df = pd.read_excel(input_path)
df = df.sort_values("timestamp").reset_index(drop=True)

total = len(df)           # 129
failure_idx = total - 1   # 128 = 고장 시점 (RUL=0)
onset_idx   = 124         # 급등 시작 시점

# ── RUL 계산 ──
# 정상 구간 (0 ~ onset_idx-1): 선형 감소
# 열화 구간 (onset_idx ~ failure_idx): 가속 감소 (제곱 비율)

rul_list = []
normal_days = 5.33  # 총 수명

for i in range(total):
    if i < onset_idx:
        # 정상 구간 선형
        rul = round(normal_days * (failure_idx - i) / failure_idx, 4)
    else:
        # 열화 구간 가속 감소
        remaining = failure_idx - onset_idx          # 4 스텝
        step = i - onset_idx                         # 0 ~ 4
        ratio = (1 - step / remaining) ** 2          # 제곱으로 가속
        rul = round(normal_days * (failure_idx - onset_idx) / failure_idx * ratio, 4)
    rul_list.append(rul)

df["rul_days"] = rul_list

# ── 위험도 등급 부여 ──
def rul_to_risk(rul):
    if rul >= 31:   return "LOW"
    if rul >= 16:   return "MEDIUM"
    if rul >= 3:    return "HIGH"
    return "CRITICAL"

df["risk_level"] = df["rul_days"].apply(rul_to_risk)

print(f"총 파일 수: {len(df)}")
print(f"RUL 범위: {df['rul_days'].min()} ~ {df['rul_days'].max()}")
print()
print("=== RUL 분포 ===")
print(df["risk_level"].value_counts())
print()
print("=== 처음 10개 ===")
print(df[["timestamp","vibration_rms","temp","rul_days","risk_level"]].head(10).to_string())
print()
print("=== 마지막 10개 ===")
print(df[["timestamp","vibration_rms","temp","rul_days","risk_level"]].tail(10).to_string())

df.to_excel(output_path, index=False)
print(f"\n엑셀 저장 완료: {output_path}")