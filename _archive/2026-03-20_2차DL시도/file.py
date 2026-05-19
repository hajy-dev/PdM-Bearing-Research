import os, glob

BASE = r"D:\project\데이터셋"

# FEMTO 내부 구조 확인
print("=== FEMTO ===")
femto = os.path.join(BASE, "10. FEMTO Bearing")
for root, dirs, files in os.walk(femto):
    depth = root.replace(femto, "").count(os.sep)
    if depth > 2:
        break
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth <= 1:
        for f in files[:3]:
            print(f"{indent}  {f}")

# XJTU 내부 구조 확인
print("\n=== XJTU ===")
xjtu = os.path.join(BASE, "XJTU-SY_Bearing_Datasets")
for root, dirs, files in os.walk(xjtu):
    depth = root.replace(xjtu, "").count(os.sep)
    if depth > 2:
        break
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/")
    if depth <= 1:
        for f in files[:3]:
            print(f"{indent}  {f}")

# KAIST CSV 샘플 컬럼 확인
print("\n=== KAIST 컬럼 구조 ===")
import pandas as pd
kaist = os.path.join(BASE, "Vibration_Bearing_RuntoFailure")
files = glob.glob(os.path.join(kaist, "**", "*.csv"), recursive=True)
fp = files[0]
df = pd.read_csv(fp, header=None, nrows=3)
print(f"파일명: {os.path.basename(fp)}")
print(f"컬럼수: {df.shape[1]}")
print(df)