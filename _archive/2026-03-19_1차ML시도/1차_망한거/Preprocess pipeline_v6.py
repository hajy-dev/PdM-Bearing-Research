import pandas as pd, os, glob

FEMTO_DIR = r"D:\project\데이터셋\10. FEMTO Bearing\FEMTOBearingDataSet\Full_Test_Set"

# 쉼표 구분자 파일 1개
b_dir = sorted(glob.glob(os.path.join(FEMTO_DIR, "Bearing*")))[0]
files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))

# 첫 번째 파일
fp = files[0]
with open(fp) as f:
    line = f.readline()
sep = ';' if ';' in line else ','
df = pd.read_csv(fp, header=None, sep=sep, nrows=3)
print(f"파일: {os.path.basename(fp)}  /  구분자: '{sep}'")
print(f"컬럼수: {df.shape[1]}")
print(df)

# Bearing1_4 (세미콜론 파일) 1개
b4 = os.path.join(FEMTO_DIR, "Bearing1_4")
files4 = sorted(glob.glob(os.path.join(b4, "*.csv")))
fp4 = files4[0]
with open(fp4) as f:
    line4 = f.readline()
sep4 = ';' if ';' in line4 else ','
df4 = pd.read_csv(fp4, header=None, sep=sep4, nrows=3)
print(f"\n파일: {os.path.basename(fp4)}  /  구분자: '{sep4}'")
print(f"컬럼수: {df4.shape[1]}")
print(df4)