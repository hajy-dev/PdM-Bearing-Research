import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob, numpy as np

BASE = r"D:\project\데이터셋"
FEMTO_DIR = os.path.join(BASE, "10. FEMTO Bearing", "FEMTOBearingDataSet", "Full_Test_Set")

# bearing 1개만 로드해서 RMS 추세 확인
b_dir = sorted(glob.glob(os.path.join(FEMTO_DIR, "Bearing*")))[0]
files = sorted(glob.glob(os.path.join(b_dir, "*.csv")))

rms_list = []
for fp in files:
    with open(fp) as f:
        line = f.readline()
    sep = ';' if ';' in line else ','
    df = pd.read_csv(fp, header=None, sep=sep)
    vib_col = 1 if df.shape[1] >= 2 else 0
    signal = df.iloc[:, vib_col].values.astype(float)
    windows = [signal[i:i+256] for i in range(0, len(signal)-256, 128)]
    for w in windows:
        rms_list.append(float(np.sqrt(np.mean(w**2))))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(rms_list)
ax.set_title(f"{os.path.basename(b_dir)} — RMS over time")
ax.set_xlabel("Window index")
ax.set_ylabel("RMS")
plt.tight_layout()
plt.savefig(r"D:\project\예지보전_v2\rms_trend_check.png")
print(f"총 windows: {len(rms_list)}")
print(f"RMS 범위: {min(rms_list):.4f} ~ {max(rms_list):.4f}")
print(f"초반 10개 평균: {np.mean(rms_list[:10]):.4f}")
print(f"후반 10개 평균: {np.mean(rms_list[-10:]):.4f}")