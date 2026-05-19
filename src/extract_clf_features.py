# -*- coding: utf-8 -*-
"""
고장분류용 피처 추출 → .npy 저장
Colab에서 Optuna + 평가를 돌리기 위한 전처리
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loaders import load_all_classification, WINDOW_SIZE
from feature_engineering import extract_fft_features_batch
from sklearn.preprocessing import LabelEncoder

OUT_DIR = r"D:\project\예지보전_v2\clf_features"
os.makedirs(OUT_DIR, exist_ok=True)

print("Step 1. 데이터 로드")
df = load_all_classification()
if df.empty:
    print("데이터 없음"); exit()

print("Step 2. 클래스 균형화")
df_fault = df[df['fault_label'] != 'healthy'].copy()
min_count = df_fault['fault_label'].value_counts().min()
sampled = []
for label, g in df_fault.groupby('fault_label'):
    sampled.append(g.sample(min(len(g), min_count), random_state=42))
df_bal = pd.concat(sampled, ignore_index=True)
print(f"  분포: {df_bal['fault_label'].value_counts().to_dict()}")

le = LabelEncoder()
y = le.fit_transform(df_bal['fault_label'].values)
class_names = le.classes_.tolist()
print(f"  클래스: {class_names}")

print("Step 3. Raw + FFT 피처 추출")
X_raw = np.stack(df_bal['raw'].values).astype(np.float32)
X_fft = extract_fft_features_batch(X_raw)
print(f"  X_raw: {X_raw.shape}")
print(f"  X_fft: {X_fft.shape}")

# 전류 피처 추가
if 'current_rms' in df_bal.columns:
    current_vals = df_bal['current_rms'].fillna(0).values.reshape(-1, 1)
    X_fft = np.hstack([X_fft, current_vals])
    print(f"  전류 추가 → X_fft: {X_fft.shape}")

print("Step 4. 저장")
np.save(os.path.join(OUT_DIR, "X_raw.npy"), X_raw)
np.save(os.path.join(OUT_DIR, "X_fft.npy"), X_fft)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
np.save(os.path.join(OUT_DIR, "class_names.npy"), np.array(class_names))

for name in ["X_raw.npy", "X_fft.npy", "y.npy", "class_names.npy"]:
    path = os.path.join(OUT_DIR, name)
    size_mb = os.path.getsize(path) / (1024**2)
    print(f"  {name}: {size_mb:.1f} MB")

print("\n완료. Colab에 업로드할 파일:")
print(f"  {OUT_DIR}")
