# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════
  고장분류 Optuna 최적화 + 평가 (Colab GPU용)
  로컬에서 추출한 X_raw.npy, X_fft.npy, y.npy를 사용
═══════════════════════════════════════════════════════════════

사용법 (Colab):
  1. 런타임 → 런타임 유형 변경 → GPU 선택
  2. 이 파일 + .npy 파일들 업로드
  3. !pip install optuna xgboost
  4. !python colab_clf_optuna.py
"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

tf.get_logger().setLevel('ERROR')

# ══════════════════════════════════════════════
# 데이터 로드
# ══════════════════════════════════════════════
DATA_DIR = "."  # Colab에서 .npy 파일이 있는 경로

print("=" * 60)
print("  고장분류 Optuna + 평가 (Colab GPU)")
print("=" * 60)

print("\nStep 1. 데이터 로드")
X_raw = np.load(os.path.join(DATA_DIR, "X_raw.npy"))
X_fft = np.load(os.path.join(DATA_DIR, "X_fft.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))
class_names = np.load(os.path.join(DATA_DIR, "class_names.npy"), allow_pickle=True).tolist()

WINDOW_SIZE = X_raw.shape[1]
n_classes = len(class_names)
print(f"  X_raw: {X_raw.shape}")
print(f"  X_fft: {X_fft.shape}")
print(f"  y: {y.shape}, classes: {class_names}")
print(f"  GPU: {tf.config.list_physical_devices('GPU')}")


# ══════════════════════════════════════════════
# Optuna 최적화
# ══════════════════════════════════════════════
def optuna_xgboost_clf(X, y, n_splits=5, n_trials=60):
    """XGBoost Optuna (GPU 지원)"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 20),
            'tree_method': 'gpu_hist',  # GPU 가속
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
            'eval_metric': 'mlogloss',
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y[tr])
            scores.append(f1_score(y[te], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  [Optuna XGB] Best F1={study.best_value:.4f}")
    print(f"  [Optuna XGB] Best params: {study.best_params}")
    return study.best_params


def optuna_rf_clf(X, y, n_splits=5, n_trials=60):
    """RandomForest Optuna (CPU)"""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 30),
            'max_features': trial.suggest_float('max_features', 0.3, 0.7),
            'random_state': 42, 'n_jobs': -1,
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            model = RandomForestClassifier(**params)
            model.fit(X_tr, y[tr])
            scores.append(f1_score(y[te], model.predict(X_te), average='macro'))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  [Optuna RF] Best F1={study.best_value:.4f}")
    print(f"  [Optuna RF] Best params: {study.best_params}")
    return study.best_params


# ══════════════════════════════════════════════
# CNN-1D
# ══════════════════════════════════════════════
def build_cnn(n_classes, win_size):
    inp = keras.Input(shape=(win_size, 1))
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ══════════════════════════════════════════════
# 평가 함수
# ══════════════════════════════════════════════
def evaluate_cnn(X_raw, y, n_classes, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list, gap_list = [], [], []
    all_yt, all_yp = [], []

    for fold, (tr, te) in enumerate(skf.split(X_raw, y)):
        X_tr = X_raw[tr][..., np.newaxis]
        X_te = X_raw[te][..., np.newaxis]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, WINDOW_SIZE)).reshape(X_tr.shape)
        X_te = scaler.transform(X_te.reshape(-1, WINDOW_SIZE)).reshape(X_te.shape)

        model = build_cnn(n_classes, WINDOW_SIZE)
        model.fit(X_tr, y[tr], validation_data=(X_te, y[te]),
                  epochs=50, batch_size=256, verbose=0,
                  callbacks=[callbacks.EarlyStopping(monitor='val_accuracy',
                             patience=10, restore_best_weights=True, verbose=0)])

        pred_tr = np.argmax(model.predict(X_tr, verbose=0), axis=1)
        pred_te = np.argmax(model.predict(X_te, verbose=0), axis=1)
        tr_acc = accuracy_score(y[tr], pred_tr)
        te_acc = accuracy_score(y[te], pred_te)
        te_f1 = f1_score(y[te], pred_te, average='macro')

        acc_list.append(te_acc); f1_list.append(te_f1); gap_list.append(tr_acc - te_acc)
        all_yt.extend(y[te].tolist()); all_yp.extend(pred_te.tolist())
        print(f"    Fold {fold+1}: Acc={te_acc:.4f} F1={te_f1:.4f} gap={tr_acc-te_acc:.4f}")
        keras.backend.clear_session()

    return {"acc_scores": acc_list, "f1_scores": f1_list, "overfit_gaps": gap_list,
            "all_y_true": all_yt, "all_y_pred": all_yp}


def evaluate_tree(model_fn, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list, f1_list, gap_list = [], [], []
    all_yt, all_yp = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_te = scaler.transform(X[te])
        model = model_fn()
        model.fit(X_tr, y[tr])
        pred_tr = model.predict(X_tr)
        pred_te = model.predict(X_te)
        tr_acc = accuracy_score(y[tr], pred_tr)
        te_acc = accuracy_score(y[te], pred_te)
        te_f1 = f1_score(y[te], pred_te, average='macro')

        acc_list.append(te_acc); f1_list.append(te_f1); gap_list.append(tr_acc - te_acc)
        all_yt.extend(y[te].tolist()); all_yp.extend(pred_te.tolist())
        print(f"    Fold {fold+1}: Acc={te_acc:.4f} F1={te_f1:.4f} gap={tr_acc-te_acc:.4f}")

    return {"acc_scores": acc_list, "f1_scores": f1_list, "overfit_gaps": gap_list,
            "all_y_true": all_yt, "all_y_pred": all_yp}


# ══════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  Step 2. Optuna 최적화")
print("=" * 60)
best_xgb = optuna_xgboost_clf(X_fft, y)
best_rf = optuna_rf_clf(X_fft, y)

print("\n" + "=" * 60)
print("  Step 3. 모델 평가")
print("=" * 60)

results = {}

print("\n  [CNN-1D (Raw)]")
results['CNN-1D'] = evaluate_cnn(X_raw, y, n_classes)

print("\n  [XGBoost+FFT (Optuna)]")
results['XGB+FFT'] = evaluate_tree(
    lambda: xgb.XGBClassifier(**best_xgb, random_state=42, verbosity=0,
                               n_jobs=-1, eval_metric='mlogloss',
                               tree_method='gpu_hist'), X_fft, y)

print("\n  [RF+FFT (Optuna)]")
results['RF+FFT'] = evaluate_tree(
    lambda: RandomForestClassifier(**best_rf, random_state=42, n_jobs=-1),
    X_fft, y)

# ══════════════════════════════════════════════
# 결과 요약
# ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  결과 요약")
print("=" * 60)

summary = {}
for m, r in results.items():
    acc = np.mean(r['acc_scores']); f1m = np.mean(r['f1_scores'])
    gap = np.mean(r['overfit_gaps'])
    print(f"  {m:<15} Acc={acc:.4f} F1={f1m:.4f} Gap={gap:.4f}")
    report = classification_report(r['all_y_true'], r['all_y_pred'],
                                    target_names=class_names, digits=4)
    print(f"\n{report}")
    summary[m] = {
        "mean_acc": float(acc), "mean_f1": float(f1m), "mean_gap": float(gap),
        "best_xgb_params": best_xgb,
        "best_rf_params": best_rf,
    }

# 결과 저장
with open("comparison_results_optuna.json", 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"\n  [저장] comparison_results_optuna.json")
print("  완료!")
