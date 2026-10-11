"""
Deep Learning Application in Data Science
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ML Repository, via sklearn)
Uses the cleaned dataset produced by Week 1's data_cleaning_pipeline.py
Framework: TensorFlow / Keras
Author: Rajkumar Yadav
"""

import os, json, itertools, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)

SEED = 42
GREEN, ORANGE, BLUE, GREY = "#4C9F70", "#D9622B", "#2F6DB5", "#8A8F98"
results = {}

def set_seed(s=SEED):
    keras.utils.set_random_seed(s)

def metrics(y_true, y_pred, y_proba):
    return {'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)}

# ============================================================
# STEP 1: DATA PREPARATION (same split as Week 4 for a fair comparison)
# ============================================================
df = pd.read_csv("cleaned_dataset.csv")                  # output of Week 1 pipeline
y = df['diagnosis'].map({'malignant': 1, 'benign': 0}).values
X = df.drop(columns=['diagnosis'])
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, stratify=y, random_state=SEED)      # identical to Week 4
scaler = StandardScaler().fit(X_train)                  # fit on training data only
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# Validation set carved out of the training set (used for early stopping and learning curves)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_s, y_train, test_size=0.2, stratify=y_train, random_state=SEED)
print(f"Train: {len(X_tr)}  Validation: {len(X_val)}  Test: {len(X_test_s)}  Features: {X_tr.shape[1]}")
results['split'] = {'train': len(X_tr), 'val': len(X_val), 'test': len(X_test_s),
                    'n_features': int(X_tr.shape[1]), 'malignant_in_test': int(y_test.sum())}

# ============================================================
# STEP 2: MODEL BUILDER
# ============================================================
def build_model(hidden=(32, 16), dropout=0.3, l2=1e-3, lr=1e-3, n_features=30):
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = keras.Sequential([layers.Input(shape=(n_features,))])
    for units in hidden:
        model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg))
        if dropout > 0:
            model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def early_stop(patience=15):
    return callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                   restore_best_weights=True)

# ============================================================
# STEP 3: OVERFITTING DEMONSTRATION (large network, no safeguards)
# ============================================================
set_seed()
big = build_model(hidden=(128, 128, 64), dropout=0.0, l2=0.0)
hist_big = big.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=200,
                   batch_size=32, verbose=0)
vl = np.array(hist_big.history['val_loss'])
best_ep = int(vl.argmin()) + 1
results['overfit'] = {
    'params': int(big.count_params()),
    'best_val_loss': float(vl.min()), 'best_epoch': best_ep,
    'final_train_loss': float(hist_big.history['loss'][-1]),
    'final_val_loss': float(vl[-1]),
    'final_train_acc': float(hist_big.history['accuracy'][-1]),
    'final_val_acc': float(hist_big.history['val_accuracy'][-1]),
}
print("\n[Unregularized 128-128-64 network]", results['overfit'])

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ep = np.arange(1, 201)
ax[0].plot(ep, hist_big.history['loss'], color=BLUE, label='Training loss')
ax[0].plot(ep, vl, color=ORANGE, label='Validation loss')
ax[0].axvline(best_ep, color=GREY, ls='--', lw=1)
ax[0].text(best_ep + 4, vl.max() * 0.85, f"best val loss\nepoch {best_ep}", color=GREY, fontsize=8)
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Binary cross-entropy loss")
ax[0].set_title("Without safeguards: validation loss turns upward"); ax[0].legend()
ax[1].plot(ep, hist_big.history['accuracy'], color=BLUE, label='Training accuracy')
ax[1].plot(ep, hist_big.history['val_accuracy'], color=ORANGE, label='Validation accuracy')
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].set_ylim(0.9, 1.005)
ax[1].set_title("Training accuracy hits 100%"); ax[1].legend(loc='lower right')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig_overfitting.png", dpi=150); plt.close()

# ============================================================
# STEP 4: HYPERPARAMETER SEARCH (5-fold stratified CV on training data only)
# Scaling is fit inside each fold, so no information leaks from the validation fold.
# ============================================================
def cv_f1(config, X_raw, y_raw, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X_raw, y_raw):
        sc = StandardScaler().fit(X_raw[tr_idx])
        Xa, Xb = sc.transform(X_raw[tr_idx]), sc.transform(X_raw[va_idx])
        Xa2, Xi, ya2, yi = train_test_split(Xa, y_raw[tr_idx], test_size=0.15,
                                            stratify=y_raw[tr_idx], random_state=SEED)
        set_seed()
        m = build_model(**config)
        m.fit(Xa2, ya2, validation_data=(Xi, yi), epochs=150, batch_size=32,
              callbacks=[early_stop()], verbose=0)
        pred = (m.predict(Xb, verbose=0).ravel() >= 0.5).astype(int)
        scores.append(f1_score(y_raw[va_idx], pred))
        keras.backend.clear_session()
    return float(np.mean(scores)), float(np.std(scores))

grid = {'hidden': [(16, 8), (32, 16), (64, 32)], 'dropout': [0.2, 0.4], 'l2': [1e-4, 1e-3]}
search = []
t0 = time.time()
for hidden, dropout, l2 in itertools.product(grid['hidden'], grid['dropout'], grid['l2']):
    cfg = {'hidden': hidden, 'dropout': dropout, 'l2': l2}
    m, s = cv_f1(cfg, X_train, y_train)
    search.append({'hidden': list(hidden), 'dropout': dropout, 'l2': l2, 'cv_f1': m, 'cv_std': s})
    print(f"hidden={hidden} dropout={dropout} l2={l2}: CV F1={m:.4f} (+/- {s:.4f})", flush=True)
print(f"Search took {time.time()-t0:.0f}s")
search.sort(key=lambda r: r['cv_f1'], reverse=True)
best = search[0]
results['search'] = search
results['best_config'] = best
print("\nBest config:", best)

# Logistic Regression under the identical CV protocol (Week 4's winner) for a fair comparison
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lr_scores = []
for tr_idx, va_idx in skf.split(X_train, y_train):
    sc = StandardScaler().fit(X_train[tr_idx])
    lr = LogisticRegression(C=10, max_iter=5000).fit(sc.transform(X_train[tr_idx]), y_train[tr_idx])
    lr_scores.append(f1_score(y_train[va_idx], lr.predict(sc.transform(X_train[va_idx]))))
results['lr_cv_f1'] = [float(np.mean(lr_scores)), float(np.std(lr_scores))]
print(f"Logistic Regression (C=10) CV F1 under same protocol: {np.mean(lr_scores):.4f} (+/- {np.std(lr_scores):.4f})")

# ============================================================
# STEP 5: FINAL TRAINING OF THE CHOSEN ARCHITECTURE
# ============================================================
cfg = {'hidden': tuple(best['hidden']), 'dropout': best['dropout'], 'l2': best['l2']}
set_seed()
model = build_model(**cfg)
model.summary()
hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=300, batch_size=32,
                 callbacks=[early_stop()], verbose=0)
n_epochs = len(hist.history['loss'])
best_epoch = int(np.argmin(hist.history['val_loss'])) + 1
results['final_training'] = {'epochs_run': n_epochs, 'best_epoch': best_epoch,
                             'params': int(model.count_params()),
                             'val_loss': float(min(hist.history['val_loss']))}
print(f"Stopped after {n_epochs} epochs; best epoch {best_epoch}")

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].plot(hist.history['loss'], color=BLUE, label='Training loss')
ax[0].plot(hist.history['val_loss'], color=ORANGE, label='Validation loss')
ax[0].axvline(best_epoch - 1, color=GREY, ls='--', lw=1)
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss (incl. L2 penalty)")
ax[0].set_title("Regularized network: curves stay together"); ax[0].legend()
ax[1].plot(hist.history['accuracy'], color=BLUE, label='Training accuracy')
ax[1].plot(hist.history['val_accuracy'], color=ORANGE, label='Validation accuracy')
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
ax[1].set_title("Accuracy"); ax[1].legend(loc='lower right')
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig_training_curves.png", dpi=150); plt.close()
model.save("breast_cancer_nn.keras")

# ============================================================
# STEP 6: HELD-OUT TEST EVALUATION (touched once) + comparison with Week 4
# ============================================================
p_test = model.predict(X_test_s, verbose=0).ravel()
pred_test = (p_test >= 0.5).astype(int)
nn_m = metrics(y_test, pred_test, p_test)
cm = confusion_matrix(y_test, pred_test)
print("\nNeural network test metrics:", {k: round(v, 4) for k, v in nn_m.items()})
print("Confusion matrix:\n", cm)

lr_final = LogisticRegression(C=10, max_iter=5000, random_state=SEED).fit(X_train_s, y_train)
p_lr = lr_final.predict_proba(X_test_s)[:, 1]
lr_m = metrics(y_test, (p_lr >= 0.5).astype(int), p_lr)
cm_lr = confusion_matrix(y_test, (p_lr >= 0.5).astype(int))
print("Logistic Regression test metrics:", {k: round(v, 4) for k, v in lr_m.items()})
results['test_nn'] = nn_m; results['test_lr'] = lr_m
results['cm_nn'] = cm.tolist(); results['cm_lr'] = cm_lr.tolist()

# Robustness: repeat the whole training with 5 different random seeds
seed_runs = []
for s in range(42, 47):
    set_seed(s)
    mm = build_model(**cfg)
    mm.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=300, batch_size=32,
           callbacks=[early_stop()], verbose=0)
    pp = mm.predict(X_test_s, verbose=0).ravel()
    seed_runs.append(metrics(y_test, (pp >= 0.5).astype(int), pp))
    keras.backend.clear_session()
seed_df = pd.DataFrame(seed_runs)
results['seed_runs'] = {'mean': seed_df.mean().to_dict(), 'std': seed_df.std().to_dict(),
                        'f1_each': seed_df['f1'].tolist()}
print("\n5-seed test metrics (mean +/- std):")
print(pd.DataFrame({'mean': seed_df.mean(), 'std': seed_df.std()}).round(4))

# Confusion matrix + ROC figure
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=18,
                   color='white' if cm[i, j] > cm.max() / 2 else 'black')
ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
ax[0].set_xticklabels(['Benign', 'Malignant']); ax[0].set_yticklabels(['Benign', 'Malignant'])
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual"); ax[0].set_title("Neural network: confusion matrix")
fpr, tpr, _ = roc_curve(y_test, p_test); fpr2, tpr2, _ = roc_curve(y_test, p_lr)
ax[1].plot(fpr, tpr, color=BLUE, lw=2, label=f"Neural network (AUC {nn_m['roc_auc']:.4f})")
ax[1].plot(fpr2, tpr2, color=ORANGE, lw=2, ls='--', label=f"Logistic Regression (AUC {lr_m['roc_auc']:.4f})")
ax[1].plot([0, 1], [0, 1], color=GREY, lw=1, ls=':')
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC curves on the held-out test set"); ax[1].legend(loc='lower right')
for a in ax[1:]:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig_evaluation.png", dpi=150); plt.close()

# ============================================================
# STEP 7: MISCLASSIFICATION + UNCERTAINTY ANALYSIS (Monte Carlo dropout)
# Week 4 found 2 "confidently wrong" errors. Can dropout-based uncertainty flag them?
# ============================================================
mc = np.stack([model(X_test_s, training=True).numpy().ravel() for _ in range(200)])
mc_mean, mc_std = mc.mean(0), mc.std(0)
wrong = np.where(pred_test != y_test)[0]
right = np.where(pred_test == y_test)[0]
err_table = pd.DataFrame({'test_row': wrong, 'actual': y_test[wrong], 'predicted': pred_test[wrong],
                          'nn_proba_malignant': p_test[wrong].round(4),
                          'lr_proba_malignant': p_lr[wrong].round(4),
                          'mc_dropout_std': mc_std[wrong].round(4)})
print("\nMisclassified by the neural network:\n", err_table)
lr_wrong = np.where((p_lr >= 0.5).astype(int) != y_test)[0]
print("Misclassified by Logistic Regression (test rows):", lr_wrong.tolist())
print(f"MC-dropout std: correct predictions mean={mc_std[right].mean():.4f}, "
      f"errors mean={mc_std[wrong].mean() if len(wrong) else float('nan'):.4f}")
results['errors'] = err_table.to_dict(orient='records')
results['lr_wrong_rows'] = lr_wrong.tolist()
results['mc_std'] = {'correct_mean': float(mc_std[right].mean()),
                     'error_mean': float(mc_std[wrong].mean()) if len(wrong) else None,
                     'correct_p90': float(np.percentile(mc_std[right], 90))}
# How many of the test cases would a "send to human review" rule catch?
thr = np.percentile(mc_std[right], 90)
flagged = mc_std > thr
results['mc_flag'] = {'threshold': float(thr), 'n_flagged': int(flagged.sum()),
                      'errors_flagged': int(flagged[wrong].sum()), 'n_errors': int(len(wrong))}
print("Review rule (MC std > 90th pct of correct cases):", results['mc_flag'])

# ============================================================
# STEP 8: WHAT DID THE NETWORK LEARN? (permutation importance, test-set F1 drop)
# Used for interpretation only -- no modelling decision depends on it.
# ============================================================
rng = np.random.default_rng(SEED)
base_f1 = f1_score(y_test, pred_test)
imp = []
for j, name in enumerate(feature_names):
    drops = []
    for _ in range(15):
        Xp = X_test_s.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        drops.append(base_f1 - f1_score(y_test, (model.predict(Xp, verbose=0).ravel() >= 0.5).astype(int)))
    imp.append((name, float(np.mean(drops))))
imp.sort(key=lambda t: t[1], reverse=True)
results['importance'] = imp[:10]
print("\nTop 10 features by permutation importance:")
for n, v in imp[:10]:
    print(f"  {n}: {v:.4f}")
top = imp[:10][::-1]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh([t[0] for t in top], [t[1] for t in top], color=BLUE)
ax.set_xlabel("Drop in test F1 when the feature is shuffled")
ax.set_title("Which features does the network rely on?")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig("fig_importance.png", dpi=150); plt.close()

# ============================================================
# STEP 9: ARCHITECTURE DIAGRAM
# ============================================================
h = list(cfg['hidden'])
blocks = [("Input", f"{X_tr.shape[1]} scaled\nfeatures", GREY)]
for i, u in enumerate(h):
    blocks.append((f"Dense {i+1}", f"{u} units\nReLU + L2({cfg['l2']:g})", BLUE))
    blocks.append((f"Dropout {i+1}", f"rate {cfg['dropout']}", ORANGE))
blocks.append(("Output", "1 unit\nSigmoid", GREEN))
fig, ax = plt.subplots(figsize=(1.7 * len(blocks), 2.6))
ax.set_xlim(0, len(blocks) * 1.7); ax.set_ylim(0, 2.6); ax.axis('off')
for i, (t, sub, c) in enumerate(blocks):
    x = i * 1.7 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.7), 1.3, 1.3, boxstyle="round,pad=0.03", fc=c, ec='none', alpha=0.9))
    ax.text(x + 0.65, 1.62, t, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    ax.text(x + 0.65, 1.12, sub, ha='center', va='center', color='white', fontsize=8)
    if i < len(blocks) - 1:
        ax.annotate("", xy=(x + 1.62, 1.35), xytext=(x + 1.34, 1.35),
                    arrowprops=dict(arrowstyle="->", color='#444'))
ax.text(len(blocks) * 1.7 / 2, 0.25,
        f"Adam (lr=1e-3), binary cross-entropy, batch size 32, early stopping on validation loss  |  "
        f"{model.count_params():,} trainable parameters", ha='center', fontsize=8.5, color='#333')
plt.savefig("fig_architecture.png", dpi=150, bbox_inches='tight'); plt.close()

json.dump(results, open("week5_results.json", "w"), indent=2, default=float)
print("\nSaved: fig_*.png, breast_cancer_nn.keras, week5_results.json")
