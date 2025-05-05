#!/usr/bin/env python
"""
train_lp.py
-----------
Train a neural network on randomly‑generated linear programs and
save model, scaler, training history, and benchmark metrics.

Outputs into results_lp/:
  • model.keras
  • best.weights.h5
  • scaler.pkl
  • history.json
  • benchmark.json
"""

# ────────────────────────────────────────────────────────────────
# Imports & logging
# ────────────────────────────────────────────────────────────────
import os, argparse, logging, math, json, pathlib, pickle
import numpy as np, matplotlib.pyplot as plt, tensorflow as tf
from tqdm import tqdm
import lp_nn as lp     # helper module you already have

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# ────────────────────────────────────────────────────────────────
# CLI arguments
# ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n",        type=int,   default=50)
parser.add_argument("--m",        type=int,   default=25)
parser.add_argument("--hidden",   type=str,   default="256,128,64")
parser.add_argument("--samples",  type=int,   default=8000)
parser.add_argument("--lam_feas", type=float, default=0.5)
parser.add_argument("--lam_gap",  type=float, default=0.1)
parser.add_argument("--epochs",   type=int,   default=120)
parser.add_argument("--batch",    type=int,   default=64)
args = parser.parse_args()

n, m          = args.n, args.m
hidden_layers = [int(x) for x in args.hidden.split(",") if x.strip()]
samples       = args.samples
lam_feas      = args.lam_feas
lam_gap       = args.lam_gap
epochs        = args.epochs
bs            = args.batch

out_dir = pathlib.Path("results_lp"); out_dir.mkdir(exist_ok=True)

# ────────────────────────────────────────────────────────────────
# 1. Generate data
# ────────────────────────────────────────────────────────────────
logging.info("Generating %d LPs  (n=%d, m=%d)…", samples, n, m)
X, Y, params, scaler = lp.generate_dataset(n, m, sparsity=0.20, num_samples=samples)
Xtr,Xva,Xte,Ytr,Yva,Yte,params_te = lp.split(X, Y, params, tr=0.7, va=0.15)
logging.info("Split: train=%d  val=%d  test=%d", len(Xtr), len(Xva), len(Xte))

# cast to float32
Xtr, Xva, Ytr, Yva = map(lambda a: a.astype(np.float32), [Xtr, Xva, Ytr, Yva])

Atr = np.stack([p[1] for p in params[:len(Xtr)]]).astype(np.float32)
btr = np.stack([p[2] for p in params[:len(Xtr)]]).astype(np.float32)
ctr = np.stack([p[0] for p in params[:len(Xtr)]]).astype(np.float32)
Ava = np.stack([p[1] for p in params[len(Xtr):len(Xtr)+len(Xva)]]).astype(np.float32)
bva = np.stack([p[2] for p in params[len(Xtr):len(Xtr)+len(Xva)]]).astype(np.float32)
cva = np.stack([p[0] for p in params[len(Xtr):len(Xtr)+len(Xva)]]).astype(np.float32)

# ────────────────────────────────────────────────────────────────
# 2. Build model
# ────────────────────────────────────────────────────────────────
model = lp.build_nn(X.shape[1], n, hidden_layers)
optimizer = tf.keras.optimizers.Adam()

@tf.function
def batch_loss(y_true, y_pred, A, b, c):
    y_true = tf.cast(y_true, tf.float32)
    mse  = tf.reduce_mean(tf.square(y_true - y_pred))
    feas = tf.reduce_mean(tf.square(tf.matmul(A, tf.expand_dims(y_pred,-1))[:,:,0] - b))
    gap  = tf.reduce_mean(tf.abs(tf.reduce_sum(c*y_pred,1) - tf.reduce_sum(c*y_true,1)))
    return mse + lam_feas*feas + lam_gap*gap, mse, feas, gap

def make_ds(X,Y,A,b,c,shuffle):
    ds = tf.data.Dataset.from_tensor_slices((X,Y,A,b,c))
    if shuffle: ds = ds.shuffle(len(X))
    return ds.batch(bs, drop_remainder=True)

train_ds = make_ds(Xtr,Ytr,Atr,btr,ctr,True)
val_ds   = make_ds(Xva,Yva,Ava,bva,cva,False)

# ────────────────────────────────────────────────────────────────
# 3. Training loop
# ────────────────────────────────────────────────────────────────
hist = {k:[] for k in ["loss","val_loss","mse","feas","gap"]}
patience, best, no_imp = 10, math.inf, 0
ckpt = out_dir/"best.weights.h5"

for epoch in range(1, epochs+1):
    bar = tqdm(train_ds, desc=f"Epoch {epoch}/{epochs}", ncols=80)
    sums = np.zeros(4, dtype=np.float64)
    for Xb,Yb,Ab,bb,cb in bar:
        with tf.GradientTape() as tape:
            pred = model(Xb, training=True)
            L,m,f,g = batch_loss(Yb, pred, Ab, bb, cb)
        optimizer.apply_gradients(zip(tape.gradient(L, model.trainable_variables),
                                      model.trainable_variables))
        sums += np.array([float(L), float(m), float(f), float(g)], dtype=np.float64)
    nb = len(train_ds)
    hist["loss"].append(float(sums[0]/nb))
    hist["mse"].append(float(sums[1]/nb))
    hist["feas"].append(float(sums[2]/nb))
    hist["gap"].append(float(sums[3]/nb))

    v = 0.0
    for Xb,Yb,Ab,bb,cb in val_ds:
        v += batch_loss(Yb, model(Xb, training=False), Ab, bb, cb)[0]
    v = float(v/len(val_ds))
    hist["val_loss"].append(v)

    logging.info("epoch %3d | train %.3f | val %.3f | feas %.3f | gap %.3f",
                 epoch, hist["loss"][-1], v, hist["feas"][-1], hist["gap"][-1])

    if v < best - 1e-6:
        best, no_imp = v, 0
        model.save_weights(ckpt)
    else:
        no_imp += 1
        if no_imp >= patience:
            logging.info("Early stop at epoch %d", epoch)
            break

model.load_weights(ckpt)
best_epoch = int(np.argmin(hist["val_loss"])) + 1
logging.info("Best epoch = %d  | best val_loss = %.3f", best_epoch, best)

# ────────────────────────────────────────────────────────────────
# 4. Loss plot
# ────────────────────────────────────────────────────────────────
plt.figure(figsize=(6,3))
plt.plot(hist["loss"], label="train"), plt.plot(hist["val_loss"], label="val")
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────────────────────
# 5. Benchmark
# ────────────────────────────────────────────────────────────────
bench = lp.benchmark(params_te[:500], scaler, model)
logging.info("Benchmark:\n%s", json.dumps(bench, indent=2))

# ────────────────────────────────────────────────────────────────
# 6. Save artefacts
# ────────────────────────────────────────────────────────────────
def to_py(o):    # make NumPy types JSON‑serialisable
    if hasattr(o, "item"): return o.item()
    return o

json.dump(hist,  open(out_dir/"history.json","w"), default=to_py)
json.dump(bench, open(out_dir/"benchmark.json","w"), default=to_py)
model.save(out_dir/"model.keras")
pickle.dump(scaler, open(out_dir/"scaler.pkl","wb"))
logging.info("Saved artefacts to %s/", out_dir)

