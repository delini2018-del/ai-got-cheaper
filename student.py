"""Tiny one-hidden-layer student in NumPy, trained either on human labels ('hard') or to imitate the teacher's
softened probabilities ('distill', Hinton et al. 2015). Usage: python student.py HIDDEN MODE [LABELLED_N]
LABELLED_N: train with only N human-labelled images; in distill mode the teacher also labels the rest."""
import sys, json, numpy as np
from confidence_lab import load_fashion
H, MODE = int(sys.argv[1]), sys.argv[2]; NLAB = int(sys.argv[3]) if len(sys.argv) > 3 else 0
Xtr, ytr, _, _, Xte, yte = load_fashion(seed=0); Pt = np.load("results/teacher_train_probs.npy")
T, EPOCHS, LR, BS = 4.0, 15, 2e-3, 128
rng = np.random.default_rng(0)
if NLAB:
    lab = rng.permutation(len(Xtr))[:NLAB]
    if MODE == "hard": Xtr, ytr, Pt = Xtr[lab], ytr[lab], Pt[lab]
Y = np.eye(10, dtype=np.float32)[ytr]
def soften(P):
    z = np.log(np.clip(P, 1e-9, 1)) / T; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
target = Y if MODE == "hard" else (0.2 * Y + 0.8 * soften(Pt) if not NLAB else soften(Pt))
if NLAB and MODE == "distill":   # human labels where we have them, teacher's soft labels everywhere else
    target = soften(Pt); target[lab] = 0.5 * Y[lab] + 0.5 * target[lab]
W1 = rng.normal(0, np.sqrt(2 / 784), (784, H)).astype(np.float32); b1 = np.zeros(H, np.float32)
W2 = rng.normal(0, np.sqrt(2 / H), (H, 10)).astype(np.float32); b2 = np.zeros(10, np.float32)
params = [W1, b1, W2, b2]; m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]; t = 0
temp = 1.0 if MODE == "hard" else T
for ep in range(EPOCHS):
    for s in range(0, len(Xtr), BS):
        idx = rng.permutation(len(Xtr))[:BS] if False else None
    order = rng.permutation(len(Xtr))
    for s in range(0, len(order), BS):
        i = order[s:s + BS]; x, y = Xtr[i], target[i]
        h = np.maximum(0, x @ W1 + b1); z = (h @ W2 + b2) / temp
        z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        dz = (p - y) / len(i) / temp; gW2 = h.T @ dz; gb2 = dz.sum(0); dh = dz @ W2.T * (h > 0)
        grads = [x.T @ dh, dh.sum(0), gW2, gb2]; t += 1
        for k, (pa, g) in enumerate(zip(params, grads)):
            m[k] = 0.9 * m[k] + 0.1 * g; v[k] = 0.999 * v[k] + 0.001 * g * g
            pa -= LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
acc = float(((np.maximum(0, Xte @ W1 + b1) @ W2 + b2).argmax(1) == yte).mean())
def qacc(bits):
    def q(W): s = np.abs(W).max() / (2 ** (bits - 1) - 1); return np.round(W / s) * s
    return float(((np.maximum(0, Xte @ q(W1) + b1) @ q(W2) + b2).argmax(1) == yte).mean())
R = dict(hidden=H, mode=MODE, labelled=NLAB or len(Xtr), acc=acc, macs=784 * H + H * 10, params=int(sum(p.size for p in params)),
         int8_acc=qacc(8), int4_acc=qacc(4))
tag = f"h{H}_{MODE}" + (f"_n{NLAB}" if NLAB else ""); json.dump(R, open(f"results/{tag}.json", "w")); print(tag, R)
