"""Teacher = ensemble of three previously trained MLPs (512-256, 1024-512, 256). Saves its soft predictions."""
import pickle, json, numpy as np
from confidence_lab import load_fashion
Xtr, ytr, Xva, yva, Xte, yte = load_fashion(seed=0)
ms = [pickle.load(open(f"results/models/{n}.pkl", "rb")) for n in ["main", "w1024", "w256"]]
def macs(m): return int(sum(w.shape[0] * w.shape[1] for w in m.coefs_))
Ptr = np.mean([m.predict_proba(Xtr) for m in ms], 0); Pte = np.mean([m.predict_proba(Xte) for m in ms], 0)
np.save("results/teacher_train_probs.npy", Ptr.astype(np.float32))
R = dict(teacher_acc=float((Pte.argmax(1) == yte).mean()), members=[float((m.predict(Xte) == yte).mean()) for m in ms],
         teacher_macs=sum(macs(m) for m in ms), teacher_params=int(sum(w.size for m in ms for w in m.coefs_) + sum(b.size for m in ms for b in m.intercepts_)))
json.dump(R, open("results/teacher.json", "w"), indent=1); print(R)
