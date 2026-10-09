import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
B, O, R, G, P = "#2563eb", "#ea580c", "#dc2626", "#6b7280", "#16a34a"
SRC = "Source: author's experiment on Fashion-MNIST. Compute = multiply-adds per prediction. Code: see repo."
def foot(fig, s=SRC): fig.text(0.01, -0.03, s, fontsize=8.5, color=G)
D = {f.split("/")[-1][:-5]: json.load(open(f)) for f in glob.glob("results/h*.json")}; T = json.load(open("results/teacher.json"))
Hs = [8, 16, 32, 64, 128]; mac = lambda h: D[f"h{h}_hard"]["macs"]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.scatter([T["teacher_macs"]], [100*T["teacher_acc"]], s=180, color="k", zorder=5); ax.annotate(f"Teacher (3-model ensemble)\n{100*T['teacher_acc']:.1f}%, 2.07M ops", (T["teacher_macs"], 100*T["teacher_acc"]), xytext=(-170, -55), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
for key, col, lab in [("hard", B, "Student, all 55,000 human labels"), ("distill_n500", O, "Student, 500 human labels + teacher's labels"), ("hard_n500", G, "Student, 500 human labels only")]:
    ax.plot([mac(h) for h in Hs], [100*D[f"h{h}_{key}"]["acc"] for h in Hs], marker="o", lw=2.5, color=col, label=lab)
ax.set(xscale="log", xlabel="Compute per prediction (log scale)", ylabel="Test accuracy (%)", ylim=(48, 92), title="Most of the skill, a sliver of the compute")
ax.legend(frameon=False, fontsize=10, loc="lower right"); foot(fig); fig.savefig("charts/01_frontier.png"); plt.close()
ns = [500, 2000, 5000]; fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(3)
hv = [100*D[f"h32_hard_n{n}"]["acc"] for n in ns]; dv = [100*D[f"h32_distill_n{n}"]["acc"] for n in ns]
b1 = ax.bar(x-.2, hv, .4, color=G, label="Human labels only"); b2 = ax.bar(x+.2, dv, .4, color=O, label="Human labels + teacher labels the rest")
for bb in list(b1)+list(b2): ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+.4, f"{bb.get_height():.1f}%", ha="center", fontsize=10.5)
ax.axhline(100*T["teacher_acc"], ls="--", color="k", lw=1); ax.text(2.45, 100*T["teacher_acc"]+.4, "teacher", ha="right", fontsize=10)
ax.set(xticks=x, ylim=(65, 93), ylabel="Test accuracy (%)", xlabel="Human-labelled training images", title="The expensive model's real job: teaching"); ax.set_xticklabels([f"{n:,}" for n in ns])
ax.legend(frameon=False, fontsize=10, loc="upper left"); foot(fig, SRC.replace("Compute", "Student with 32 hidden units. Compute")); fig.savefig("charts/02_scarce_labels.png"); plt.close()
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(3); hs = [32, 64, 128]
for i, (k, lab, col) in enumerate([("acc", "32-bit (original)", B), ("int8_acc", "8-bit (4x smaller)", P), ("int4_acc", "4-bit (8x smaller)", R)]):
    v = [100*D[f"h{h}_hard"][k] for h in hs]; bb = ax.bar(x+(i-1)*.27, v, .27, color=col, label=lab)
    for b_, val in zip(bb, v): ax.text(b_.get_x()+b_.get_width()/2, val+.3, f"{val:.1f}", ha="center", fontsize=9)
ax.set(xticks=x, ylim=(70, 92), ylabel="Test accuracy (%)", title="Shrinking the numbers: 8-bit is free, 4-bit is not"); ax.set_xticklabels([f"Student, {h} units" for h in hs])
ax.legend(frameon=False, fontsize=10, loc="upper left"); foot(fig); fig.savefig("charts/03_quantization.png"); plt.close()
print("ok")
