import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIRES_PATH = "forestfires.csv"
RICE_PATH = "Rice_Cammeo_Osmancik.arff"
OUT_DIR = "results"
SHOW_PLOTS = False
SEEDS = [0, 1, 2, 3, 4]
BATCH = 32
AE_EPOCHS = 100
AE_LR = 1e-3
HEAD_EPOCHS = 30
FT_LR = 1e-3
MAPE_MIN_Y = 1.0

DATASETS = {
    "Forest Fires (area, регрессия)": dict(task="regression", hidden=[32, 24, 16, 8], batch=32, epochs=150, fracs=[1.0, 0.3]),
    "Rice (Cammeo/Osmancik, классификация)": dict(task="classification", hidden=[32, 24, 16, 8], batch=32, epochs=100, fracs=[1.0, 0.05]),
}

M_BASE = "Без предобучения"
M_ONLY = "Только предобучение (без дообучения)"
M_PRE = "С предобучением + дообучение"


def load_fires():
    df = pd.read_csv(FIRES_PATH)
    y = df["area"].values.astype(float)
    X = pd.get_dummies(df.drop(columns="area"), columns=["month", "day"]).astype(float).values
    return X, y, None


def load_rice():
    names, rows, in_data = [], [], False
    with open(RICE_PATH, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("%"):
                continue
            if s.lower().startswith("@attribute"):
                names.append(s.split()[1])
            elif s.lower().startswith("@data"):
                in_data = True
            elif in_data:
                rows.append(s.split(","))
    df = pd.DataFrame(rows, columns=names)
    y = (df["Class"].str.strip() == "Osmancik").astype(float).values
    X = df.drop(columns="Class").astype(float).values
    return X, y, ["Cammeo", "Osmancik"]


def train_test_split(X, y, task, seed, test_size=0.2):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(X))
    groups = [idx[y == c] for c in (0, 1)] if task == "classification" else [idx]
    tr, te = [], []
    for g in groups:
        g = rng.permutation(g)
        n_te = int(round(len(g) * test_size))
        te += list(g[:n_te]); tr += list(g[n_te:])
    return rng.permutation(tr), rng.permutation(te)


def act(name, z):
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    return z


def dact(name, a):
    return a * (1 - a) if name == "sigmoid" else np.ones_like(a)


class MLP:

    def __init__(self, dims, out_act, rng):
        self.acts = ["sigmoid"] * (len(dims) - 2) + [out_act]
        self.W = [rng.randn(i, o) * np.sqrt(2.0 / (i + o)) for i, o in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros((1, o)) for o in dims[1:]]

    def copy(self):
        m = MLP.__new__(MLP)
        m.acts = list(self.acts)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        return m

    def features(self, X):
        for W, b, a in zip(self.W[:-1], self.b[:-1], self.acts[:-1]):
            X = act(a, X @ W + b)
        return X

    def forward(self, X):
        outs = [X]
        for W, b, a in zip(self.W, self.b, self.acts):
            outs.append(act(a, outs[-1] @ W + b))
        return outs

    def predict(self, X):
        return self.forward(X)[-1]


def loss_value(task, p, y):
    if task == "regression":
        return float(np.mean((p - y) ** 2))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def batches(n, bs, rng):
    perm = rng.permutation(n)
    for s in range(0, n, bs):
        yield perm[s:s + bs]


class Adam:
    def __init__(self, params, lr):
        self.p, self.lr, self.t = params, lr, 0
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]

    def step(self, grads):
        self.t += 1
        for i, (q, g) in enumerate(zip(self.p, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            q -= self.lr * (self.m[i] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[i] / (1 - 0.999 ** self.t)) + 1e-8)


def backprop(net, xb, yb, first_layer=0):
    outs = net.forward(xb)
    delta = (outs[-1] - yb) / len(xb)
    gW, gb = [None] * len(net.W), [None] * len(net.W)
    for k in range(len(net.W) - 1, first_layer - 1, -1):
        gW[k] = outs[k].T @ delta
        gb[k] = delta.sum(0, keepdims=True)
        if k > first_layer:
            delta = (delta @ net.W[k].T) * dact(net.acts[k - 1], outs[k])
    return gW[first_layer:], gb[first_layer:]


def train_supervised(net, Xtr, ytr, Xte, yte, task, epochs, rng, only_last=False):
    k0 = len(net.W) - 1 if only_last else 0
    opt = Adam(net.W[k0:] + net.b[k0:], FT_LR)
    hist = {"train": [], "test": []}
    for _ in range(epochs):
        for idx in batches(len(Xtr), BATCH, rng):
            gW, gb = backprop(net, Xtr[idx], ytr[idx], k0)
            opt.step(gW + gb)
        hist["train"].append(loss_value(task, net.predict(Xtr), ytr))
        hist["test"].append(loss_value(task, net.predict(Xte), yte))
    return hist



def pretrain_with_autoencoders(net, X, rng):
    ae_history, cur = [], X
    for k in range(len(net.W) - 1):
        n_in, n_hid = net.W[k].shape
        ae = MLP([n_in, n_hid, n_in], "linear", rng)
        ae.W[0], ae.b[0] = net.W[k].copy(), net.b[k].copy()
        opt = Adam(ae.W + ae.b, AE_LR)
        losses = []
        for _ in range(AE_EPOCHS):
            total = 0.0
            for idx in batches(len(cur), BATCH, rng):
                xb = cur[idx]
                gW, gb = backprop(ae, xb, xb)
                opt.step(gW + gb)
            losses.append(loss_value("regression", ae.predict(cur), cur))
        ae_history.append(losses)
        net.W[k], net.b[k] = ae.W[0].copy(), ae.b[0].copy()
        cur = act("sigmoid", cur @ net.W[k] + net.b[k])
    return ae_history


def regression_metrics(y_true, y_pred):
    err = y_pred - y_true
    a_true, a_pred = np.expm1(y_true), np.clip(np.expm1(y_pred), 0, None)
    mask = a_true >= MAPE_MIN_Y
    mape = float(np.mean(np.abs(a_pred[mask] - a_true[mask]) / a_true[mask]) * 100) if mask.any() else np.nan
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "R2": float(1 - (err ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()), "MAPE, %": mape}


def classification_metrics(y_true, prob):
    yh, yt = (prob >= 0.5).astype(int), y_true.astype(int)
    tp = int(((yh == 1) & (yt == 1)).sum()); tn = int(((yh == 0) & (yt == 0)).sum())
    fp = int(((yh == 1) & (yt == 0)).sum()); fn = int(((yh == 0) & (yt == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"Accuracy": (tp + tn) / len(yt), "Precision": prec, "Recall": rec, "F1": f1}


def evaluate(net, X, y_true, task, y_scaler):
    out = net.predict(X).ravel()
    if task == "regression":
        out = out * y_scaler[1] + y_scaler[0]
        return regression_metrics(y_true, out), out
    return classification_metrics(y_true, out), out


def run_seed(X_all, y_all, task, hidden, epochs, frac, seed):
    ytask = np.log1p(y_all) if task == "regression" else y_all
    tr, te = train_test_split(X_all, ytask, task, seed)
    if frac < 1.0:  # имитация малой выборки
        tr = np.random.RandomState(seed).permutation(tr)[:max(int(len(tr) * frac), 20)]
    Xtr, Xte, ytr, yte = X_all[tr], X_all[te], ytask[tr], ytask[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd
    y_scaler, ytr_s, yte_s = None, ytr, yte
    if task == "regression":
        y_scaler = (ytr.mean(), ytr.std() + 1e-8)
        ytr_s, yte_s = (ytr - y_scaler[0]) / y_scaler[1], (yte - y_scaler[0]) / y_scaler[1]
    ytr_s, yte_s = ytr_s.reshape(-1, 1), yte_s.reshape(-1, 1)

    dims = [X_all.shape[1]] + hidden + [1]
    out_act = "linear" if task == "regression" else "sigmoid"
    base = MLP(dims, out_act, np.random.RandomState(seed))
    pre = base.copy()

    hist_base = train_supervised(base, Xtr_s, ytr_s, Xte_s, yte_s, task, epochs, np.random.RandomState(seed))
    m_base, out_base = evaluate(base, Xte_s, yte, task, y_scaler)

    rng = np.random.RandomState(seed)
    ae_hist = pretrain_with_autoencoders(pre, Xtr_s, rng)
    train_supervised(pre, Xtr_s, ytr_s, Xte_s, yte_s, task, HEAD_EPOCHS, rng, only_last=True)
    m_only, _ = evaluate(pre, Xte_s, yte, task, y_scaler)

    hist_pre = train_supervised(pre, Xtr_s, ytr_s, Xte_s, yte_s, task, epochs, rng)
    m_pre, out_pre = evaluate(pre, Xte_s, yte, task, y_scaler)
    return dict(metrics={M_BASE: m_base, M_ONLY: m_only, M_PRE: m_pre},
                hist_base=hist_base, hist_pre=hist_pre, ae_hist=ae_hist,
                y_true=yte, out_base=out_base, out_pre=out_pre)

def make_figure(name, frac, task, runs, class_names, path):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for k in range(len(runs[0]["ae_hist"])):
        ax[0, 0].plot(np.mean([r["ae_hist"][k] for r in runs], 0), label=f"Слой {k + 1}")
    ax[0, 0].set(title="Предобучение: потери реконструкции (MSE)", xlabel="Эпоха", ylabel="MSE")
    ax[0, 0].legend(title="Автоэнкодер"); ax[0, 0].grid(alpha=.3)
    for key, label in (("hist_base", M_BASE), ("hist_pre", M_PRE)):
        ax[0, 1].plot(np.mean([r[key]["test"] for r in runs], 0), label=label)
    ax[0, 1].set(title="Ошибка на тесте (среднее по запускам)", xlabel="Эпоха",
                 ylabel="MSE (норм.)" if task == "regression" else "Cross-entropy")
    ax[0, 1].legend(); ax[0, 1].grid(alpha=.3)
    r0 = runs[0]
    for j, (key, label) in enumerate((("out_base", M_BASE), ("out_pre", "С предобучением"))):
        a = ax[1, j]
        if task == "regression":
            t, p = r0["y_true"], r0[key]
            a.scatter(t, p, s=14, alpha=.6)
            lim = [min(t.min(), p.min()), max(t.max(), p.max())]
            a.plot(lim, lim, "r--")
            a.set(title=f"{label}: прогноз и факт", xlabel="Истинное log(1+area)", ylabel="Предсказанное log(1+area)")
        else:
            yh, yt = (r0[key] >= 0.5).astype(int), r0["y_true"].astype(int)
            cm = np.array([[((yt == i) & (yh == k)).sum() for k in (0, 1)] for i in (0, 1)])
            a.imshow(cm, cmap="Blues")
            for (u, v), z in np.ndenumerate(cm):
                a.text(v, u, int(z), ha="center", va="center", fontsize=16)
            a.set(title=f"{label}: матрица ошибок (seed 0)", xlabel="Предсказано", ylabel="Истинно",
                  xticks=[0, 1], yticks=[0, 1], xticklabels=class_names, yticklabels=class_names)
    fig.suptitle(f"{name}\nдоля обучающей выборки = {frac}", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    if not SHOW_PLOTS:
        plt.close(fig)

def main():
    global BATCH
    os.makedirs(OUT_DIR, exist_ok=True)
    loaders = {"Forest Fires (area, регрессия)": load_fires, "Rice (Cammeo/Osmancik, классификация)": load_rice}
    rows = []
    for name, cfg in DATASETS.items():
        X, y, class_names = loaders[name]()
        task = cfg["task"]
        BATCH = cfg["batch"]
        print(f"\n{'=' * 70}\n{name}: {X.shape[0]} объектов, {X.shape[1]} признаков")
        print(f"Архитектура: {[X.shape[1]] + cfg['hidden'] + [1]}")
        for frac in cfg["fracs"]:
            runs = []
            t0 = time.time()
            for seed in SEEDS:
                r = run_seed(X, y, task, cfg["hidden"], cfg["epochs"], frac, seed)
                runs.append(r)
                for method, m in r["metrics"].items():
                    rows.append({"Датасет": name, "Доля train": frac, "Метод": method, "seed": seed, **m})
                print(f"  доля={frac}, seed={seed} готово ({time.time() - t0:.0f} c)")
            make_figure(name, frac, task, runs, class_names,
                        os.path.join(OUT_DIR, f"{name.split()[0]}_frac{frac}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "all_runs.csv"), index=False, encoding="utf-8-sig")
    metric_cols = [c for c in df.columns if c not in ("Датасет", "Доля train", "Метод", "seed")]
    summary = df.groupby(["Датасет", "Доля train", "Метод"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.to_csv(os.path.join(OUT_DIR, "summary.csv"), encoding="utf-8-sig")

    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("\n" + "=" * 70 + "\nИТОГОВЫЕ РЕЗУЛЬТАТЫ (mean по запускам; std в summary.csv)\n" + "=" * 70)
    for (ds, frac), g in df.groupby(["Датасет", "Доля train"], sort=False):
        print(f"\n{ds}, доля обучающей выборки = {frac}")
        cols = [c for c in metric_cols if g[c].notna().any()]
        print(g.groupby("Метод", sort=False)[cols].mean().round(4).to_string())
    print(f"\nГрафики и таблицы сохранены в папку '{OUT_DIR}'")
    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()