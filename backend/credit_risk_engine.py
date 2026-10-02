"""
Q17 - Financial Risk Analysis
Monte Carlo credit-portfolio risk engine + computational scaling study.

Pipeline
  1. Clean data, fit a logistic-regression PD (probability of default) model.
  2. Treat each loan as an exposure: EAD = loan_amnt, LGD = 45% (assumption).
  3. Simulate portfolio loss with a one-factor Gaussian copula (Vasicek):
        default_i  <=>  sqrt(rho)*Z + sqrt(1-rho)*eps_i  <  Phi^-1(PD_i)
     Z = systematic (economy) factor, eps_i = idiosyncratic shock.
  4. Risk metrics: EL, VaR(95/99/99.9), Expected Shortfall(99), unexpected loss.
  5. Scaling study over N (loans) and S (scenarios), three solvers:
        naive   - pure-Python double loop            O(N*S), huge constant
        vector  - chunked NumPy Monte Carlo           O(N*S), small constant, O(chunk*N) memory
        bucket  - PD-bucketed large-portfolio approx  O(B*S), independent of N
Run:  python credit_risk_engine.py  [path_to_csv]
"""
import sys, time, json
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data", "credit_risk_dataset.csv")
OUT = os.path.join(HERE, "..", "results")
RHO, LGD, SEED = 0.15, 0.45, 42
ALPHAS = (0.95, 0.99, 0.999)


# ----------------------------------------------------------------- 1. data + PD model
def load_and_fit(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df[(df.person_age <= 100) & (df.person_emp_length.fillna(0) <= 60)].copy()
    df["person_emp_length"] = df.person_emp_length.fillna(df.person_emp_length.median())
    df["loan_int_rate"] = df.loan_int_rate.fillna(df.groupby("loan_grade").loan_int_rate.transform("median"))
    df["log_income"] = np.log(df.person_income)
    X = pd.get_dummies(
        df.drop(columns=["loan_status", "person_income"]),
        columns=["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"],
        drop_first=True,
    ).astype(float)
    y = df.loan_status.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    auc = roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1])
    pd_all = clf.predict_proba(sc.transform(X))[:, 1]
    pd_all = np.clip(pd_all, 1e-4, 0.999)
    info = dict(rows_raw=n0, rows_clean=len(df), auc=round(auc, 4),
                mean_pd=round(float(pd_all.mean()), 4), observed_default=round(float(y.mean()), 4))
    return pd_all, df.loan_amnt.values.astype(float), info


# ----------------------------------------------------------------- 2. solvers
def metrics(losses):
    out = {"EL": losses.mean(), "std": losses.std()}
    for a in ALPHAS:
        v = np.quantile(losses, a)
        out[f"VaR{a*100:g}"] = v
        out[f"ES{a*100:g}"] = losses[losses >= v].mean()
    out["UL99"] = out["VaR99"] - out["EL"]
    return out


def sim_vector(pds, ead, S, rho=RHO, lgd=LGD, seed=SEED, max_elems=4_000_000):
    """Chunked NumPy Monte Carlo. Peak memory ~ chunk*N, not S*N."""
    rng = np.random.default_rng(seed)
    N = len(pds)
    c = ndtri(pds).astype(np.float32)
    w = (ead * lgd).astype(np.float32)
    a, b = np.float32(np.sqrt(rho)), np.float32(np.sqrt(1 - rho))
    chunk = max(1, min(S, max_elems // N))
    losses = np.empty(S)
    for s in range(0, S, chunk):
        m = min(chunk, S - s)
        z = rng.standard_normal((m, 1), dtype=np.float32)
        eps = rng.standard_normal((m, N), dtype=np.float32)
        losses[s:s + m] = ((a * z + b * eps) < c).astype(np.float32) @ w
    return losses


def sim_naive(pds, ead, S, rho=RHO, lgd=LGD, seed=SEED):
    """Reference implementation: explicit Python loops (what not to do at scale)."""
    rng = np.random.default_rng(seed)
    N = len(pds)
    c = ndtri(pds).tolist()
    w = (ead * lgd).tolist()
    a, b = rho ** 0.5, (1 - rho) ** 0.5
    losses = np.zeros(S)
    for s in range(S):
        z = rng.standard_normal()
        eps = rng.standard_normal(N).tolist()
        tot = 0.0
        for i in range(N):
            if a * z + b * eps[i] < c[i]:
                tot += w[i]
        losses[s] = tot
    return losses


def make_buckets(pds, ead, B=200):
    """Aggregate loans into B PD-buckets -> (mean-PD threshold, total weighted EAD)."""
    edges = np.quantile(pds, np.linspace(0, 1, B + 1))
    idx = np.clip(np.searchsorted(edges, pds, side="right") - 1, 0, B - 1)
    wsum = np.bincount(idx, weights=ead, minlength=B)
    psum = np.bincount(idx, weights=pds * ead, minlength=B)
    ok = wsum > 0
    return (psum[ok] / wsum[ok]), wsum[ok]


def sim_bucket(bpd, bead, S, rho=RHO, lgd=LGD, seed=SEED, max_elems=4_000_000):
    """Large-portfolio approximation: loss | Z = sum_b EAD_b*LGD*Phi((c_b - sqrt(rho) Z)/sqrt(1-rho)).
    Idiosyncratic risk is diversified away, so cost is O(B*S) regardless of N."""
    rng = np.random.default_rng(seed)
    c = ndtri(bpd)
    w = bead * lgd
    B = len(c)
    chunk = max(1, min(S, max_elems // B))
    losses = np.empty(S)
    for s in range(0, S, chunk):
        m = min(chunk, S - s)
        z = rng.standard_normal((m, 1))
        losses[s:s + m] = ndtr((c - np.sqrt(rho) * z) / np.sqrt(1 - rho)) @ w
    return losses


def timeit(f, *a, **k):
    t = time.perf_counter(); r = f(*a, **k); return time.perf_counter() - t, r


def peak_mem_mb(f, *a, **k):
    import tracemalloc
    tracemalloc.start(); f(*a, **k); _, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return p / 1e6


# ----------------------------------------------------------------- 3. experiments
def main():
    pds, ead, info = load_and_fit(CSV)
    N0 = len(pds)
    print("DATA/MODEL:", info)
    res = {"model": info}

    # --- base-case risk report (full portfolio)
    S_base = 50_000
    t, L = timeit(sim_vector, pds, ead, S_base)
    base = metrics(L)
    bpd, bead = make_buckets(pds, ead)
    tb, Lb = timeit(sim_bucket, bpd, bead, S_base)
    bb = metrics(Lb)
    res["base"] = {"N": N0, "S": S_base, "full_mc_sec": t, "bucket_sec": tb,
                   "total_EAD": ead.sum(), "full_mc": base, "bucket": bb}
    print(f"\nBASE CASE  N={N0:,}  S={S_base:,}  total exposure={ead.sum():,.0f}")
    print(f"{'metric':8s}{'full MC':>16s}{'bucketed':>16s}{'diff %':>9s}")
    for k in base:
        print(f"{k:8s}{base[k]:16,.0f}{bb[k]:16,.0f}{100*(bb[k]/base[k]-1):9.2f}")
    print(f"time: full MC {t:.1f}s | bucketed {tb:.3f}s  (speed-up {t/tb:.0f}x)")

    # --- A: naive vs vector (small, to show constant-factor gap)
    rows = []
    print("\nA) naive vs vectorised (S=200)")
    for N in (250, 500, 1000, 2000):
        sub = np.random.default_rng(1).choice(N0, N, replace=False)
        tn, _ = timeit(sim_naive, pds[sub], ead[sub], 200)
        tv, _ = timeit(sim_vector, pds[sub], ead[sub], 200)
        rows.append(dict(exp="A", method="naive", N=N, S=200, sec=tn))
        rows.append(dict(exp="A", method="vector", N=N, S=200, sec=tv))
        print(f"  N={N:5d}  naive {tn:7.3f}s  vector {tv:7.4f}s  speed-up {tn/tv:6.0f}x")

    # --- B: scale N (portfolio size), S fixed
    print("\nB) scaling in N (S=1000)")
    bp = np.random.default_rng(2)
    for N in (1_000, 3_000, 10_000, 30_000, 100_000, 300_000):
        idx = bp.integers(0, N0, N)
        p, e = pds[idx], ead[idx]
        tv, _ = timeit(sim_vector, p, e, 1000)
        mem = peak_mem_mb(sim_vector, p, e, 100)           # chunked peak (S-independent)
        full_mb = 1000 * N * 4 / 1e6                         # if the S x N matrix were held at once
        bp_, be_ = make_buckets(p, e)
        tb, _ = timeit(sim_bucket, bp_, be_, 1000)
        rows += [dict(exp="B", method="vector", N=N, S=1000, sec=tv, peak_mb=mem, full_matrix_mb=full_mb),
                 dict(exp="B", method="bucket", N=N, S=1000, sec=tb)]
        print(f"  N={N:7,d}  vector {tv:7.2f}s  chunked-mem {mem:6.1f}MB (full matrix {full_mb:8.0f}MB)  bucket {tb:.4f}s")

    # --- C: scale S (scenarios), N fixed at full portfolio
    print(f"\nC) scaling in S (N={N0:,})")
    for S in (500, 1_000, 3_000, 10_000, 30_000):
        tv, _ = timeit(sim_vector, pds, ead, S)
        rows.append(dict(exp="C", method="vector", N=N0, S=S, sec=tv))
        print(f"  S={S:7,d}  vector {tv:7.2f}s")
    for S in (1_000, 10_000, 100_000, 1_000_000):
        tb, _ = timeit(sim_bucket, bpd, bead, S)
        rows.append(dict(exp="C", method="bucket", N=N0, S=S, sec=tb))
        print(f"  S={S:9,d}  bucket {tb:7.3f}s")

    # --- D: statistical convergence of VaR with S (replicated runs)
    print("\nD) Monte-Carlo error of VaR vs S (N=5,000, 8 replicates)")
    sub = np.random.default_rng(3).choice(N0, 5000, replace=False)
    conv = []
    for S in (500, 1_000, 2_000, 5_000, 10_000, 20_000):
        reps = [metrics(sim_vector(pds[sub], ead[sub], S, seed=100 + r)) for r in range(8)]
        row = {"S": S}
        for a in ALPHAS:
            k = f"VaR{a*100:g}"
            v = np.array([r[k] for r in reps])
            row[k + "_relSE"] = v.std(ddof=1) / v.mean()
        conv.append(row)
        print(f"  S={S:6d}  relSE  VaR95 {row['VaR95_relSE']:6.2%}  VaR99 {row['VaR99_relSE']:6.2%}  VaR99.9 {row['VaR99.9_relSE']:6.2%}")
    conv = pd.DataFrame(conv)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/scaling_results.csv", index=False)
    conv.to_csv(f"{OUT}/convergence_results.csv", index=False)

    # --- fitted exponents  (time ~ N^k)
    def slope(d, x):
        return np.polyfit(np.log(d[x]), np.log(d.sec), 1)[0]
    ex = {
        "vector_vs_N": slope(df[(df.exp == "B") & (df.method == "vector") & (df.N >= 10_000)], "N"),
        "bucket_vs_N": slope(df[(df.exp == "B") & (df.method == "bucket")], "N"),
        "vector_vs_S": slope(df[(df.exp == "C") & (df.method == "vector") & (df.S >= 3000)], "S"),
        "bucket_vs_S": slope(df[(df.exp == "C") & (df.method == "bucket") & (df.S >= 10_000)], "S"),
    }
    # relSE ~ S^-0.5 check at 99.9
    ex["relSE_vs_S_VaR99.9"] = np.polyfit(np.log(conv.S), np.log(conv["VaR99.9_relSE"]), 1)[0]
    res["exponents"] = ex
    print("\nFITTED EXPONENTS (log-log slope):", {k: round(v, 2) for k, v in ex.items()})
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=2, default=float)

    # --- figures
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()
    ax[0].hist(L / 1e6, bins=120, color="#4c78a8")
    for a, col in zip(ALPHAS, ("#f58518", "#e45756", "#72001c")):
        ax[0].axvline(base[f"VaR{a*100:g}"] / 1e6, color=col, ls="--", label=f"VaR {a*100:g}%")
    ax[0].axvline(base["EL"] / 1e6, color="k", label="EL")
    ax[0].set(title=f"Portfolio loss distribution (N={N0:,}, S={S_base:,})", xlabel="Loss (millions)", ylabel="scenarios")
    ax[0].legend()

    d = df[df.exp == "A"]
    for m, g in d.groupby("method"):
        ax[1].loglog(g.N, g.sec, "o-", label=m)
    ax[1].set(title="Naive loops vs vectorised (S=200)", xlabel="N loans", ylabel="seconds"); ax[1].legend(); ax[1].grid(alpha=.3, which="both")

    d = df[df.exp == "B"]
    for m, g in d.groupby("method"):
        ax[2].loglog(g.N, g.sec, "o-", label=f"{m}  (slope {slope(g[g.N>=10_000] if m=='vector' else g,'N'):.2f})")
    ax[2].set(title="Runtime vs portfolio size N (S=1000)", xlabel="N loans", ylabel="seconds"); ax[2].legend(); ax[2].grid(alpha=.3, which="both")

    d = df[df.exp == "C"]
    for m, g in d.groupby("method"):
        ax[3].loglog(g.S, g.sec, "o-", label=m)
    ax[3].set(title=f"Runtime vs scenarios S (N={N0:,})", xlabel="S scenarios", ylabel="seconds"); ax[3].legend(); ax[3].grid(alpha=.3, which="both")

    d = df[(df.exp == "B") & (df.method == "vector")]
    ax[4].loglog(d.N, d.full_matrix_mb, "o-", label="hold full S x N matrix")
    ax[4].loglog(d.N, d.peak_mb, "o-", label="chunked (this engine)")
    ax[4].set(title="Memory vs N (S=1000)", xlabel="N loans", ylabel="MB"); ax[4].legend(); ax[4].grid(alpha=.3, which="both")

    for a, col in zip(ALPHAS, ("#f58518", "#e45756", "#72001c")):
        ax[5].loglog(conv.S, conv[f"VaR{a*100:g}_relSE"], "o-", color=col, label=f"VaR {a*100:g}%")
    ax[5].loglog(conv.S, 0.5 * conv.S.iloc[0] ** .5 * conv.S ** -.5 * conv["VaR99.9_relSE"].iloc[0] * 2, "k:", label="1/sqrt(S)")
    ax[5].set(title="Estimator error vs S (more S needed in the tail)", xlabel="S scenarios", ylabel="relative std. error"); ax[5].legend(); ax[5].grid(alpha=.3, which="both")
    plt.tight_layout(); plt.savefig(f"{OUT}/scaling_analysis.png", dpi=130)
    print("\nSaved: scaling_analysis.png, scaling_results.csv, convergence_results.csv, results.json")


if __name__ == "__main__":
    main()
