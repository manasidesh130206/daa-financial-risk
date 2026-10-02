"""
CQM - Compress -> Quadrature -> Mixture-quantile   (our own credit-risk algorithm)

Inspired by:  Vasicek one-factor model (conditional independence given Z),
              large-portfolio approximation / bucketing,
              Gauss-Hermite quadrature (replace random sampling by deterministic integration).

IDEA
  Loss | Z is a sum of independent Bernoulli*weight terms  ->  approx Normal(m(Z), v(Z)).
  So portfolio loss is a *mixture of normals* over the one systematic factor Z.
  Instead of simulating S scenarios x N loans, we
    1. COMPRESS : one streaming O(N) pass puts loans into B bins of the probit(PD) axis
                  and keeps only (sum w, sum w^2, w-weighted PD) per bin.  Memory O(B).
    2. QUADRATURE: K Gauss-Hermite nodes for Z; at each node m_k, v_k cost O(B).
    3. MIXTURE  : F(x) = sum_k omega_k * Phi((x - m_k)/sd_k).
                  VaR = root of F(x)=alpha (bisection), ES in closed form. No sampling error.

COMPLEXITY
  time  O(N + B*K + K*I)   (I ~ 60 bisection steps)  -> linear in N, independent of S
  space O(B + K)           (N never stored; works on a stream of chunks)
  lower bound Omega(N) (must read every loan once) -> optimal in N.
"""
import time, sys, os
import numpy as np
from scipy.special import ndtr, ndtri
from numpy.polynomial.hermite_e import hermegauss

RHO, LGD = 0.15, 0.45


class CQM:
    def __init__(self, B=256, K=200, x_lo=-5.0, x_hi=3.0, rho=RHO, lgd=LGD):
        self.B, self.K, self.rho, self.lgd = B, K, rho, lgd
        self.lo, self.hi = x_lo, x_hi                      # probit(PD) range covered by bins
        self.W = np.zeros(B); self.W2 = np.zeros(B); self.WP = np.zeros(B)

    # ---- step 1: streaming compression, O(chunk) work per chunk, O(B) state
    def update(self, pd, ead):
        x = ndtri(np.clip(pd, 1e-6, 1 - 1e-6))
        idx = np.clip(((x - self.lo) / (self.hi - self.lo) * self.B).astype(np.int64), 0, self.B - 1)
        self.W += np.bincount(idx, ead, self.B)
        self.W2 += np.bincount(idx, ead * ead, self.B)
        self.WP += np.bincount(idx, ead * pd, self.B)

    # ---- steps 2+3: quadrature + mixture quantiles
    def solve(self, alphas=(0.95, 0.99, 0.999)):
        ok = self.W > 0
        W, W2 = self.W[ok] * self.lgd, self.W2[ok] * self.lgd ** 2
        c = ndtri(self.WP[ok] / self.W[ok])                # bin threshold from w-weighted mean PD
        z, om = hermegauss(self.K); om = om / om.sum()
        a, b = np.sqrt(self.rho), np.sqrt(1 - self.rho)
        q = ndtr((c[None, :] - a * z[:, None]) / b)        # K x B conditional PDs
        m = q @ W                                          # conditional mean loss
        sd = np.sqrt(np.maximum((q * (1 - q)) @ W2, 1e-9)) # conditional std (idiosyncratic risk)
        F = lambda x: om @ ndtr((x - m) / sd)
        out = {"EL": float(om @ m), "std": float(np.sqrt(om @ (sd ** 2 + m ** 2) - (om @ m) ** 2))}
        for al in alphas:
            lo, hi = 0.0, W.sum()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if F(mid) < al else (lo, mid)
            v = 0.5 * (lo + hi)
            d = (v - m) / sd
            es = om @ (m * (1 - ndtr(d)) + sd * np.exp(-0.5 * d * d) / np.sqrt(2 * np.pi)) / (1 - al)
            out[f"VaR{al*100:g}"] = v; out[f"ES{al*100:g}"] = float(es)
        out["UL99"] = out["VaR99"] - out["EL"]
        return out


def cqm_run(pd, ead, chunk=1_000_000, **kw):
    e = CQM(**kw)
    for s in range(0, len(pd), chunk):
        e.update(pd[s:s + chunk], ead[s:s + chunk])
    return e.solve()


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import credit_risk_engine as E
    import pandas as pd
    pds, ead, info = E.load_and_fit(E.CSV)
    N0 = len(pds)

    # ---------- 1. accuracy vs brute-force Monte Carlo (S=50k, full N)
    t0 = time.perf_counter(); mc = E.metrics(E.sim_vector(pds, ead, 50_000)); t_mc = time.perf_counter() - t0
    t0 = time.perf_counter(); cq = cqm_run(pds, ead); t_cq = time.perf_counter() - t0
    print(f"ACCURACY  N={N0:,}   MC(S=50k) {t_mc:.1f}s   CQM {t_cq*1000:.0f} ms   speed-up {t_mc/t_cq:,.0f}x")
    print(f"{'metric':8s}{'MC':>14s}{'CQM':>14s}{'diff %':>9s}")
    for k in ("EL", "std", "VaR95", "VaR99", "VaR99.9", "ES99", "UL99"):
        print(f"{k:8s}{mc[k]:14,.0f}{cq[k]:14,.0f}{100*(cq[k]/mc[k]-1):9.2f}")

    # ---------- 2. small portfolio: does the variance term help? (CQM vs LHP-bucket vs MC)
    print("\nSMALL PORTFOLIO (N=300, MC S=200k as truth)")
    sub = np.random.default_rng(5).choice(N0, 300, replace=False)
    tr = E.metrics(E.sim_vector(pds[sub], ead[sub], 200_000, seed=9))
    cs = cqm_run(pds[sub], ead[sub])
    bp, be = E.make_buckets(pds[sub], ead[sub], 50)
    lh = E.metrics(E.sim_bucket(bp, be, 200_000))
    for k in ("VaR95", "VaR99", "ES99"):
        print(f"  {k:6s} MC {tr[k]:12,.0f} | CQM {100*(cs[k]/tr[k]-1):+6.2f}% | LHP-bucket {100*(lh[k]/tr[k]-1):+6.2f}%")

    # ---------- 3. scaling in N up to 10 million (streamed, tiled from real PDs)
    print("\nSCALING IN N (streamed in 1M chunks)")
    rng = np.random.default_rng(1)
    for N in (10**4, 10**5, 10**6, 3 * 10**6, 10**7):
        idx = rng.integers(0, N0, N)
        p, e = pds[idx], ead[idx]
        t0 = time.perf_counter(); cqm_run(p, e); t = time.perf_counter() - t0
        print(f"  N={N:>10,d}  CQM {t:6.3f}s   ({t/N*1e9:5.0f} ns/loan)   state memory ~{(3*256+200*2)*8/1e3:.0f} KB")

    # ---------- 4. parameter sensitivity: bins B and nodes K
    print("\nSENSITIVITY (VaR99 / ES99 error vs MC, full portfolio)")
    for B in (16, 64, 256, 1024):
        r = cqm_run(pds, ead, B=B)
        print(f"  B={B:5d}  VaR99 {100*(r['VaR99']/mc['VaR99']-1):+6.2f}%   ES99 {100*(r['ES99']/mc['ES99']-1):+6.2f}%")
    for K in (20, 50, 100, 200, 400):
        t0 = time.perf_counter(); r = cqm_run(pds, ead, K=K); t = time.perf_counter() - t0
        print(f"  K={K:5d}  VaR99 {100*(r['VaR99']/mc['VaR99']-1):+6.2f}%   ES99 {100*(r['ES99']/mc['ES99']-1):+6.2f}%   {t*1000:.0f} ms")
