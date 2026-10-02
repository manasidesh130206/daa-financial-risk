/* Credit-risk engine (JS port of backend/factorfold_algorithm.py). Runs in browser and Node. */
(function (root) {
  const SQ2PI = Math.sqrt(2 * Math.PI);

  function erfc(x) { // Numerical Recipes erfcc, rel. error < 1.2e-7
    const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? r : 2 - r;
  }
  const ndtr = (x) => 0.5 * erfc(-x / Math.SQRT2);
  const npdf = (x) => Math.exp(-0.5 * x * x) / SQ2PI;

  function ndtri(p) { // Acklam's inverse normal CDF
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    if (p < 0.02425) { const q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { const q = Math.sqrt(-2 * Math.log(1 - p));
      return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    const q = p - 0.5, r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  function rng(seed) { // mulberry32 + Box-Muller
    let s = seed >>> 0, spare = null;
    const u = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
    return () => { if (spare !== null) { const v = spare; spare = null; return v; }
      const r = Math.sqrt(-2 * Math.log(1 - u())), a = 2 * Math.PI * u(); spare = r * Math.sin(a); return r * Math.cos(a); };
  }

  // portfolio of N loans resampled from the real sample
  function makePortfolio(sample, N, seed = 1) {
    const pd = new Float64Array(N), ead = new Float64Array(N), M = sample.pd.length;
    let s = seed * 2654435761 >>> 0;
    for (let i = 0; i < N; i++) { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; const k = s % M; pd[i] = sample.pd[k]; ead[i] = sample.ead[k]; }
    return { pd, ead };
  }

  // ---- FactorFold: compress -> quadrature -> mixture quantile
  function factorfold(pf, o) {
    const B = o.B || 64, K = o.K || 121, lo = -5, hi = 3, a = Math.sqrt(o.rho), b = Math.sqrt(1 - o.rho), lgd = o.lgd;
    const W = new Float64Array(B), W2 = new Float64Array(B), WP = new Float64Array(B);
    for (let i = 0; i < pf.pd.length; i++) { // O(N) streaming pass
      const x = ndtri(Math.min(1 - 1e-6, Math.max(1e-6, pf.pd[i])));
      const k = Math.min(B - 1, Math.max(0, Math.floor((x - lo) / (hi - lo) * B)));
      W[k] += pf.ead[i]; W2[k] += pf.ead[i] * pf.ead[i]; WP[k] += pf.ead[i] * pf.pd[i];
    }
    const c = [], w1 = [], w2 = [];
    for (let k = 0; k < B; k++) if (W[k] > 0) { c.push(ndtri(WP[k] / W[k])); w1.push(W[k] * lgd); w2.push(W2[k] * lgd * lgd); }
    const nb = c.length, zs = [], om = [], m = [], sd = []; let tot = 0;
    for (let j = 0; j < K; j++) { const z = -6 + 12 * j / (K - 1); zs.push(z); const w = npdf(z); om.push(w); tot += w; }
    for (let j = 0; j < K; j++) {
      om[j] /= tot; let mm = 0, vv = 0;
      for (let k = 0; k < nb; k++) { const q = ndtr((c[k] - a * zs[j]) / b); mm += q * w1[k]; vv += q * (1 - q) * w2[k]; }
      m.push(mm); sd.push(Math.sqrt(Math.max(vv, 1e-9)));
    }
    const F = (x) => { let s = 0; for (let j = 0; j < K; j++) s += om[j] * ndtr((x - m[j]) / sd[j]); return s; };
    const totalW = w1.reduce((s, v) => s + v, 0), out = { EL: 0 };
    for (let j = 0; j < K; j++) out.EL += om[j] * m[j];
    for (const al of [0.95, 0.99, 0.999]) {
      let l = 0, h = totalW;
      for (let it = 0; it < 50; it++) { const mid = 0.5 * (l + h); if (F(mid) < al) l = mid; else h = mid; }
      const v = 0.5 * (l + h); let es = 0;
      for (let j = 0; j < K; j++) { const d = (v - m[j]) / sd[j]; es += om[j] * (m[j] * (1 - ndtr(d)) + sd[j] * npdf(d)); }
      const t = String(al * 100); out["VaR" + t] = v; out["ES" + t] = es / (1 - al);
    }
    out.density = (x) => { let s = 0; for (let j = 0; j < K; j++) s += om[j] * npdf((x - m[j]) / sd[j]) / sd[j]; return s; };
    out.totalW = totalW;
    return out;
  }

  // ---- brute-force Monte Carlo (one-factor Gaussian copula)
  function monteCarlo(pf, S, o) {
    const N = pf.pd.length, g = rng(o.seed || 42), a = Math.sqrt(o.rho), b = Math.sqrt(1 - o.rho);
    const c = new Float64Array(N), w = new Float64Array(N), L = new Float64Array(S);
    for (let i = 0; i < N; i++) { c[i] = ndtri(Math.min(1 - 1e-6, Math.max(1e-6, pf.pd[i]))); w[i] = pf.ead[i] * o.lgd; }
    for (let s = 0; s < S; s++) { const z = a * g(); let loss = 0;
      for (let i = 0; i < N; i++) if (z + b * g() < c[i]) loss += w[i]; L[s] = loss; }
    const sorted = Float64Array.from(L).sort(), out = { EL: L.reduce((s, v) => s + v, 0) / S, losses: L };
    for (const al of [0.95, 0.99, 0.999]) {
      const k = Math.min(S - 1, Math.floor(al * S)); let t = 0; for (let i = k; i < S; i++) t += sorted[i];
      const key = String(al * 100); out["VaR" + key] = sorted[k]; out["ES" + key] = t / (S - k);
    }
    return out;
  }

  root.Engine = { factorfold, monteCarlo, makePortfolio, ndtr, ndtri };
  if (typeof module !== "undefined") module.exports = root.Engine;
})(typeof window !== "undefined" ? window : globalThis);
