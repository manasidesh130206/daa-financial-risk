# Credit Portfolio Risk Lab (Q17: Financial Risk Analysis)

Computational system for evaluating credit-portfolio risk over many scenarios, a study of how cost grows with problem size, and our own algorithm, **FactorFold**.

## Layout
- `backend/credit_risk_engine.py`: data cleaning, logistic-regression PD model, Monte Carlo (naive, vectorised, bucketed), scaling study. Run: `python backend/credit_risk_engine.py`
- `backend/factorfold_algorithm.py`: our algorithm (Compress, Quadrature, Mixture-quantile) plus validation against Monte Carlo. Run: `python backend/factorfold_algorithm.py`
- `frontend/index.html`: interactive dashboard. Open it directly in a browser (no server needed).
- `export_frontend_data.py`: re-creates `frontend/data.js` (800-loan real sample) from the dataset.
- `data/`: credit_risk_dataset.csv. `results/`: charts and CSVs from the backend runs.

Setup: `pip install -r requirements.txt`

## FactorFold in brief
Given the economy factor Z, portfolio loss is roughly Normal, so total loss is a mixture of normals over Z.
1. Compress: one streaming pass puts loans into bins on the probit(PD) axis, keeping sum(EAD), sum(EAD^2) and EAD-weighted PD per bin.
2. Quadrature: K fixed nodes for Z; conditional mean and variance cost O(bins) each.
3. Mixture: loss CDF = weighted sum of normal CDFs. VaR by bisection, expected shortfall in closed form.

Time O(N + B*K), independent of the number of scenarios. Space O(B + K). Reading every loan is unavoidable, so Omega(N) is the lower bound.

## Results (32,574 loans)
- FactorFold 15 ms vs Monte Carlo (50k scenarios) 23.6 s. 10 million loans in 0.39 s.
- Expected loss within 0.02% of Monte Carlo; VaR95 within 0.4%; VaR99 0.7%; VaR99.9 2% (about Monte Carlo's own sampling noise).

## Assumptions and limits
- LGD 45% and asset correlation 0.15 are assumptions (adjustable in the dashboard).
- Normal approximation of conditional loss is weaker for very small or concentrated portfolios.
- Python uses Gauss-Hermite nodes (K <= 200; larger K overflows). The browser port uses equally spaced nodes on Z in [-6, 6].
- One systematic factor only.
