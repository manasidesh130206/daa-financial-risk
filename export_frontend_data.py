"""Exports a small real-data sample (PD + exposure) for the browser demo -> frontend/data.js"""
import os, json, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
import credit_risk_engine as E
pds, ead, info = E.load_and_fit(E.CSV)
idx = np.random.default_rng(7).choice(len(pds), 800, replace=False)
data = {"pd": [round(float(x), 4) for x in pds[idx]], "ead": [int(x) for x in ead[idx]], "model": info}
open(os.path.join(os.path.dirname(__file__), "frontend", "data.js"), "w").write("window.PORTFOLIO = " + json.dumps(data) + ";")
print("wrote frontend/data.js", info)
