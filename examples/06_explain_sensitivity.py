"""Explainability and sensitivity analysis (0.2.0)."""

import dfre
from dfre.core.sensitivity import partial_derivatives, tornado, elasticity

signals = {"M": 0.30, "V": 0.10, "D": 0.45, "U": 0.20, "freshness": 0.60}

# 1. Why is the risk what it is?
e = dfre.explain(signals)
print(e.text)
for name in e.ranked():
    print(f"  {name:12s} share={e.shares[name]:.1%}  contribution={e.contributions[name]:.4f}")

# 2. Which signal should we fix first? (highest gradient = fastest risk reduction)
grads = partial_derivatives(signals)
best = max(grads, key=grads.get)
print(f"\nFastest risk reduction comes from improving: {best} (dR/ds = {grads[best]:.4f})")

# 3. Tornado: how much does each signal swing the risk by ±0.1?
print("\nTornado (delta=0.1):")
for name, row in tornado(signals, delta=0.1).items():
    print(f"  {name:12s} {row['risk_minus']:.4f} .. {row['risk_plus']:.4f}  swing={row['swing']:.4f}")

# 4. Elasticity: % risk change per 1% signal change
print("\nElasticity:", {k: round(v, 3) for k, v in elasticity(signals).items()})
