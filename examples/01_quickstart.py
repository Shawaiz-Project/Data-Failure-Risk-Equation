"""The simplest possible DFRE integration."""

import dfre

# One line. That's it.
result = dfre.score(missing=0.08, invalid=0.04, drift=0.12, uncertainty=0.20, n=1000)

print(f"R = {result.risk:.4f}")
print(f"Band    : {result.band}")
print(f"Action  : {result.action}")
print(f"Main    : {result.main_component:.4f}")
print(f"Pairwise: {result.pairwise_component:.4f}")
print(f"Four-way: {result.four_way_component:.4f}")

# Which signal is driving the risk?
e = dfre.explain({"M": 0.08, "V": 0.04, "D": 0.12, "U": 0.20})
print(f"\nTop driver: {e.top_driver}")
print(e.text)
