"""Escalation policies (anti-flapping) and HTTP serving (0.2.0)."""

from dfre import DFRE, EscalationPolicy, Policy

# --- Escalation: 3 consecutive HIGH batches escalate to CRITICAL -------------
policy = EscalationPolicy(Policy(), consecutive=3, escalate_at="HIGH", cooldown=2)
model = DFRE(policy=None)  # bands handled by the escalation wrapper here

risks = [0.75, 0.78, 0.72, 0.10, 0.08, 0.12]
print("Escalation demo:")
for i, r in enumerate(risks):
    band = policy.classify(r)
    print(f"  batch {i}: raw risk={r:.2f} -> effective band={band}")

# --- Serve over HTTP ----------------------------------------------------------
# pip install "dfre[serve]"
#   python -m dfre.integrations.fastapi_ext --port 8000
# or:
#   dfre serve --port 8000 --model dfre_model.json
#
# Then:
#   curl -X POST http://127.0.0.1:8000/score \
#        -H 'Content-Type: application/json' \
#        -d '{"signals": {"M": 0.1, "V": 0.2, "D": 0.1, "U": 0.3}, "n": 5000}'
