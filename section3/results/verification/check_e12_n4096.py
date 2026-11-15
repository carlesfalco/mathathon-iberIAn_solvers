"""Reproduce the coordinator's E12 cost check; run from the project root."""
import json
from unified.explore.e12_oseen.oseen import area_problem
from unified.explore.e12_oseen.harness import area_case, dense_solve

p = area_problem(4096, 320.0)
xd = dense_solve(p.A, p.b)
out = {}
for rule in ("strong", "dir:3,3,1"):
    r = area_case(p, rule, "exact", xd)
    out[rule] = r
    print(rule, {k: r[k] for k in ("skel_total", "root", "stored", "flops", "err", "backward", "skel_max")}, flush=True)
with open("results/verification/e12-n4096.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
b = out["strong"]
d = out["dir:3,3,1"]
print("ratios", {k: d[k] / b[k] for k in ("skel_total", "stored", "flops", "root")})
