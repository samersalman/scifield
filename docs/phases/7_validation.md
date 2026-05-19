# Phase 7 — Validation

## Phase objective

Demonstrate the framework scales beyond surgery/orthopedics and that the
v1 findings generalize. The corpus expands to v2 — 25–40 journals and
500k–1M papers — adding additional surgical specialties (neurosurgery,
cardiothoracic, plastic, vascular) plus 3–5 non-surgical specialties
(cardiology, hematology, oncology, neurology, internal medicine) for
cross-specialty breadth. The full pipeline is re-run; runtime and memory
characteristics are benchmarked; any awkwardness in scaling (embedding
memory blow-ups, graph query slowness) is fixed here. Findings are
re-evaluated: if F1 holds in 8 of 10 surgical specialties but not in 2
non-surgical ones, that specialty-dependence is itself an interesting
finding. Deliverables: v2 corpus and results, a scaling benchmark report
(runtime/memory/cost vs. corpus size), and updated figures incorporating
v2 data.
