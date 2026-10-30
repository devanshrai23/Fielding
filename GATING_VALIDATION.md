# Gating implementation and experiment protocol

FIELDING (attached AISTATS 2026 paper) already combines individual migration
with selective global reclustering. This extension adds a group-level
performance gate with optional two-window smoothing and temporal novelty.

The gate uses assigned-cluster evaluation accuracy, timestamped by training
round. Once clustered models exist, global-model evaluations are excluded.
The existing test loader supplies these observations: experiments must reserve
an independent final test set before claiming unbiased generalization results.
A separate validation split is not created automatically by this change.

With smoothing enabled, compare two adjacent windows of W observations (six
observations for W=3). Raw impact is a counterfactual diagnostic, not an earlier
mandatory gate. Insufficient window history falls back to raw impact; no usable
accuracy history preserves original FIELDING behavior. Threshold .05 means five
percentage points. Group impact is the mean across clients with observations.

Rejected drift remains pending. Fresh observations reconsider it. Repeated calls
with identical measurement histories reuse the decision and do not advance
counters. Temporal history advances only for clients with changed observation
histories. Persistence uses training-round timestamps; delta history remains a
window of observations, not consecutive rounds. D_X and D_C remain unimplemented
zeros, and M duplicates label magnitude. Temporal filtering uses any-client
significance. Ten consecutive fresh harmful group decisions bypass the temporal
filter (max_impact_deferrals). This bound applies to group decisions, not each
individual client's waiting time.

Both FMoW 200-round profiles now enable smoothing. For ablations use identical
configurations and seeds, varying only the gating flags: baseline, raw impact,
smoothed impact, and smoothed impact plus temporal. Run multiple seeds and save
resolved configurations, source revision, per-round accuracy and timing.

Gate approvals are not actual global reclustering events. The label-based path
logs ACTUAL_GLOBAL_RECLUSTER separately. Smoothing counters describe decisions
on the observed trajectory, not causal baseline savings. Report actual global
operations, migrations, communication, elapsed time, evaluation cost, tail-client
accuracy, and recovery delay. Existing 38-to-62 percent claims are unverified;
no new benchmark results are asserted by this patch.

Run: python test_gate_regressions.py
The regression suite extracts the actual gate via AST, avoiding the GPU/runtime
imports. Full distributed integration and dataset benchmarks require the
FedScale/PyTorch environment and datasets.
