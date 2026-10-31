# Auxo adapter for FMoW

Status: experimental integration, not yet validated by distributed GPU training.
This is an Auxo clustering baseline adapted to the FIELDING/FedScale runtime,
not a bit-for-bit reproduction of the upstream Auxo scheduler or published results.

## Isolation

The aggregator entry point is fedscale/cloud/auxo/aggregator.py. FIELDING's
aggregator, client manager and existing configs are unchanged by this addition.
The adapter subclasses the runtime and imports its impact gate; it overrides
label-based assignment with the vendored upstream Auxo reward-table policy.
Provenance and Apache license are in vendor/.

Configurations are benchmark/configs/auxo/fmow_{baseline,impact,smoothing,temporal}.yml.
Each uses a unique job name, benchmark/auxo_fmow log directory, and port 7200.
Run jobs sequentially if they share the GPU. Do not reuse FIELDING checkpoints.

## Run from repository root in the Linux FedScale environment

    python -c "import torch, sklearn, numpy; print(torch.cuda.is_available())"
    python test_auxo_adapter.py
    bash fedscale.sh driver start benchmark/configs/auxo/fmow_baseline.yml
    bash fedscale.sh driver start benchmark/configs/auxo/fmow_impact.yml
    bash fedscale.sh driver start benchmark/configs/auxo/fmow_smoothing.yml
    bash fedscale.sh driver start benchmark/configs/auxo/fmow_temporal.yml

Run ONE start command at a time. Launcher output is auxo_fmow_MODE_logging.
Use a separate final test set: the inherited gate currently consumes evaluation
results from the existing test loader.

## What is implemented

Start with one cohort, obtain update differences from participating clients,
retain the last tenth of coordinates (following upstream metadata), center and
normalize updates, and use upstream QTable for subgroup rewards, exploration,
migration and automatic splitting. Dense internal indices support sparse FMoW
client IDs. Index zero remains the upstream sentinel. New children inherit
parent model weights. Invalid updates fail explicitly. Inactive clients are
filtered using streaming availability metadata.

Baseline omits all three flags. Impact uses evaluation accuracy; smoothing
compares adjacent windows; temporal adds the existing label-state novelty filter.
The first cohort split bypasses gates to permit initial discovery. Thereafter
candidate table/membership updates are committed only when the gate permits.
Rejected updates are retained for reconsideration, while reward-table state is
unchanged. This gates the entire candidate update, including reward learning;
it does not save the computation already spent constructing the candidate.
The temporal option does not independently detect feature or concept drift.

## Differences requiring disclosure in comparisons

The FIELDING harness still trains a global reference model, uses its participant
allocation, and schedules cohort adaptation at the cluster-1 round boundary.
Upstream Auxo has a different scheduler. Raw vectors retained from different
participation times may span different model versions; this adapter follows
upstream batch accumulation but needs distributed validation of freshness.
Empty proposed cohorts are deferred. Initial discovery bypass and gating are
extensions, not part of plain upstream Auxo. Auxo's automatic split policy and
its upstream minimum cohort size are retained. Profiles use 100 participants
and 2001 rounds: at 10 participants the upstream size threshold can prevent
any split in FMoW. Match these settings in FIELDING comparisons, include global
reference costs, and report actual splits/migrations, evaluation cost, wall time
and accuracy rather than interpreting gate decisions as saved reclusterings.

## Validation

Four CPU tests exercise actual upstream splitting, complete/disjoint client
membership, rejection without reward-table mutation, zero-divergence rewards,
and exclusion of global-reference feedback. Python compilation passes.
No full distributed FMoW run was executed here: PyTorch is unavailable.
Check AUXO_ADAPT and AUXO_GATE logs in a smoke run before using long benchmarks.
