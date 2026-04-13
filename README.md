# Mathy

> **Branch status:** This README is preserved for the branch historical experiment state. For the frozen submission artifact, use `release/paper-v1` or tag `paper-v1.0`. Later results supersede parts of the branch-local narrative below: the paper story is the moderate-depth frontier-reranker rescue, the depth-7 failure boundary, and negative learned-frontier/RL frontier-controller diagnostics.


Goal-conditioned HyperKAN for verified algebraic rewriting.

The project is organized around a local-first POC:

- Phase 1: prove the symbolic environment, baselines, and HyperKAN behavior on Strix Halo.
- Phase 2: promote the same pipeline to a single H200/B200 only if the local gate passes.
- Phase 3: consider multi-GPU only if the single-GPU run materially changes the research scope.

## Current scope

The first implementation pass focuses on:

- a verified SymPy rewrite environment with six actions
- backward trajectory generation
- structural tokenization
- three policy heads: MLP, static KAN, HyperKAN
- beam-search inference with exact verification
- minimal visualization utilities for diagnostic plots

## Expected runtime environment

Development happens locally. The code is written so the eventual ROCm 7.2 toolbox run on `golde-r` is an execution concern, not a code fork.

## Layout

- `configs/`: config files for local and promoted runs
- `data_gen/`: symbolic actions, canonicalization, generation, validation
- `tokenizer/`: structural tokenization and vocabulary
- `models/`: encoders and policy heads
- `train/`: losses and train loop helpers
- `search/`: beam search for verified inference
- `eval/`: end-to-end evaluation entrypoints
- `viz/`: diagnostic plotting
- `scripts/`: shell wrappers for local and promoted runs

