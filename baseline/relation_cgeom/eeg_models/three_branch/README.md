# EEG Three-Branch Model

This package contains a configurable three-branch EEG classifier for downstream finetuning:

- raw temporal branch
- filter-bank Hilbert complex branch
- hyperbolic relation branch

The design is intentionally modular so branches, bands, and fusion operators can be swapped during ablations.
