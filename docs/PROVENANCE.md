# Provenance of this repository

## Where the code comes from

This repository began on 19 November 2026 with a single commit. Its content is that of
`tatendatobaiwa/algorithmic-reconstruction-framework` at commit `ae54e5b` (branch `release/2.0-cloud-checks`: `main`
with milestone 6b of release 2 and its check on the real runs). Only `CITATION.cff` (the repository address),
`results/release2_checks/README.md` (two commit references replaced by the `arf` version) and this file differ.

The earlier repository's history held, among others, the release of record 1.5.3 for Paper 1 (6 November 2026),
release 1.5.6 with protocol amendments A14 to A19 (16 November 2026) and release 2 up to version 2.0.0.dev9
(16 to 19 November 2026). A git bundle of that history was made at the move. Branch names and pull-request numbers in
`results/release2_checks/README.md` refer to that history. How AI tools were used is described in `AI_ASSISTANCE.md`.

## What verifies the code

* **The registered pipelines**: `protocol/arf_confirmatory_protocol.json` records the SHA-256 of release 1.5.3
  (`pipeline_sha256`) and of release 1.5.6 (amendments A18 and A19). `tests/test_repository.py` checks both, and that
  the amendments are in order.
* **Release 2**: `tests/parity/` executes the registered releases' cells unchanged and requires release 2 to give the
  same results; `results/release2_checks/` records the same comparison on the real runs.

## `SHA256SUMS.txt`

The file list of release 1.5.3 as published on 6 November 2026, kept unchanged as that release's record. Every entry
under `pipeline/`, `analyses/` and `manuscript_tools/` still matches. Six entries do not, because the files changed
after release 1.5.3:

| File | Why it differs |
|------|----------------|
| `.gitignore` | Packaging entries added for release 2 (`*.egg-info/`, `build/`, `dist/`, `.ruff_cache/`) |
| `AI_ASSISTANCE.md` | A draft note removed (7 November 2026) |
| `README.md` | Updated for release 1.5.6, amendments A14 to A19 and release 2 |
| `protocol/arf_confirmatory_protocol.json` | Amendments A14 to A19 appended; amendments A1 to A13 are unchanged |
| `tests/test_repository.py` | Updated for amendments A14 to A19 (the order check now runs to A19) and tests of release 1.5.6 added |
| `CITATION.cff` | The repository address of this repository |

To check: `sha256sum -c SHA256SUMS.txt`.
