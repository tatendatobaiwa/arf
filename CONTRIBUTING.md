# Contributing

Thank you for looking at this code. Two rules matter more than any style point.

## 1. The registered files are never edited

`pipeline/`, `analyses/`, `protocol/` and `manuscript_tools/` hold the research releases, the amendment scripts and the
protocol. Their SHA-256 hashes are recorded in `protocol/arf_confirmatory_protocol.json`, and `tests/test_repository.py`
checks them. Do not reformat, rename or "tidy" anything there, however small. If you find an error in one of those
files, open an issue: the remedy is a new release and a protocol amendment, as release 1.5.6 and amendment A18 did for
the Alpha-5 catalogue numbers.

## 2. Release 2 changes structure, not results

New code goes in `src/arf/`. A module is accepted only with a test showing it reproduces the registered release
exactly (`docs/ARCHITECTURE.md`). A change that alters a result is a scientific change and needs its own discussion.

## Working on release 2

```
pip install -e ".[dev]"
ruff check . && ruff format --check .
python -m pytest -q tests
```

The access tests (`tests/parity/test_access_*`) start a Java machine for Orekit. They need Orekit's data files to
match a real run's conditions: point `ARF_OREKIT_DATA` at a folder holding `tai-utc.dat`, `itrf-versions.conf` and
`Earth-Orientation-Parameters/IAU-2000/` from https://gitlab.orekit.org/orekit/orekit-data, as continuous integration
does. Without it they run with the leap seconds only, which is still a strict comparison with the registered code.
pytest's faulthandler plugin is switched off in `pyproject.toml`: it collides with the Java machine's own signal
handling and turns a clean exit into a segmentation fault.

* Google-style docstrings and type hints on everything public.
* Names carry their unit (`_min`, `_km`, `_percent`); no single-letter names outside short mathematical expressions.
* Tests in `tests/unit/` are small enough to verify by hand; tests in `tests/parity/` compare with the registered code.
* No element sets, no protected-area boundaries and no cell-level run files in a commit (see `DATA.md`).

## Reporting a problem

Open an issue with the release, the command, and the smallest input that shows it. For a suspected error in a
published number, name the table or figure and the file under `results/` it comes from.
