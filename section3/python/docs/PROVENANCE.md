# Provenance and cleanup ledger

## Origin

This tree was copied from the supplied `updated/unified` directory, not rewritten from the earlier `source/` snapshot. Supplied files in both directories are retained unchanged as the reference/audit input. SHA-256 hashes for `core/*.py` and `drivers/*.py` at the time of the copy are in `core-provenance.json`.

The package directory inside this submission is `python-unified/unified/`, matching the normal `import unified` convention when this directory is installed or placed on `PYTHONPATH`.

## Deliberate cleanup

| Change | Reason |
|---|---|
| Added a standard `pyproject.toml` distribution, optional pytest extra and console entry point. | Reproducible install instead of a required, unbundled `.venv` path. |
| Updated README, testing, status and results guides. | Current working commands and truthful accessibility; avoid repeating old or unverified headline claims. |
| Removed Python cache directories only in the new tree. | Source distribution hygiene. |
| Preserved experiment outputs, references and tests. | The evidence is inspectable without a long regeneration. |
| Kept experiment and showcase metadata distinct from the small shared core. | A small interface does not mean that physics and special functions are trivial. |
| Kept original README as `LEGACY_README.md`, with `SPEC.md` and `EXPLORE.md` under `docs/`. | Preserve design history without presenting stale commands as current instructions. |
| Corrected the E10 subnormal test. | The previous test's failure assertion was platform-dependent. The guard itself is now tested. |
| Restored the SVD comparison helper as `unified.reference.butterfly` and updated only its loaders/importers. | The supplied `code/butterfly.py` was absent; all seven reference-comparison tests now pass locally. |

| Corrected the E10 Gauss-node Newton derivative and added a regression for its sign. | The audited old sign moved iterates away from roots. Orthogonality is checked directly. |

## External paths removed from the run guide

The older README referred to `.venv/`, `research/`, and `code/butterfly.py`. Only the last of these is required by the reference-comparison tests. No external `research/` content is needed to import the core code or to inspect the saved showcase results. Original literature references remain in the findings documents and in the independent audit.

## No hidden algorithm rewrite

The shared `core/` and `drivers/` Python files match the supplied implementation byte-for-byte at the copy point. There is no attempt to turn the dense reference direct solver into a genuinely sparse solver merely by editing documentation. Large numerical costs, experiment caveats and tolerance conventions remain explicit.

## Submissions and licenses

The new package manifest intentionally adds no new third-party code outside the optional reference helper. Rust's existing manifest and the original authors' licensing should be reviewed by the project owner before external redistribution; this cleanup does not independently establish the intended license for upstream material.
