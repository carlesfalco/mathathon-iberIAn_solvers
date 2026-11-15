# Rust port: scope, difficulty, and staged plan

## Short answer

**A small demonstration is easy enough to be worthwhile. Complete feature parity is not a weekend translation.** The Python core and two drivers are about 850 lines, but that line count excludes the numerical backend, sparse data structures, and most physics. A useful Rust core is a focused multi-day to multi-week job. A production-quality port of the ten-model showcase and E1–E12 is a multi-week to multi-month program, depending on how much special-function and solver infrastructure can be reused.

The working prototype already demonstrates the easy end of this spectrum: three 11–12-line real-kernel examples, QR compression, storage/apply measurements, and direct FFT correctness and an exact DFT-butterfly factorization. It does not yet demonstrate package-wide parity.

## Current implementation boundary

| Python component | What it includes | Current Rust status |
|---|---|---|
| `core/types.py` | Matrix/vector conventions, kernel/compressor/admissibility protocols | Small closure-based kernel API only |
| `core/tree.py` | Uniform $2^d$-tree, coordinates, parents/children, neighbours, interaction lists | Not ported; current compressor uses index-range recursion |
| `core/admissibility.py` | Weak, strong and complementary node interactions | Not ported as a public API |
| `core/compress.py` | Pivoted-QR column ID, exact/proxy compression, interpolation matrix $T$ | Related but different: block $Q_rR_rP^{-1}$ compression; no skeleton $T$ or proxy rule |
| `core/kernels.py` | Real Laplace, complex Helmholtz and FIO phase | Three real one-dimensional kernels only |
| `drivers/butterfly_id.py` | Two trees, staged IDs, sparse factors, complex apply | Root crate: direct RustFFT plus exact DFT-butterfly; generic FIO exists in the companion reference, not in the common API |
| `drivers/rskel.py` | Bottom-up sparsification, LU, Schur updates, modified $Q$ set, forward/back solve | Not ported |
| Showcase | Ten physics models, quadrature, special functions and analytic checks | Not ported |
| E1–E12 | Twelve research harnesses and many alternative compressors | Not ported |

Calling the current code `unified-rs 0.1` is therefore appropriate; calling it “the Python unified package in Rust” would be premature.

**Companion prototype:** `core-prototype/` additionally implements the arena tree, weak/strong admissibility, exact/proxy compressors, complex FIO butterfly and recursive LU solve, with 72 exact cross-language golden keys and 17 documented tolerance-boundary rank/count differences. The status table above describes the cleaner root crate alone. The companion establishes feasibility of those pieces; it does not supply the production backend, a unified API, or special-function physics.

## Why the solver is more than a mechanical translation

### 1. Ownership and index identity

Python nodes contain parent/child object references and mutable index arrays. A natural Rust representation is an arena such as `Vec<Node>` with `NodeId` parent/children IDs. Keeping stable IDs is important: the recursive solver continually maps global unknowns to active, near, far, skeleton, and redundant sets. This is design work, not syntax substitution.

The Python solver also shallow-copies a compressor and assigns its `.A` field dynamically (`drivers/rskel.py::_skeletonize`). Rust should make that context explicit in a trait or a `CompressionContext` containing the current working matrix and the modified-$Q$ rows, rather than emulating monkey-patching.

### 2. Real and complex arithmetic are not interchangeable

A production port needs either a carefully chosen scalar trait or separate real/complex implementations. More importantly, `T.T` and `T.conj().T` cannot share one convenient “transpose” label. The E5 results require a deliberate representation of ordinary versus conjugate transpose and separate pivot-stability tests.

The current prototype intentionally does not expose a complex solver API. Adding one before that convention is fixed would risk looking complete while implementing the wrong algebra.

### 3. Numerical backends carry mathematical responsibility

Python delegates pivoted QR, triangular solves, LU, SVD and special functions to SciPy. Rust needs equivalents with tested accuracy, permutation conventions and underflow behavior. `nalgebra` makes this prototype small, but a production port should choose and pin its backend explicitly—pure Rust factorizations, `faer`, or a BLAS/LAPACK binding—and validate it before comparing timings.

Required tests include exact rank-deficient matrices, tiny/subnormal blocks, empty far fields, rank-zero blocks, nontrivial column permutations, and real and complex triangular solves. The prototype includes zero, diagonal, separable rank-one, smooth log-kernel, FFT and Parseval tests; that is only the start.

### 4. Construction is different from solving and applying

The Python recursive solver keeps a dense working matrix for clarity. A Rust line-for-line port can do the same, and this is useful for correctness, but it does not produce an asymptotically sparse direct solver. Reaching production performance requires a separate data-structure phase.

All benchmarking must therefore separate:

1. dense/operator construction;
2. compression and factorization;
3. stored representation size;
4. apply time;
5. forward/backward solve time and residual;
6. wall-clock, allocation, and memory behavior on controlled inputs.

## Special functions are the larger dependency

The showcase imports or uses Hankel functions and derivatives, $K_\nu$ and scaled $K_\nu$, $J_\nu$, $Y_0$, $E_1$, Gamma/log-Gamma, Riemann zeta, associated Legendre functions, spherical $j_n$, Gauss–Legendre nodes, ODE integration, and parameter-dependent quadrature. Having a crate function with the right name is not enough: the Python experiments depend on scaling, singularity treatment, accuracy near the origin, overflow/underflow regimes and analytic continuation where relevant.

The sensible order is to port **formulation and reference checks together**. A kernel implementation without its analytic/dense reference is not evidence of parity.

## Recommended stages and effort

These are engineering estimates, not measurements or delivery promises. They assume one developer already comfortable with Rust and numerical linear algebra, using the existing Python code and saved results as the oracle.

| Stage | Deliverable and acceptance gate | Rough effort |
|---|---|---|
| A. Demonstration slice | Real block compression, 3 kernel files, direct FFT, honest apply/storage benchmarks | **Implemented here** |
| B. Faithful core hardening (a reference implementation now exists) | Arena tree; weak/strong/complementary rules; real/complex matrix boundary; pivoted column ID with $T$; exact/proxy compressors; cross-language rank and error fixtures | **1–2 focused weeks** |
| C. Generic butterfly hardening (a reference implementation now exists) | Two virtual trees, sparse factors, complex phase, DFT exact-count test plus B1/FIO error/rank/storage parity at multiple tolerances | **About 1 additional week** if the exact DFT reference is reused |
| D. Recursive-solver hardening (a representative implementation now exists) | Working-matrix RS, LU/pivots, Schur updates, modified $Q$ set, real/complex row transformations, root solve, forward/backward error and pivot-health parity | **1–3 additional weeks** |
| E. First useful showcase | Laplace, DFT, FIO, Gaussian/screened examples, short per-kernel files and the same accuracy reports | **Several days** once B–D are stable |
| F. Full showcase physics | Stokes, Helmholtz, Yukawa/Rothe, fractional, Lippmann–Schwinger, spherical harmonics, including special functions and references | **Several additional weeks**, with SHT and singular quadrature likely dominating |
| G. E6–E12 research parity | All experiment variants, controls, plots and regression evidence | **Several more weeks**; a multi-month program if production cost and all controls are included |

A pragmatic first milestone is hardening **B plus C**, not “all physics”: the reference implementation exists, but rank policy, backend choice, ownership boundaries and CI need review. Stage D makes “direct solver” an accurate production description rather than only a prototype result.

## Benchmark protocol for credible Rust-versus-Python claims

Before any timing comparison:

1. Use the same input points, kernel values, tree, admissibility, tolerance and permutations.
2. Check dense block entries, selected ranks, factor nonzeros, and apply/solve error against saved Python fixtures.
3. Use equivalent numerical backends and thread counts; do not compare optimized SciPy/BLAS with unreleased Rust loops or vice versa.
4. Report construction, factorization, storage, apply, solve, and end-to-end time separately.
5. Use warm-ups, raw samples, min/median values and a size sweep, with identical deterministic inputs. Record peak RSS alongside stored entries. Keep process startup out of apply benchmarks and include it only in end-to-end benchmarks.
6. State whether dense working matrices remain. Do not describe dense-to-compressed construction as $\mathrm O(N\log N)$.
7. Compare specialized with specialized implementations. Direct RustFFT versus the generic butterfly builder is a useful case study, not a fair Rust-versus-Python language race.

Current Rust results deliberately avoid a language-speedup claim: dense and compressed apply run in one Rust process, and direct FFT is compared with a simple naive DFT solely for correctness and scaling.

## Design for short one-file examples

The long-term user-facing API should let each problem file provide only:

1. point geometry;
2. a kernel or block-kernel function;
3. physics-specific sampling/quadrature choices;
4. a small set of checks.

The library should own traversal, compression, sparse factors, solve sweeps, timing, and result serialization. The current three kernels already demonstrate that interface. A useful next cleanup is to move the remaining timing code out of `examples/fft.rs` so specialized transforms are equally concise.

“One file per problem” should never mean one giant file of numerical infrastructure. The showcase's automatic distinction between dial code and physics setup is worth preserving in Rust: concise glue is compatible with substantial, carefully tested physics underneath.

## Practical recommendation

Adopt a two-track plan:

- **Research track:** harden the reference core and generic butterfly against backend and tolerance parity, then productionize the recursive solver using Python fixtures as the oracle.
- **Communication track:** use the current 10-minute deck, the three small kernel examples and the exact DFT demonstration. They communicate the idea now without implying that the full port is done.

The most expensive mistake would be to port every kernel before fixing the tree/ID/solver interface and parity harness. That creates many short example files with no trustworthy common library underneath.
