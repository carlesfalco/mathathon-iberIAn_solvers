# Rust port of `updated/unified/{core,drivers}` — port map, traps, staged estimate

Scope: the 851 physical / 351 code lines of `updated/unified/core` + `updated/unified/drivers`.
Everything below is either **measured** (I ran it; commands and outputs are in this directory) or labelled
**estimate**. A representative core was actually ported to Rust and diffed against the Python driver on
89 golden values — that is the evidence base, not an opinion about Rust.

Files in this directory:

| file | what |
|---|---|
| `port-map.md` | this report |
| `evidence.txt` | captured output of every run quoted below |
| `evidence-inventory.txt` | `tools/inventory.py` output: per-file/per-function LOC and external primitives |
| `tools/golden.py` | generates `golden.txt` from the **Python** tree (deterministic, RNG-free inputs) |
| `README.md` | index of this bundle and the exact commands |
| `dumps/` | per-box decision dumps and the 304×55 ID block behind trap 5 |
| `tools/inventory.py`, `tools/dump_steps.py`, `tools/mini.py`, `tools/dump_block.py`, `tools/bench_py.py` | inspection scripts |
| `rust/` | the prototype: `src/main.rs` 1214 lines (1104 code), `src/cases.rs` 513 lines (golden harness), no crate dependencies |
| `golden.txt` | 89 `key=value` golden numbers the Rust port must reproduce |

Reproduce (from this directory):

```bash
PYTHONPATH=<repo>/updated python3 tools/golden.py > golden.txt   # Python side
(cd .. && cargo build --release --workspace)                    # ~4 s cold, no deps
../target/release/proto check                                    # 72/89 exact, 17 near-eps rank counts
../target/release/proto bench | grep BENCH                       # timings quoted in §4.4
```

---

## 1. The port surface, measured

`tools/inventory.py` (AST-based) over the eight modules:

| file | physical | code | comment | docstring | defs | classes |
|---|---:|---:|---:|---:|---:|---:|
| `core/types.py` | 67 | 26 | 13 | 9 | 2 | 3 |
| `core/tree.py` | 125 | 78 | 15 | 14 | 9 | 2 |
| `core/admissibility.py` | 36 | 8 | 9 | 9 | 3 | 3 |
| `core/compress.py` | 85 | 30 | 20 | 20 | 6 | 2 |
| `core/kernels.py` | 55 | 29 | 6 | 6 | 7 | 0 |
| `core/__init__.py` | 9 | 6 | 1 | 1 | 0 | 0 |
| `drivers/rskel.py` | 212 | 71 | 60 | 56 | 4 | 4 |
| `drivers/butterfly_id.py` | 262 | 103 | 62 | 55 | 10 | 5 |
| **total** | **851** | **351** | **186** | **170** | **41** | **19** |

So the "~850 LOC" in the brief is *physical*; the algorithmic content is 351 lines, 55 % of it in the two
drivers. Comment/docstring density (42 %) is the reason the tree reads well and also the reason a port cannot
be a line-by-line transliteration: half the meaning is in prose that must be re-derived.

External primitives actually called (inventory, plus `np.linalg.norm` reached through an attribute chain):

- NumPy (20): `abs arange array asarray concatenate diag exp flatnonzero ix_ log log2 mean ones outer sin sum union1d vstack zeros` and `linalg.norm`.
- SciPy (5): `linalg.qr(pivoting=True)`, `linalg.solve_triangular`, `linalg.lu_factor`, `linalg.lu_solve`, `sparse.lil_matrix`.
- `scipy.special.hankel1` in `core/kernels.py:9,41` (only used by `helmholtz2d`).

Rust replacements, in order of how much design they cost:

| Python | Rust | cost |
|---|---|---|
| `sla.qr(M, mode="economic", pivoting=True)` | LAPACK `dgeqp3`/`zgeqp3` binding, **or** Businger–Golub by hand (94 code lines measured) | the whole rank-decision question, §4.2/§6 |
| `sla.solve_triangular` | 12 lines of back-substitution, or `dtrsm` | low (but LAPACK also scales near-singular `R`; see trap 6) |
| `sla.lu_factor`/`lu_solve` | `dgetrf`/`dgetrs`, or 66 lines measured | low, one silent trap (§4.3) |
| `sp.lil_matrix` + `tocsr` | `sprs`/`faer`, or dense per-tile blocks + an apply loop (168 lines measured) | storage convention changes (§trap 7) |
| `np.ix_`, fancy indexing, broadcasting | explicit gather/scatter index loops | this is most of the 2× LOC ratio |
| `np.result_type` in `Factorization.solve` | `Scalar` trait + promotion | 37 lines measured (trait + `f64`/`C64` impls) |

Test surface available as a specification: **126 pytest tests** collected from
`tests/test_core.py test_rskel.py test_spec_rskel.py test_butterfly_id.py test_spec_butterfly.py test_spec_problems.py`.
They pin numerical gates (`relres ≤ 100ε`), stats bookkeeping (`eliminated + root == N`), the weak≡strong-with-empty-near
identity (bitwise, `test_rskel.py:78-89`), the Q-set requirement (`without_q > 30 * with_q`,
`test_rskel.py:119-130`), the butterfly `nnz < N²` / `nnz(4N)/nnz(N) < 8` laws and the `t=0` FFT identity.
That set, not the prose, is what a port should be driven by — and it is portable to Rust as data, as
`tools/golden.py` demonstrates.

---

## 2. Module-by-module port map

Legend for "kind": **M** mechanical, **D** needs a design decision, **N** needs a numerics crate/FFI, **X** skip/stub.

### `core/types.py` (M, 26 code lines)
Type aliases are documentation only — the module says so (`types.py:3-6`) and nothing is runtime-checked.
Port as: `struct IdResult<T> { skel: Vec<usize>, red: Vec<usize>, t: Mat<T> }`; `enum Adm { Weak, Strong }`;
a `Compressor` **trait whose method takes `&Mat<T>` explicitly** (trap 1); `Matrix<T>`/`Vector<T>` as
`Mat<T>` + `Vec<T>`. Do *not* port the aliases as newtypes unless you want the compile-time separation —
it costs ~40 lines and buys ordering/semantic safety the Python never had.

### `core/tree.py` (M, 78 code lines → 175 Rust code lines measured)
The one real design decision: `Node.parent` / `Node.children` form a **cyclic object graph** (`tree.py:20-37`).
Rust: an arena `Vec<Node>` with `parent: Option<usize>` and `children: Vec<usize>`, plus
`by_key: HashMap<(usize, Vec<usize>), usize>` mirroring `_by_key` (`tree.py:73`). No `Rc<RefCell<_>>` needed;
the arena also makes `levels: Vec<Vec<usize>>` and `nodes_postorder` trivial. Ordering must be preserved:
Python builds nodes from `sorted(groups)` (`tree.py:80`), so use `BTreeMap<Vec<usize>, Vec<usize>>`.
`neighbors` iterates `np.ndindex(*(3,)*d)` (`tree.py:111`) — a `d`-digit odometer with the last coordinate
fastest; the *order* leaks into `near` and hence into floating-point summation order.
`interaction_list` (`tree.py:119-125`) is dead code for both drivers; port it only if you port the test
(`test_core.py:23-34`) that pins it. The prototype here does port it, and the golden harness compares the
resulting coordinates (`tree2d_ilist_mid`), because it doubles as a 2-D neighbour-logic check.

### `core/admissibility.py` (M, 8 code lines → 19 measured)
`Weak`/`Strong` are one line each. `Complementary.pairs` is a Cartesian product of two levels with an
`assert` on equal depth (`admissibility.py:33-36`). A two-variant enum plus one function.

### `core/compress.py` (D+N, 30 code lines → 72 measured for both compressors, 94 for the QR/ID)
`interp_decomp` (`compress.py:23-35`) is 13 code lines that hide: economic pivoted QR, the truncation rule,
a triangular solve for `T`, and three degenerate shapes (`n==0`, `m==0`, `k==0`). The rule
`k = int(np.sum(d > eps * d[0]))` (`:32`) is a **count**, not a first-crossing search — see trap 5.
`ExactCompressor` is a `vstack` of `A[far,box]` and `A[box,far].T` (`:46`); `ProxyCompressor` adds a proxy
circle (2 points in 1-D, `n_proxy=64` in 2-D, radius `2.5·half_width`) plus the far unknowns *inside* that
radius as explicit rows (`:76-84`). Both build a matrix whose column count is the box's unknown count and
whose row count is `|far|+|box|` or `n_proxy+|inside|`.

### `core/kernels.py` (M+N, 29 code lines → 60 measured)
`laplace2d` (log, zero diagonal), `fio1d` (`exp(2πi(xξ + t·c(x)|ξ|))`, sine speed profile) are mechanical.
`helmholtz2d` needs `H₀⁽¹⁾` (`:9,41`) — **blocked without a Bessel implementation**: either a crate
(`special`, `bessel`) or FFI to AMOS/SpecialFunctions. For core parity only `laplace2d` and `fio1d` are needed;
`helmholtz2d` is used by the showcase/E-problems.

### `drivers/rskel.py` (D, 71+17+3+17 code lines → 220 measured)
Loop `for l in depth..1` over `tree.levels[l]` (`:164`), per box: active box, near list from the admissibility
rule, `far` as a complement mask, the Q set, then skeleton → sparsify → eliminate (`:180-199`), with the
per-box matrices stored in a `_Step` for `solve` (`:78-88, 100-116`). Everything is index bookkeeping over a
dense working matrix; the dense copy is explicit in the docstring (`:28-30`: `O(N²)` memory, "fine for
N ≤ 16k on a 24 GB machine: 2 GB for real float64 at N = 16384"). A port must decide whether to keep that
(the honest choice for a representative core) or to store tiles.

### `drivers/butterfly_id.py` (D, 103+9+7+8 code lines → 168 measured)
`build_trees` → two trees of depth `log2 N`; `Complementary.pairs` per stage; stage 0 = column ID of
`K(all x, B)` (`:171-177`); stages `l→l+1` merge the two parents' skeleton columns as candidates and ID
`K(A_child, candidates)` (`:183-198`); final stage evaluates `K(A, skel)` exactly into `U` (`:201-204`).
Factors are CSR; `apply` is `L+2` sparse mat-vecs (`:125-130`). Storage is reported as `nnz` (`:132-135`).

---

## 3. Traps (each one found while porting; file:line; what I did)

1. **The compressor is mutated behind its protocol.** `drivers/rskel.py:130-131` does
   `comp = copy.copy(compressor); setattr(comp, "A", W)` — so inside the driver the constructor argument `A`
   is dead and the *current Schur-complemented* matrix arrives through an attribute the `Compressor` protocol
   (`core/types.py:58-60`) never declares. A faithful trait must take the matrix as a parameter:
   `fn call(&self, w: &Mat<T>, box: &[usize], far: &[usize], node: &Node, q_extra: &[usize]) -> IdResult<T>`.
   In Rust this is not optional: there is no way to bolt the field on afterwards.
2. **`_skeletonize` type-tests the compressor.** `rskel.py:132-140` checks `isinstance(comp, ProxyCompressor)`
   and then reads `comp.kernel`, `comp.proxy_points`, `comp.radius_factor`, `comp.eps`. So the protocol is
   only half a protocol. Rust: an `enum Comp<T> { Exact{..}, Proxy{..} }` (measured: 72 lines for both) or a
   trait with an `explicit_rows()` hook. Do not start from the protocol and hope.
3. **Update order inside one step is load-bearing.** `rskel.py:187-191`: the *column* update reads
   `W[ix_(loc, skel)]` before it is written, then the *row* update reads `W[ix_(skel, loc)]` **after** the
   column update. Swapping the two lines or fusing them into one symmetric update changes results at O(1),
   not O(eps). My prototype keeps the two statements adjacent and mirrors the order.
4. **"The R–F blocks become O(eps) and are dropped" is only half true.** `rskel.py:13-14` and `:187`. The ID
   guarantees the *far* blocks (`W[far,red] = W[far,skel]·T`) and the transpose relation; after the sparsify the
   **skeleton–redundant** blocks stay O(1). Measured in the prototype: `max|W[S,R]| = 1.38` against
   `max|W| = 5.2` on the P1 leaves. Those entries are simply never read again because `red` is deactivated.
   A port that "zeroes what the comment says" is harmless; a port that *relies* on them being small is wrong.
5. **The truncation rule is a count and it sits exactly at the accuracy floor.** `compress.py:32`:
   `k = #{i : |R_ii| > eps·|R_00|}`. On this project's own blocks `|R_kk|/|R_00|` lands within a factor of a
   few of `eps`, so `k` is decided by the last bits of the QR. Measured on the *same* 304×55 block of the P1
   level-1 ID matrix (`tools/dump_block.py` + `proto idfile`, both in `evidence.txt`): LAPACK gives `k=36`,
   `id_relerr = 2.33e-9`; my Businger–Golub gives `k=35`, `id_relerr = 5.28e-9` — same pivot order, both
   inside `10ε`. Consequence: **skeleton counts, root size and per-stage storage are not portable numbers.**
   Either bind LAPACK to keep them comparable, or gate the port on achieved error (which is what the SPEC
   tests do) and expect ±1–4 skeletons per level.
6. **`solve_triangular` is not just back-substitution.** SciPy/LAPACK replaces near-zero diagonal elements by
   a scaled value rather than dividing by them. With `k` chosen by the count rule the diagonal is nonzero,
   so a plain loop is fine *here*, but a port that also reuses its triangular solve elsewhere inherits a
   silent difference. Nine lines of Rust, one comment in `main.rs`.
7. **`nnz` counts nonzeros, not stored entries.** The butterfly builds each factor block as a dense
   `r × |cand|` array (`butterfly_id.py:96-102`, `:175-177`, `:191-198`); `lil → csr` then **drops exact
   zeros**. At stage 1 the interpolation matrix is a 2×2 identity, so half the assigned entries vanish:
   `G[0].nnz = 128` for 64 blocks of 2×2. Measured: at N=512, Python reports `nnz = 170 843` while the same
   ranks stored densely are 312 235 slots (**1.83×**). The port must pick a convention and say which one it
   reports; the project's storage ratios inherit the nonzero convention.
8. **`Factorization.solve` promotes dtypes.** `rskel.py:104`: `np.result_type(self.dtype, b.dtype)`, so a real
   factorization solves a complex right-hand side. In Rust this is the `Scalar` trait plus either a promotion
   path or two monomorphizations; measured cost of the generic core: 37 lines (trait + `f64`/`C64`) and no
   change to any algorithm body.
9. **`Tree.__init__` runs its depth heuristic before any node exists.** `tree.py:64-70` builds a Python set of
   coordinate tuples per candidate level (`while depth < 30`) — O(n·depth) tuples, the slowest part of tree
   construction in Python. In Rust: the same loop over a `BTreeMap`, still O(n·depth), ~15 lines. Keep the
   `or 1.0` guard for degenerate boxes (`tree.py:59`) and the `1e-12` origin shift (`:60`), or points on
   dyadic boundaries land in the wrong box (`test_core.py:81-83` pins exactly this).
10. **The Q set is only needed for proxy + strong, but it is recorded unconditionally.** `rskel.py:166,178-179,202-204`.
    For `ExactCompressor` it is dead weight; for `ProxyCompressor` with strong admissibility it is the
    difference between a usable and a broken factorization (`test_rskel.py:119-130`: `without_q > 30·with_q`).
    A port can make it a policy flag, but the test pins the default.
11. **`ProxyCompressor` silently moves mass into the exact part.** `compress.py:76-84`: far points inside the
    proxy radius are appended as explicit rows. So a radius change is not a resolution change (this is
    §4.2-item 2 of `results/update-review.md`). A port must reproduce the same `inside` set with the same
    strict `<` and the same sorted union with `q_extra` (`np.union1d`).
12. **`stats` is consumed by substring matching.** `updated/unified/bench/adapters.py:36-71` finds stats by
    looking for keys *containing* `root`, `time`, `skel`, `elim`. A Rust struct with nicer field names breaks
    the adapters and therefore the 126-test gate. Keep the key names (or provide a serialization layer).
13. **`drivers/rskel.py:28-30` is a memory contract.** The dense working copy is `O(N²)`; at N=16384 real
    float64 that is 2 GB. A Rust port that keeps this is not "faster and leaner" by construction; it is the
    same algorithm with a better constant. Say so before benchmarking (`update-review.md` §6 already lists
    larger-N wall-clock studies with dense work arrays as not worth prioritising).

---

## 4. What the prototype measured

`rust/src/main.rs` (1104 code lines, no dependencies) implements: uniform 2^d-ary tree, weak/strong
admissibility, one column ID over a hand-written pivoted Householder QR, exact and proxy compressors,
`rskel_factor` + `solve` + stats, and the 1-D ID butterfly with apply. `rust/src/cases.rs` (513 lines) is the
golden harness. Rust 1.93, `--release`, cold build **≈4 s** with no dependency graph at all.

### 4.1 Line-count ratios (measured)

| port block | Python code lines | Rust code lines | ratio |
|---|---:|---:|---:|
| scalar + dense matrix (NumPy's job) | 0 | 187 | new |
| pivoted QR + ID | 13 (`compress.py:23-35`) | 94 | 7.2× |
| LU factor + solve | 1 (`rskel.py:119-121`) | 66 | new |
| tree | 78 | 175 | 2.2× |
| admissibility | 8 | 19 | 2.4× |
| kernels (3 kernels) | 29 | 60 | 2.1× |
| compressors | 30 | 72 | 2.4× |
| rskel driver + solve | 88 | 220 | 2.5× |
| butterfly driver + apply | 127 | 168 | 1.3× |
| **total** | **351** | **1104** | **3.1×** |

The interesting part: **the algorithmic lines port at 1.3–2.5×**, and the multiplier is *not* Rust's
verbosity but the loss of vectorization (`M[ix_(rows, cols)]` becomes a double loop) and of NumPy's type
promotion. The 3× headline comes from the numerical kernels, which Python gets from LAPACK for free.

### 4.2 Equivalence against the Python drivers (89 golden keys)

`proto check` against `golden.txt` (deterministic, RNG-free, same libm):

- **72 of 89 keys exact**: every tree number (level sizes, occupied counts, neighbour/interaction-list
  coordinates), every ID decision on the rank-7 test matrix (`k`, skeleton indices, `1e-16` reconstruction),
  all 1-D RS statistics (per-level `skel_max/skel_total/eliminated`, root, step count) for weak **and** strong,
  and the butterfly **ranks and storage** at N=64 for t=0 and t=3.
- **17 keys differ, all of one kind**: per-level skeleton counts and root size of the four 2-D P1 RS cases and
  the butterfly storage at N=128. Measured deltas: root `131 vs 135`, `145 vs 150`, `154 vs 158`, `209 vs 211`
  (≤ 3.7 %), butterfly `nnz 20 084 vs 20 214` (0.6 %). Cause: trap 5 — a ±1 skeleton decision at the ε
  boundary propagates to the root.
- **Achieved accuracy agrees** where it matters: P1 RS residuals, Rust vs Python —
  weak/exact `4.57e-10 vs 2.30e-10`, weak/proxy `5.10e-10 vs 2.64e-10`, strong/exact `2.56e-10 vs 2.93e-10`,
  strong/proxy `1.30e-10 vs 1.06e-10` (all ≈ ε = 1e-8 as the SPEC requires); 1-D RS residuals
  `8.3e-16 vs 6.7e-16` and `8.1e-16 vs 6.1e-16`; butterfly apply error `5.97e-16 vs 4.67e-16` (t=0),
  `1.86e-10 vs 1.93e-10` (t=3, N=64), `2.09e-9 vs 4.87e-10` (t=1, N=128). Butterfly storage is exact at
  N=64 (`5504` and `5280` nonzeros) and 0.6 % low at N=128 (`20 084 vs 20 214`).

Read this as: the port is *numerically equivalent to within the tolerance the drivers themselves promise*,
not *bit-comparable*. Any acceptance criterion phrased on counts rather than errors will fail a good port.

### 4.3 The bug that mattered, and the test that catches it

My first LU stored the row exchanges as `piv.swap(k, p)` instead of LAPACK's `ipiv[k] = p`. Every solve with
pivoting was then silently wrong — and **the 1-D RS cases still passed** (their blocks are diagonally
dominant, so no exchanges happen), while the 2-D P1 residuals were O(1). The unit check that catches it in
one line (`proto lutest`, `evidence.txt`): `n=6 → 7.4e-17` after the fix, `1.6e-1` before.
Lesson for the port plan: port the *golden harness* first; the existing tests' residual gates on
`rs1d`-style problems are too weak to catch a broken LU.

### 4.4 Timings (measured, single run, this VM)

| case | Python 3.12 / numpy 2.5.3 / scipy 1.18 | Rust prototype |
|---|---|---|
| P1 strong/exact N=512, factor | 0.013 s | 0.007 s |
| P1 strong/exact N=1024, factor | 0.037 s | 0.027 s |
| P1 strong/exact N=1024, solve | 0.0016 s | 0.0001 s |
| butterfly build N=512 | 0.354 s | 0.067 s |

What these numbers do and do not say:

- They are **not** a speedup claim for the framework. Both implementations keep the same dense `O(N²)`
  working array and the same algorithm. At these sizes the Python drivers are dominated by per-box NumPy
  call overhead, not arithmetic: the P1 N=1024 strong/exact run makes **100 box eliminations** (4 boxes at
  level 1 with 0 eliminated, 12/132 at level 2, 30/215 at level 3, 66/514 at level 4) in **39.1 ms**, i.e.
  **391 µs per box** for a pivoted QR plus an LU of blocks with ~10–30 columns. Rust removes that overhead —
  hence 1.4× on the factor and 5× on the butterfly build, not 30×.
- The Python side runs LAPACK; the Rust side runs my 94-line Householder. A production port should bind
  LAPACK and give up part of the constant-factor win to keep rank decisions comparable.
- The solve ratio (15×) is the real structural win: 100 steps × 2 LU solves per right-hand side, 1.6 ms in
  Python (≈ 8 µs per small solve) against 0.1 ms in Rust.
- Honest estimate, not measured: a LAPACK-backed, allocation-free port with tiled storage would be a
  few times faster than Python on the same problems, and the *dense work array* would still cap the
  asymptotics. Do not promise O(N log N) behaviour from a port.

---

## 5. Staged effort estimate

Anchors: `*` = measured in this session (Rust lines, or wall-clock of the prototype).
Effort days are **estimates** for one engineer fluent in both languages, including tests, excluding review.

| stage | content | Rust code lines | effort (days) | main risk |
|---|---|---:|---:|---|
| S0 | crate layout, `Scalar` trait, `Mat<T>`, error types | 187* | 0.5–1 | none |
| S1 | tree (arena, BTreeMap ordering, odometer neighbours), admissibility | 194* | 1–2 | ordering/precision of `1e-12`/`1e-9` constants |
| S2 | kernels `laplace2d`, `fio1d`; `helmholtz2d` needs Bessel H₀ | 60* | 0.5–1 | **blocked** for Helmholtz without a crate/FFI |
| S3 | ID: bind `dgeqp3`/`zgeqp3` (+`dtrsm`) **or** 94 lines by hand | 30 / 94* | 0.5 (LAPACK) / 2–4 (hand) | rank decisions at ε (trap 5) |
| S4 | LU: `dgetrf`/`dgetrs` or 66 lines | 20 / 66* | 0.2 / 0.5–1 | pivot convention (trap 4.3) |
| S5 | `rskel_factor` + `solve` + stats + compressor enum + Q set | 220* | 2–3 | update order, stats key names |
| S6 | butterfly: stages, sparse-or-dense factors, apply, storage convention | 168* | 2–3 | nnz convention, memory |
| S7 | golden harness + rank policy + CI gate | 513* (harness) | 2–3 | deciding LAPACK-or-tolerance |
| **total, core+drivers** | | **~800–1100*** | **8–15** | |

i.e. **2–3 calendar weeks** for a numerically equivalent core with LAPACK, **3–5 weeks** if the numerics are
hand-rolled and expected to be production quality. My own prototype took one focused session of ~2 h for
1104 lines, including ~30 min lost to the LU pivot bug — that is a *feasibility* data point for a
representative core, not an estimate for a reviewed library.

What is deliberately **not** in that estimate (and should not be promised):

- **E1–E12 + showcase parity.** `explore/` + `showcase/` are **10,676 LOC** and import **19**
  `scipy.special` entry points (`exp1 factorial gamma gammaln h1vp hankel1 j0 jn_zeros jv jvp k0 kv kve
  lpmv roots_legendre spherical_jn y0 yn_zeros zeta`), plus heavy `scipy.linalg` and `scipy.integrate`.
  The integrate use is narrow and worth stating exactly: production code integrates the Bessel ODE with
  `solve_ivp` (DOP853) in E9 (`explore/e9_lippmann_schwinger/ls_problem.py:32,205`), while `quad` appears
  only in a test (`tests/test_spec_problems.py:22`). That is months, and the Bessel/Legendre layer is the
  binding constraint, not the hierarchical machinery.
- **Lean one-file examples per problem.** Once the core exists these are genuinely small (30–80 lines each)
  *iff* the kernel avoids special functions: the DFT, `fio1d` butterfly, Laplace HBS/RS and the Oseen/Oseen-like
  exponential-scaled kernels are straightforward; the E7/E9/E10/E11 kernels are not, without a Bessel crate.
  A realistic deliverable is 4–6 one-file Rust examples on the algebraic problems plus the Python
  experiments kept as the reference for physics.
- **Performance claims.** See §4.4. The dense work array dominates, and `update-review.md` §6 already
  deprioritises large-N wall-clock studies for exactly that reason.

Recommended shape of the deliverable, if this goes ahead: keep this prototype as `rust-unified/` with
`special`-free kernels, LAPACK behind one `id()`/`lu()` module, the golden harness in CI, and a written
statement that parity is *core + drivers*, measured by error, not by skeleton counts.

---

## 6. What I did not do

- I did not port the showcase or E1–E12, and I make no claim about their portability beyond the
  special-function inventory above.
- I did not bind LAPACK, so my rank decisions differ from the Python driver's by ±1 on 2-D blocks (§3 trap 5).
  That is a deliberate decision to expose the issue, not an oversight.
- The prototype does not implement the SVD-reference comparison (`butterfly_id.py:226-251`, which needs the
  missing `source/code/butterfly.py`) or the `bench/adapters.py` duck-typed stats layer. It *does* implement
  `interaction_list`, and the golden harness checks it against the Python coordinates (`tree2d_ilist_mid`).
- Timing numbers are single runs on this VM. The Rust prototype is not linked to BLAS/LAPACK; the Python side
  is (NumPy here is built against OpenBLAS 3.9.0 and LAPACK), which is exactly the asymmetry the timings show.
