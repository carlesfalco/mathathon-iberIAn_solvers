# Soundness and research-interest review

## Verdict

**The common structure is sound as an organizing abstraction. The stronger claim that one fixed program derives the papers by turning four dials is not sound.** The most useful version of the project is a carefully scoped explanatory and experimental framework, not a new universal solver.

The research interest is real but currently uneven:

- **Strong as pedagogy:** one vocabulary, explicit small examples, and honest comparisons can make a difficult literature accessible.
- **Useful as an experimental harness:** swapping trees, interaction patterns, and compression methods can reveal failure modes.
- **Not established as a new framework:** closely related classifications and composable implementations already exist.
- **Potentially new as numerical analysis:** the high-wavenumber skeletonization instability and its remedy deserve further investigation, but require independent reproduction, a precise theorem, and a much stronger novelty search.

The replacement beginner's guide is [framework-explained.md](framework-explained.md). Its numerical examples are in [examples.py](audit-sources/examples.py), with output in [example-output.txt](audit-sources/example-output.txt).

## Status after the E6–E12 update

A second audit covers the seven new experiments and ten-model showcase in [update-review.md](update-review.md). The central verdict above still stands: the update strengthens the compatibility map and the analysis of when it fails, not the universal-solver claim.

The main changes to the research assessment are:

- **E5:** the transpose-versus-adjoint pivot mechanism is identified and independently checked. Its practical severity is kernel- and regime-dependent; E9 includes an exact/realified error ratio of $135\times$ at strong admissibility, $k=80$, $\varepsilon=10^{-6}$, so a universal small loss factor should not be quoted. A theorem with useful hypotheses and a stronger prior-art search remain open.
- **E6–E10:** these provide breadth and several strong controls, but also separate dense agreement from physical accuracy, per-box rank from total storage, and per-order SHT compression from all-order transform cost.
- **E8/E11:** proxy surfaces and multi-order lifted rows are promising constructive sampling experiments. Their exact local-lift proofs are limited to integer-codimension cases; fractional cases remain numerical evidence.
- **E12:** directional admissibility gives a robust approximately threefold root reduction at high $\lambda$, but no measured total-cost win. At $N=4096$, skeletons fall $21.6\%$ while storage and elimination flops rise $28.6\%$ and $67.3\%$.

Several original textual corrections were not carried into `updated/`: the standard FMM interaction count, the FMM-LU cost table, the general butterfly rank-proof hypotheses, and the separation of caustics from phase regularity and inverse conditioning. New headline corrections—including butterfly storage, test-count wording, SHT Newton polishing, and E12 cost—are listed in the update review. The sections below retain the original `source/`-based audit for traceability; use the update review for current E6–E12 claims.

## What is solid, conditional, or unsupported

| Claim | Assessment | Reason |
|---|---|---|
| Hierarchical grouping plus compressed interactions is shared by these algorithms | **Sound, with qualifications** | This is the hierarchical-matrix/H² point of view. Different methods use different trees, pairings, bases, and traversals. |
| The four dials are an independent menu from which every paper follows | **Unsupported** | The choices are coupled, and each paper needs additional operators and analysis. |
| A single formula $C_\varepsilon+c(kw\theta)^{d-1}$ predicts rank across all regimes | **Heuristic only** | Rank depends on source/target geometry and norm; ambient and intrinsic dimensions differ; accuracy dependence changes by regime. |
| Low numerical rank explains fast application | **Partly true** | Construction, nesting, translation cost, and the number of levels are also essential. |
| Low numerical rank certifies a fast, accurate direct solve | **False in general** | Elimination also needs suitable fill-in, pivot conditioning, and control of propagated errors. |
| Proxy compression works because a matrix is smooth | **False** | It needs an appropriate PDE/Green's representation or another proved sampling argument, plus correct treatment of exceptions and modified entries. |
| The FFT is an exact rank-one butterfly in bit-reversed order | **True in the stated pair of orderings** | The residue-class proof is exact. The Cooley–Tukey recursion is still needed to obtain the FFT algorithm. |
| The broad framework is a new research contribution | **Not supported by the literature checked** | Existing taxonomies, a generic four-component matvec framework, and algorithm-selection work are close precedents. |
| The high-$k$ exact-ID failure is worth studying | **Mechanism supported; scope and novelty open** | E5 identifies the transpose/adjoint congruence mechanism. E9 shows that the loss is regime-dependent and can reach $135\times$ in a strong-admissibility coarse-tolerance case; neither a general severity bound nor novelty is established. |
| The ten-model showcase establishes breadth and speed | **Breadth supported; speed not established** | All ten quick agreement checks pass, but five of ten recorded fast paths were slower than dense at the tested sizes and dense work arrays remain. |
| Lifted/multi-order proxies are proved for the full fractional sweeps | **Measured more broadly than proved** | E8 and E11 supply useful constructions and controls; the exact integer-dimensional lift arguments do not cover the full fractional cases. |
| SHT turning-point tiles necessarily lose compression | **Contradicted by E10's measurements** | Straddling tiles had lower mean rank and rank tracked nonseparable mixed-phase variation. Transform-level build/storage remains expensive. |
| Directional admissibility gives an Oseen total-cost win | **Not supported at the tested depths** | Root size falls roughly threefold, but the independently reproduced $N=4096$ case costs $28.6\%$ more storage and $67.3\%$ more elimination flops. |

---

## What should be kept

The documents also contain strong material and several of the necessary self-corrections: the deck's speaker notes explicitly reject “literally one loop”; the wideband level-cost model is mostly careful; the README says dense bookkeeping invalidates its timing slopes; E1 acknowledges the nested-format comparison bias; E4 states that butterfly insertion into the solver was not done; and E5 presents a concrete stability mechanism and fixes. The measured weak/strong rank contrast, the exact DFT rank-one proof, and the small shared ID implementation are good foundations. The editorial task is to make these qualifications consistent with the headlines, not to discard the project.

## Highest-priority corrections

### 1. Narrow the scope before presenting the framework

**Where:** the opening claims in `source/md/proposal.md`, `source/md/unified_map.md`, and `source/unified/README.md`.

The documents sometimes move from “matrices from physics/PDEs” to dense integral operators without saying that the class has changed. The recommendations do not automatically apply to sparse finite-element or finite-difference matrices. Make the main scope:

> Dense integral-operator matrices, and dense blocks such as Schur complements for which a separate rank argument is available.

Also distinguish applying $A$ from solving $Ax=b$. FMM and butterfly are primarily fast operator application technologies; HBS/RS-S/FMM-LU are factorization technologies. “Algebra = apply/factor” is a useful label, but those branches require different correctness statements.

### 2. Replace “one program, four dials” with “four design choices, three certificates”

**Where:** `source/md/proposal.md:5`; `source/unified/README.md:34`; `source/md/unified_map.md` opening.

The README says that the variants “differ only in the admissibility rule,” although butterfly requires two trees and a different assembly, and the solver requires elimination. Replace that sentence and avoid “recovered by our code unchanged.” Code written to share primitives demonstrates **abstraction and reuse**, not mathematical derivation of every algorithm.

A sound claim is:

> The algorithms can be compared using partition, interaction pattern, representation, and assembly. For each instantiation we separately certify block approximation, numerical stability, and total cost.

The wideband translation, FMM-LU proxy/Q-set and quadrature, butterfly transfer construction, and FFT radix recursion must appear as explicit additional steps, not hidden dial values.

### 3. Demote the rank formula to a model with stated coordinates

**Where:** `source/md/proposal.md:45–53`; `source/md/sota.md:111,122`; `source/md/deck.md:94,500,512`.

The formula

$$
r_\varepsilon\approx C_\varepsilon+c(kw\theta)^{d-1}
$$

is not a theorem covering all the listed admissibility rules. Specific issues:

- $d$ must not switch between ambient dimension and the dimension of a surface or source cluster;
- the numerical rank also depends on target/source shapes, separation, aspect ratio, kernel, and error norm;
- the notes vary between $C_\varepsilon$, $C\log(1/\varepsilon)$, and $C\log^{d-1}(1/\varepsilon)$;
- Engquist–Zhao provides important separability bounds in specified settings, not this universal interpolation formula;
- if $\theta\approx w/R$, the guard $kw^2/R\lesssim1$ says $kw\theta\lesssim1$, so the growing term is $O(1)$ precisely where the formula is declared valid; the following high-rank regimes need a separate argument rather than extrapolation of the same sentence;
- the defensible explicit derivation is much narrower: transverse geometry, $kw\gg1$, $\theta\ll1$, and $kw\theta^2\ll1$ reduce the block to the mixed phase $e^{-i(k/R)u\cdot v}$. Collinear sources have a different rank at the same coordinates, so orientation and intrinsic dimension are indispensable;
- the project's 3D cube measurements do not exhibit its claimed asymptotic exponent. Independent replication gives ranks $28,29,32,39,50,77,105,134,196,259$ at $kD=0.5,1,2,4,8,16,24,32,48,64$ and a log–log exponent of $0.46$. This is useful pre-asymptotic/decision-chart evidence, not confirmation of $(kD)^2$; no alternative universal onset threshold should be invented without a derivation;
- “the count chooses the algorithm” omits translation and construction costs and has direct prior art in Yang's NUFFT-versus-butterfly selector.

Use the formula only after defining its geometry and error measure, and label it **a fitted/heuristic degrees-of-freedom model**. Present the phase-peeling calculation for specific tiles first; then introduce phase space as intuition.

### 4. Correct the FMM interaction-list count and low-frequency explanation

**Where:** `source/md/wideband_fmm.md:69,167,200`.

The same-level interaction list is bounded by $6^d-3^d$ ($27$ in 2D, $189$ in 3D), not $7^d-3^d$ ($40$ and $316$). The supplied `Tree.interaction_list` uses the children of the parent's touching neighbours, consistent with the standard list.

The statement $p\approx kD$ “plus a few terms” hides the accuracy-dominated low-frequency regime. Cheng et al.'s expansion orders can be large even when $kD$ is small. This is not a minor wording issue: the low-frequency breakdown of diagonal forms occurs because the required expansion order can greatly exceed the frequency-size parameter, causing large intermediate quantities and cancellation. It is not explained by rank becoming large.

The good news is that the reported transition sizes $0.25/3.5/12$ wavelengths for $3/6/9$ digits agree with Table 1 of the primary paper.

There is also a regime-classification error in `source/md/proposal.md:35` and `source/md/sota.md:104`: the wideband method is described as effectively directional **above** the cutoff. In the paper, the high-frequency signature method uses the ordinary interaction list and the whole sphere of directions. The six-way directional face split belongs to the **low-frequency** exponential-translation machinery. Above the cutoff, ranks still grow with box size; the improvement comes from diagonal translation, not $O(1)$ directional rank.

### 5. Treat pivot stability as a first-class part of the solver

**Where:** `source/md/fmm_lu.md` elimination discussion; `source/unified/drivers/rskel.py`; `source/unified/tests/test_spec_rskel.py`; E3/E5 findings.

The shipped compressor stacks the two interaction directions using a transpose and the driver applies the row operation with `T.T`. For complex $T$, this need not be a congruence. With $\Pi=[I;-T]$,

$$
A=I,\quad T=i
\quad\Longrightarrow\quad
\Pi^{\mathsf T}A\Pi=0,
\qquad
\Pi^*A\Pi=2.
$$

Thus accurate far-field compression does not imply safe elimination pivots. This is not merely a hypothetical example. Independent run on the attached code, P3 with $N=512$, $k=80$, weak admissibility:

| Compressor | Relative forward error | Relative residual |
|---|---:|---:|
| Exact ID | $2.351\times10^{-5}$ | $2.369\times10^{-5}$ |
| Proxy ID | $8.461\times10^{-11}$ | $8.450\times10^{-11}$ |

Here $\kappa(A)\approx1.052$, so conditioning of the original matrix does not explain the failure. The supplied tests stop the exact complex Helmholtz cases at $k=20$, below the reported failure regime. An independent reimplementation at the same $N$ gave, at $k=100$, forward errors $4.09\times10^{-4}$ with $T^{\mathsf T}$, $1.76\times10^{-11}$ with the compatible $T^*$ stack and row operation, and $6.26\times10^{-11}$ with the proxy. This supports the proposed repair on this test problem, but does not establish that the published proxy-based FMM-LU pipeline fails or that the observation is new.

**Recommended action:** add regression tests at $k=80$ and above; state precisely when transpose, conjugate transpose, or separate row/column IDs are intended; verify any replacement against both matrix approximation error and pivot conditioning; do not claim novelty until the literature search is much stronger. The documents also need synchronising: `deck.md` and `unified_map.md` still describe E5 as investigating/unexplained, whereas `explore/e5_highk_instability/findings.md` already gives a mechanism and fixes; E3's earlier non-radiating-subspace conjecture is therefore stale as well.

Two further proxy details matter. The paper's proxy sphere has radius $5R/2$ for box side $R$ and a whole-box $Q/P$ split; the code uses radius $2.5\times(R/2)=1.25R$ and a point-wise inside test, so it is a working variant, not a test of the paper's guarantee. The Q-set stress result is real: at $\varepsilon=10^{-10}$ I reproduced residuals $1.09\times10^{-12}$ with tracking and $1.10\times10^{-9}$ without, a factor of $1017$. But $1.10\times10^{-9}<100\varepsilon=10^{-8}$, so the docstring's word “breaks” is too strong here; the Q set buys accuracy rather than always determining pass/fail.

### 6. Replace the FMM-LU cost table rather than patching one number

**Where:** `source/md/fmm_lu.md:131–147`, especially lines 143 and 145.

Several quantities in the derivation are inconsistent:

- the weak-surface row writes $r_\ell\sim n_\ell$ but uses $(n_\ell)^{1/2}$ in its displayed sum;
- active-set size and retained skeleton size are used interchangeably in the per-box cubic work;
- the fixed-points-per-wavelength sum has order $\Theta(N^3)$, not $N^3/16$. For example, including $\ell=0$ gives

$$
\sum_{\ell\ge0}4^\ell(N4^{-\ell})^3
=N^3\sum_{\ell\ge0}16^{-\ell}
\sim\frac{16}{15}N^3;
$$

starting at $\ell=1$ instead gives $\sim N^3/15$. The level indexing must be stated, but either convention contradicts the displayed estimate;
- that row also conflicts with the cited paper's own regimes: §4.3 states a typical $O(n^2)$ factorization cost at fixed points per wavelength, while §6.2 reports a measured root size $n_0\sim n^{4/9}$ and hence a root-dominated $O(n^{4/3})$ estimate. Neither supports the table's $N^3$ row.

Derive cost from explicit counts of active variables, eliminated variables, neighbours, and root size. Cite the paper's own regime analyses and their different statuses rather than trying to infer everything from the heuristic $(kw)^2$ alone. A root-size measurement by itself does not determine the whole level-by-level rank profile.

### 7. Limit the butterfly rank proof to what its assumptions prove

**Where:** `source/md/butterfly.md:83–103`; `source/md/butterfly_steps.md`.

The exact phase-peeling identity and the bound

$$
|R|\le\frac{M}{4}w_Aw_B
$$

are the right explanation. The later statement that a general phase needs only $q=O(\log(1/\varepsilon))$ Chebyshev terms is too quick as written: the approximation argument must quantify analyticity/smoothness, amplitude of the remainder, singularities such as $\xi=0$, and the constant's dependence on $M$. Bounded mixed derivative alone is not a complete proof of all claimed FIO cases. In particular, Candès–Demanet–Ying's Theorem 3.1 assumes joint real-analytic estimates on all mixed derivatives and obtains $r_\varepsilon\lesssim\log^4(1/\varepsilon)$; it does not prove the notes' $O(\log(1/\varepsilon))$ conclusion from the single derivative bound.

Likewise, “one tile is one Heisenberg cell” is intuition, not a proof that the tile has rank one. Give the complete exponential-series proof for $\Phi=x\xi$, state the limited result it proves, then cite the separate theorem needed for a general FIO.

The construction-cost explanation in `source/md/butterfly.md:143` has the correct final exponent but intermediate steps that need rewriting. The correct middle-level count is $O(N)$ tiles of size $O(\sqrt N)\times O(\sqrt N)$; the delivered text loses the required square-root formatting and counts the tiles incorrectly. A rank-$r$ randomized sampling approximation of one tile costs $O(r\sqrt N)$, giving $O(rN^{3/2})$ overall.

The apply-storage derivation also needs more care than the competing $O(r^2N)$ and $O(rN)$ formulas in §4 and §6. At one merge, a tile with new rank $r_2$ reads two old rank vectors of sizes $r_0$ and $r_1$, so its dense interpolation block has $r_2(r_0+r_1)$ entries. A stage has $N$ tiles, giving the bound $O(Nr_2(r_0+r_1))$ before zeros and identities are removed. Summing $r_2(r_0+r_1)$ over the merges gives total storage that is $O(N\log N)$ when all ranks are bounded; it is not universally $2rN$. Direct inspection of the shipped factors shows exactly these merge dimensions and substantial stage-dependent sparsity.

### 8. Do not identify caustics with non-smooth phase or with a general butterfly failure

**Where:** `source/md/butterfly.md:102`; `source/md/walkthrough.md:234`.

A caustic is a folding/degeneracy of a ray map; it does not automatically mean that the kernel phase $\Phi(x,\xi)$ is non-smooth. The supplied parametrix experiment becomes ill-conditioned near a caustic, but this does not prove that every complementary tile loses low rank there. Separate three claims:

1. phase regularity needed for tile compression;
2. complementary low rank of the matrix;
3. conditioning of the operator and quality of an approximate inverse.

They can fail independently.

### 9. Make experimental comparisons like-for-like

**Where:** E1–E4 and corresponding claims in `source/md/unified_map.md` and `source/md/deck.md`.

The E1 report itself identifies an important bias: flat weak/strong storage is compared with a nested butterfly. The Gaussian-kernel advantage can therefore come from nesting, not complementary admissibility. There is also a measurement-terminology issue: `interp_decomp` chooses skeletons from pivoted-QR diagonal ratios and returns no per-box spectral-error estimate. Its skeleton count is not automatically the SVD $\varepsilon$-rank defined in the prose, even when the two agree numerically on these examples. Report “QR skeleton count at this pivot tolerance” unless a separate error certificate is supplied. More generally:

- compare nested weak/strong formats with nested butterfly formats;
- tune every method to the same **achieved** global error, not only the same local $\varepsilon$;
- report build, storage, apply, and solve separately;
- use direct dense references and multiple right-hand sides;
- distinguish fitted asymptotics from measured finite-$N$ trends;
- do not compare one format's nonzeros with another's flop count without a validated cost model.

E4's conclusion is currently a **far-block representation result**, not a completed “butterfly inside FMM-LU” result. The report explicitly says the insertion into `rskel` was not done. The summary storage number is also stale: `source/md/unified_map.md` says “0.18× dense at $N=4096$, $\varepsilon=10^{-6}$.” The shipped benchmark at the tighter $\varepsilon=10^{-8}$ gives $2{,}392{,}863/4096^2=0.143$, while an independent rerun at $\varepsilon=10^{-6}$ gives ratios $0.116$ ($t=1$) and $0.112$ ($t=3$). None of these supports $0.18$.

### 10. Repair reproducibility before using test counts as evidence

The attachments contain `source/unified/` and `source/md/`, but not the explainer `code/` directory, `code/butterfly.py`, the cited `research/` directory, or the `.venv` named in the README. Consequently:

- tests requiring the SVD reference cannot run as shipped;
- several figures and the wideband/butterfly browser explainers cannot be regenerated from the attachments;
- `source/unified/drivers/butterfly_id.py::load_svd_reference` points to a nonexistent file;
- benchmark comparisons against that reference are not reproducible from the archives;
- `deck.md` says 151 tests while the collected suite has 126, and several summaries still say E5 is unfinished although its findings are present.

This does not imply that the original repository failed its tests. It means the supplied artifact does not substantiate all of the README's claims on its own. Package the missing dependencies or mark those results as external and unreproduced.

### 11. Reframe novelty using explicit prior art

**Where:** `source/md/proposal.md` candidate contributions; `source/md/sota.md:111–132`.

The checked literature contains:

- Martinsson–O'Neil's weak/strong by general/nested-basis classification;
- Li–Liu's extension to oscillatory hierarchical formats and factorizations;
- Agullo–Bramas–Coulaud–Gicquel's four modular components, generic matvec, and `composyx` implementation;
- Börm–Börst–Melenk's interpretation of butterfly as a special directional H² structure;
- Yang's explicit NUFFT-versus-butterfly selection algorithm.

Therefore avoid “first unified view,” “no source uses it to choose an algorithm,” and “new as an artifact” without much narrower qualifications. A measured rank map in particular coordinates may still be useful, but absence of prior art has not been established.

One citation nuance: Zheng–Riccietti–Gribonval v9 contains **both** Example 3.3 (the DFT's butterfly recursion) and Proposition 3.19 (complementary low rank for a particular pair of trees). Cite the one matching the exact claim rather than replacing one with the other wholesale.

### 12. Simplify terminology and numbering

**Where:** `source/md/walkthrough.md` “simple setup”; the repeated “HBS / FMM-LU / strong RS” passages.

The setup shows four dials but labels five items, with compression/sampling/representation numbered inconsistently. This makes an already difficult argument harder to audit. Use exactly four top-level headings and put sampling **inside** representation, or explicitly introduce five components throughout.

Do not treat HBS, RS-S, and the full FMM-LU solver as synonyms. Say which is a storage format, which is a factorization scheme, and which is the complete boundary-integral solver with adaptivity and quadrature.

The Markdown conversion also strips important mathematical notation: several files contain no `√` character and flatten superscripts and math delimiters. Thus formulas appear as `N3/2`, `N4−ℓ`, or `d−1`, and some middle-level square roots disappear entirely. This is a presentation defect, but for a non-expert it makes correct derivations look false and hides actual errors. The replacement documents use explicit LaTeX for every equation.

---

## Verification performed for this review

### Direct checks

- Extracted both archives while excluding macOS metadata and checking for unsafe paths.
- Read the proposal, unified map, walkthrough, three main paper explainers, code specification/README, experiment findings, core compression, both drivers, and relevant tests.
- Ran the standalone examples: separable-log error bound, rank-one DFT tiles, one radix-2 Cooley–Tukey recombination, and the $T=i$ pivot counterexample. All assertions passed.
- Reproduced DFT rank-one bit-reversal at $N=256$: ranks were $[1,2,4,7,8,7,4,2,1]$ in natural order and all $1$ with both sides bit-reversed; nonzeros were $52{,}010$ versus $4{,}608$ in this environment.
- Reproduced the high-$k$ exact/proxy discrepancy reported above at $N=512$.
- Reopened primary-source pages for RS-S, FMM-LU, butterfly factorization, Yang's selector, Li–Liu, and Agullo et al.; inspected the wideband paper and Zheng et al. v9 locally.

### Unit tests on the supplied archive

Command:

```bash
PYTHONPATH=source python3 -m pytest -q source/unified/tests
```

The completed run gave **119 passed, 7 failed in 316.59 seconds**. All seven failures are `FileNotFoundError`/`ModuleNotFoundError` for the absent SVD reference; the other 119 tests passed. The README's “126 tests” should therefore not be quoted as a result reproduced from this archive.

### Not established

- No claim here is an exhaustive proof that a measured map, stability mechanism, or exposition is unprecedented.
- No wall-clock claim from the original laptop was independently validated.
- No high-order singular quadrature or full adaptive 3D FMM-LU implementation was supplied or tested.
- The dense working matrix in `rskel` means its timings do not validate the asymptotic cost of a genuinely sparse recursive solver.

## Recommended project focus

A coherent, defensible project would contain three pieces:

1. **The guide:** derive compression from the log example, elimination from a small Schur complement, butterfly rank from phase peeling, and the FFT from residue classes plus radix recursion.
2. **The harness:** one explicit schema and a claim ledger, with every implementation labelled as exact/approximate, full/simplified, and reproduce/not reproduced.
3. **One research question, not several stretch goals:** either the complex-pivot stability mechanism or a validated cost-and-error selection rule. Keep Monarch/BTT, neural operators, and further analogies out of the core until that question is answered.

That focus preserves what is interesting in the current work while removing the claims most likely to fail under expert review.
