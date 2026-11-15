# Eight-minute speaker notes

**Slides:** `slides.pdf`, pages 1–10. Pages 11–13 are backup answers.  
**Reading time:** about 7–7.5 minutes at 120–130 words/minute, leaving 30–60 seconds for pointing and pauses in an eight-minute slot.  
**Goal:** explain the shared block strategy and how it reveals what each algorithm or experiment needs.

## Slide 1 — A common block strategy (0:00–0:30)

> FMM, direct solvers, butterfly transforms and the FFT look unrelated. They all begin, however, by grouping entries in a large interaction matrix and replacing suitable groups by shorter descriptions. Our five design choices organize that process. This is not one magic program: it is one sequence of questions that exposes the mathematics each method still needs.

## Slide 2 — The expensive interaction table (0:30–1:15)

> With $N$ sources and $N$ targets, a direct method stores $N^2$ interactions. But a separated log-kernel interaction has a power series whose terms split into a target-only factor and a source-only factor. A few terms therefore describe many entries. That is low-rank compression. Nearby data stay exact; distant data use the short description. A tree supplies the bookkeeping for choosing groups and reusing work, but it does not prove compressibility.

**Optional:** numerical rank always includes a tolerance and a norm.

## Slide 3 — Five choices (1:15–2:10)

> The five questions are: which groups, which block pairs, which short representation, which samples construct it, and which algebra assembles it? FMM, butterfly and the FFT answer all five, but differently. The choices are coupled. Reordering changes the blocks; changing a block can change its best sampler; elimination adds pivot questions. Every method therefore needs three separate certificates: approximation error, numerical stability and total cost—including construction and the exact near field.

## Slide 4 — FMM, wideband FMM, solvers, butterfly, FFT (2:10–3:10)

> FMM contributes analytic expansions and translations. Wideband FMM changes those representations with scale and uses stable diagonal high-frequency translations. The direct solver selects skeleton interactions, sparsifies redundant ones and eliminates them through Schur complements; modified blocks and pivots are essential. Butterfly uses two trees, complementary tiles and nested bases. The FFT combines exact rank-one residue-class tiles with Cooley–Tukey recursion. These are shared questions, not interchangeable presets.

Call recursive skeletonization the **direct solver** after its first mention.

## Slide 5 — Butterfly: two ideas (3:10–4:00)

> Butterfly shows why compression alone is insufficient. Peeling away row-only and column-only phases leaves a mixed phase. Complementary tiles pair narrower position boxes with wider frequency boxes to control that remainder, so local rank stays small. But independently factoring every middle-level tile still costs order $N^{3/2}$. Reusing old summaries through small transfer matrices gives order-$N$ work per logarithmic stage and order-$N\log N$ overall. In short: **tiles give low rank; nesting gives speed**.

## Slide 6 — The two trees, visually (4:00–4:35)

> Here are the two trees explicitly. Position intervals split as depth grows; their frequency partners sit at complementary depth and merge. The highlighted two-position box therefore meets a four-frequency box. All four displayed tiles contain eight entries, with different aspect ratios. That fixed entry area controls the leftover mixed phase. The second tree chooses the blocks; nesting then connects their bases across stages.

Point once to $A$ and $B$; do not read every tile label.

## Slide 7 — FFT and the direct solver (4:35–5:25)

> For the DFT, residue classes produce an integer term in the phase, whose exponential is one. The tile is exactly rank one—not merely low rank at a tolerance. Both orderings must match, and Cooley–Tukey recursion is still the algorithmic step. A direct solve uses a different grammar: redundant interactions are reconstructed from skeletons, eliminated locally and passed upward through Schur complements. Accurate compression is necessary, but pivot stability and modified-block bookkeeping must also be checked.

## Slide 8 — Experiments by problem (5:25–6:25)

> The experiments deliberately push individual assumptions. A proxy can be structurally insufficient even with more points on the same circle. Tiny screened or evanescent blocks need absolute or global tolerances. Complex elimination needs an adjoint congruence. Stokes needs block sampling; scattering needs contrast on the unknown side; Matérn needs related kernel orders. Oseen shows the final caution: a threefold smaller root can cost more storage and elimination work. The framework's value is to show exactly which new mathematics enters.

Choose one example closest to the audience rather than reading all eight rows.

## Slide 9 — Python, Rust and measured results (6:25–7:20)

> Python supplies the broader experimental package: trees, ID, recursive solves, butterfly, ten showcase models and all twelve experiments. Rust supplies short real-kernel examples, exact DFT factors and a separate complex/recursive prototype. At $N=2048$, the saved Rust measurements use 3.6–7.7 percent of dense storage and apply 11.8–14.7 times faster in the same process. Construction takes about 0.96 seconds and remains a dense-to-compressed oracle, so this is not a language race or a full production-solver claim.

## Slide 10 — Takeaway (7:20–7:50)

> Blocks expose structure; trees choose geometry; representations shorten descriptions; sampling constructs them; assembly makes an algorithm. Butterfly is one worked example, not the whole framework. Individual papers and experiments still supply the decisive identities, geometries, translations and stability analysis. The sentence to remember is: **one block strategy, different mathematics for each problem**.

## Backup answers

### Is this one program that derives all the algorithms?

No. It is a common set of design choices and analysis. Recursive solve versus apply, translations, pivot stability and specialized sampling cannot be deduced from a universal menu. The useful claim is transferable structure and experiment-driven understanding.

### Why can low rank not tell whether a method is fast?

Because storage, construction, data movement, translation and solve work can dominate. A smaller summary can cost more to build, and a well-conditioned apply does not imply stable elimination.

### Is Rust now equivalent to Python?

No. Python has far more special functions, quadrature and physics. Rust shows compact examples and representative core algorithms. Complex FIO and recursive solve are present in the companion prototype, but a unified production API and the complete physics experiment suite are still future work.

### Where is the proof behind the slide numbers?

The report has the log-tail bound, mixed-phase peeling, merge count, Schur elimination and exact DFT residue-class identity. Rust and Python reproduce all rank-one DFT stages and $22{,}528$ stored entries at $N=1024$. Timing records and core golden fixtures are in `rust/results/`. Experiment-specific details are in each `python/unified/explore/e*/findings.md` file and in `results/update-review.md`.

### What should I say if someone asks about pitfalls?

Be brief but honest: “some operators are ill-conditioned, some sampling choices are only valid for particular kernels or dimensions, and a smaller root can come with larger build/storage cost. The audit documents those cases. Our framework says compatibility must be measured, not assumed.”

## One-minute vocabulary rehearsal

- **Block:** a rectangular group of matrix entries.
- **Low rank:** many entries generated by a small number of building patterns.
- **Skeleton:** selected interactions retained to represent redundant interactions.
- **Schur complement:** the leftover system after a local elimination.
- **Complementary tiles:** finer boxes in one tree paired with coarser boxes in the other.
- **Proxy sampling:** auxiliary observations used instead of evaluating every distant interaction.
- **Apply:** evaluate $Ax$. **Solve:** find $x$ for $Ax=b$.
- **$\eps$:** requested tolerance. A rank without its tolerance has no unique meaning.
