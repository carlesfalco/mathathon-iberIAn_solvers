# Fast methods for PDE matrices: a common language, without the overclaim

This is a self-contained replacement for the conceptual parts of the supplied notes. It does not assume familiarity with fast multipole methods, hierarchical matrices, or butterflies. The original files are unchanged.

## 0. The short answer

There **is** a useful common pattern:

1. partition sources and targets hierarchically;
2. choose interactions that have a compact description;
3. build that description using an expansion, interpolation, selected matrix entries, or random samples;
4. use the resulting representation either to apply an operator or to eliminate unknowns.

But this is a **map of a family of algorithms**, not one universal program and not a universal solver for matrices from physics. Each paper adds essential mathematics that the four choices do not supply: translation formulas, stability results, elimination order, quadrature, pivot control, or a special algebraic identity.

A reliable development therefore needs **four design choices plus three certificates**:

| Design choice | Question | Separate certificate still needed |
|---|---|---|
| Partition | How are rows and columns grouped? | Does the geometry make the intended blocks compressible? |
| Interaction pattern | Which blocks are stored densely, compressed, or paired complementarily? | What error and rank assumptions justify that choice? |
| Representation | What replaces a dense block, and how was it computed? | Approximation error; construction and translation cost |
| Assembly | How are the pieces applied or eliminated? | Total cost; for a solve, numerical stability |

The updated implementation splits **sampling** out of representation and therefore calls these **five dials**. That is a useful bookkeeping refinement, not an additional correctness certificate: approximation, stability, and total cost are still required either way.

The rest of this guide derives the pattern from explicit calculations and then locates each paper on it.

---

## 1. First, what problem are we solving?

### Applying is not solving

Given a matrix $A$, two different tasks appear in the notes:

- **Apply:** compute $y=Ax$.
- **Solve:** given $b$, compute $x=A^{-1}b$.

An FMM and a standard butterfly factorization primarily provide a fast **apply**. Such an apply can be used inside an iterative solver such as GMRES. HBS, recursive skeletonization, and FMM-LU construct an approximate **factorization**, so they address the second task more directly.

A representation can make $Ax$ cheap and accurate while still being a poor or unstable way to compute $A^{-1}b$. Solving introduces pivots and elimination errors that do not occur in a single matrix–vector product.

### Which PDE matrices are in scope?

Boundary-integral discretizations often produce entries of the form

$$
A_{ij}=w_jG(x_i,x_j),\qquad i\ne j,
$$

possibly plus diagonal or near-field quadrature corrections. Here $G$ is a Green's function, $x_i$ and $x_j$ are target and source points, and $w_j$ is a quadrature weight. Every source can interact with every target, so $A$ is dense.

That is the main setting for this guide. Finite-difference and finite-element discretizations often instead produce sparse matrices. Multigrid, sparse direct methods, and other solvers may then be more appropriate. “Comes from a PDE” is not enough information to select an algorithm.

---

## 2. Compression means replacing one kernel by a short sum

Take a target set $X$ and a source set $Y$. The interaction block is

$$
A_{X,Y}=(G(x,y))_{x\in X,\;y\in Y}.
$$

Suppose we can approximate

$$
G(x,y)\approx \sum_{q=1}^{r}u_q(x)v_q(y)\qquad(x\in X,\;y\in Y).
$$

Then

$$
A_{X,Y}\approx UV^{\mathsf T},
$$

where $U_{xq}=u_q(x)$ and $V_{yq}=v_q(y)$. If $|X|=m$, $|Y|=n$, and $r\ll\min(m,n)$, applying this block costs roughly $O(r(m+n))$ work rather than $O(mn)$. The representation stores $O(r(m+n))$ numbers, not merely $r$ numbers.

### 2.1 Numerical rank, precisely

For a matrix block $B$, its **numerical rank at tolerance $\varepsilon$**, using the relative spectral norm, is the smallest $r$ for which some rank-$r$ matrix $B_r$ satisfies

$$
\|B-B_r\|_2\leq\varepsilon\|B\|_2.
$$

By the Eckart–Young theorem, this is the number of singular values satisfying $\sigma_q>\varepsilon\sigma_1$. Other norms and other error measures give different numerical ranks. The norm and whether $\varepsilon$ is relative or absolute must therefore be stated.

Rank is a property of a **particular block**, at a **particular tolerance**, not a property of a PDE alone. Geometry, source and target sets, discretization, and normalization all matter.

### 2.2 A complete example: no uncertainty principle needed

Consider the one-dimensional real kernel $G(x,y)=\log(x-y)$, with $x>0$ and $|y|<x$. Think of $y$ as lying in a small source cluster around $0$, and $x$ as a distant target. The Taylor series of $\log(1-z)$ gives

$$
\begin{aligned}
\log(x-y)
&=\log x+\log\left(1-\frac{y}{x}\right)\\
&=\log x-\sum_{m=1}^{\infty}\frac{1}{m}\left(\frac{y}{x}\right)^m.
\end{aligned}
$$

The equality holds because $|y/x|<1$. Keeping $m=0,1,\ldots,p$ gives $p+1$ separable terms: one term depending only on $x$, and $p$ products of $x^{-m}$ with $y^m$.

If $|y|\le a$ and $\rho=a/x<1$, the discarded terms obey

$$
\begin{aligned}
\left|\sum_{m=p+1}^{\infty}\frac{1}{m}(y/x)^m\right|
&\le \sum_{m=p+1}^{\infty}\frac{\rho^m}{m}\\
&\le \frac{\rho^{p+1}}{(p+1)(1-\rho)}.
\end{aligned}
$$

This is already the core mechanism of an analytic FMM:

- **far enough** means $\rho<1$, preferably well below $1$;
- **compressible** means the tail is small after a few terms;
- **the approximation certificate** is the displayed tail bound.

For $x=2$, $|y|\le0.25$, and $p=3$, four terms give a maximum measured error of $6.79\times10^{-5}$ over a test grid; the proved bound is $6.98\times10^{-5}$. These numbers are reproduced by [examples.py](audit-sources/examples.py).

The same idea works for many non-oscillatory kernels, but the expansion and its hypotheses change. No single distance rule proves compressibility for every PDE kernel.

### 2.3 Why a tree appears

One fixed source box is not far from every target. A hierarchy fixes this:

- small boxes handle interactions at short range;
- progressively larger boxes handle interactions that are far relative to their size;
- nearby interactions that fail the separation condition remain dense.

A tree is an accounting device that ensures all interactions are covered without doing all of them at the most expensive scale. It does not itself create low rank.

### 2.4 Weak, strong, and complementary are not interchangeable predicates

In the terminology used in the supplied documents:

- **weak admissibility:** distinct boxes may be compressed even if they touch;
- **strong admissibility:** only sufficiently separated boxes are compressed; touching boxes remain exact;
- **complementary pairing:** a source box and target box are paired according to their depths in **two trees**, not accepted or rejected by one near/far predicate.

These names are conventions, so their definitions should always be repeated. Complementary pairing answers a different kind of question from weak/strong separation.

---

## 3. Rank is useful, but it is not the whole theory

### 3.1 What physics can predict

For wave kernels, phase variation limits how well a low-rank approximation can perform. Two useful quantities are:

- box size in wavelengths, often written $kw$ or $kD$;
- the range of directions from which one cluster is seen, often summarized by an opening angle $\theta$.

Phase-space or degrees-of-freedom estimates are valuable **models** for how numerical rank grows. The proposed expression

$$
r_\varepsilon\approx C_\varepsilon+c(kw\theta)^{d-1}
$$

should be treated as such a model, not as a general theorem. Its hypotheses and even the meaning of $d$ must be specified: ambient dimension and the dimension of a surface or source set are not interchangeable. The accuracy term also differs between regimes; it cannot be assumed to be one universal $C\log(1/\varepsilon)$.

A precise version exists, but only with additional geometry. Put a transverse source cluster at $u$ and a target at $R\hat e+v$, where $u,v$ lie in the plane perpendicular to $\hat e$ and both have magnitude $O(w)$. Then

$$
k|R\hat e+v-u|
=kR+\frac{k}{2R}(|u|^2+|v|^2)-\frac{k}{R}u\cdot v
+O\!\left(\frac{k w^3}{R^2}\right).
$$

The first three terms are constant or depend on only one variable, so they do not increase rank. The remaining mixed phase is $e^{-i(k/R)u\cdot v}$. With $\theta\approx w/R$, its dimensionless frequency parameter is $\nu=k w^2/R\approx kw\theta$. The displayed remainder is small when $kw\theta^2\ll1$, alongside $kw\gg1$ and $\theta\ll1$. This **Fresnel/narrow-cone regime** is where the phase-space count has a concrete derivation. The approximate rank is governed by the mixed kernel's degrees of freedom,

$$
r_\varepsilon\approx d_\varepsilon\!\left(\frac{(kw\theta)^{d-1}}{(2\pi)^{d-1}}\right)
+\text{a smooth-kernel accuracy floor},
$$

where $d_\varepsilon$ is the Slepian-type dimension of a phase-space area, not merely an arbitrary fitted constant. A collinear source removes the transverse mixed term and can therefore be much simpler at exactly the same $k,w,\theta$: the formula is not a function of those coordinates alone.

Engquist and Zhao prove important upper and lower separability bounds for Helmholtz Green's functions in specified geometries. They do not prove this single formula for every weak, strong, directional, and complementary block.

There is also a regime-logic problem in the proposal. If $\theta\approx w/R$ is the angular size of the target as seen from the source, then $kw\theta\approx kw^2/R$. Its validity guard $kw^2/R\lesssim1$ is therefore exactly the regime where the frequency-dependent term is $O(1)$. The same formula cannot simply be reused outside that guard to justify every high-rank regime in the following bullets. The bounded and growing regimes need separate hypotheses.

### 3.2 What rank does not predict

Even an exactly known rank leaves four questions unanswered:

1. **Construction:** do we read the whole block, evaluate a formula, use proxy points, or perform matrix–vector products?
2. **Translation:** what does it cost to move or merge a summary?
3. **Nesting:** can a parent reuse its children's bases?
4. **Stability:** if we eliminate unknowns, are the pivots well conditioned?

There is also a difference between the rank of one block and the size of a **shared** representation. If one box uses the same outgoing expansion for many interaction partners, that expansion must describe the union of their requirements; it can be longer than the numerical rank of any one partner block. A rank map measured block by block therefore does not automatically predict the size of an FMM's nested, shared summaries.

The wideband FMM is a particularly clear warning: one can change the cost of a translation while leaving the summary length essentially unchanged. Therefore “physics determines the rank” cannot by itself explain the paper's improvement in asymptotic cost.

A useful cost summary is

$$
C\approx C_{\rm near}+C_{\rm build}
+\sum_{\ell}n_\ell c_\ell C_{\rm transfer}(r_\ell)
+C_{\rm root},
$$

where level $\ell$ has $n_\ell$ boxes, about $c_\ell$ compressed interactions per box, summaries of size $r_\ell$, and a transfer cost $C_{\rm transfer}$. The tree, interaction rule, and representation enter different factors. Low $r_\ell$ helps, but does not determine the whole sum.

---

## 4. Paper 1: from the common pattern to FMM and wideband FMM

### 4.1 Classical FMM

Start with the task

$$
u_i=\sum_{j\ne i}G(x_i,x_j)q_j.
$$

The FMM instantiates the common pattern as follows.

1. **Partition:** build a spatial tree of source and target boxes.
2. **Interaction pattern:** compute touching/near interactions directly; use sufficiently separated interactions at the coarsest valid level.
3. **Representation:** approximate a separated interaction with an analytic expansion or another separable representation.
4. **Assembly:** merge summaries upward, translate them between interaction lists, pass incoming information downward, and evaluate at leaves.

The five usual operation names describe exactly this assembly:

- **S2M:** sources to a leaf's outgoing summary;
- **M2M:** child summaries to a parent summary;
- **M2L:** an outgoing summary to another box's incoming summary;
- **L2L:** parent incoming summaries to children;
- **L2T:** an incoming summary to target values.

The common skeleton does not derive these operators. A kernel layer must supply them and prove their approximation errors. Reusing translations on a regular grid is another essential source of speed.

The standard same-level FMM interaction list has at most $6^d-3^d$ candidates on a full grid: $27$ in 2D and $189$ in 3D. The larger $7^d-3^d$ count refers to a larger buffer convention, not the standard list.

### 4.2 What the wideband paper adds

At high frequency, a large source box can require a long radiation summary. Classical translation of that summary can be expensive. Plane-wave or far-field-signature representations make translation diagonal: each sampled direction is multiplied independently. This changes the **cost per translation**, not merely the numerical rank.

At small scales, the diagonal plane-wave representation suffers numerical cancellation. Cheng et al. therefore combine different representations and translations at different tree levels, with a precision-dependent transition. Their Table 1 reports transition box sizes of $0.25$, $3.5$, and $12$ wavelengths for requested accuracies $10^{-3}$, $10^{-6}$, and $10^{-9}$.

So the route is not simply

> strong admissibility + analytic compression + apply = wideband FMM.

It is more informative to write

> FMM assembly + scale-appropriate expansion families + diagonal high-frequency translations + stable conversions and an accuracy-dependent switch = wideband FMM.

The stability analysis and translation formulas are central content of the paper, not configuration details.

---

## 5. Paper 2: from compression to recursive skeletonization and FMM-LU

### 5.1 Interpolative decomposition: redundancy with a concrete meaning

Let $B$ be a box's unknowns, and let $M$ be a matrix whose columns describe their effects on selected observations. An interpolative decomposition (ID) partitions $B$ into skeleton indices $S$ and redundant indices $R$ such that

$$
M_{:,R}\approx M_{:,S}T.
$$

Every redundant column is approximately a linear combination of skeleton columns. Unlike an arbitrary SVD basis, the retained columns are selected from the original matrix.

For a far set $F$, this can be used to approximate

$$
A_{F,R}\approx A_{F,S}T.
$$

A corresponding row relation must also be constructed. After subtracting $T$ times skeleton columns from redundant columns, the new far block is

$$
A_{F,R}-A_{F,S}T\approx0.
$$

The redundant unknowns consequently have almost no direct coupling to $F$. A compatible row transformation removes the reverse coupling as well. Near interactions are retained.

### 5.2 Elimination is a second mathematical step

Once the redundant variables couple only to a limited set of remaining variables, they can be eliminated locally. Algebraically, for a block system

$$
\begin{bmatrix}
P&Q\\R&S
\end{bmatrix}
\begin{bmatrix}a\\b\end{bmatrix}
=
\begin{bmatrix}f\\g\end{bmatrix},
$$

the first equation gives $a=P^{-1}(f-Qb)$. Substitution into the second gives

$$
(S-RP^{-1}Q)b=g-RP^{-1}f.
$$

The matrix $S-RP^{-1}Q$ is the **Schur complement**. Recursive skeletonization repeats compression and elimination bottom-up, passing skeletons to parent boxes and finishing with a dense root solve.

This is not just applying the compressed operator. It changes matrix blocks, creates fill-in, and requires pivots $P$ to be invertible and numerically safe. Proxy surfaces reduce the cost of computing IDs only when an appropriate Green's representation and sampling theorem applies. Singular quadrature corrections and previously modified Schur-complement entries must be handled explicitly; FMM-LU's bookkeeping is designed for this.

A concise but faithful map is therefore

> recursive compression and elimination + strong/weak interaction rules + compatible row/column IDs + stable pivots = skeletonization solver;

and

> strong recursive skeletonization + adaptive level-restricted octree + proxy-surface analysis + quadrature and modified-block bookkeeping = FMM-LU.

“FMM run backwards” is only a mnemonic. RS-S is more accurately described as a multiplicative or factorization-based analogue using related strong-admissibility structure.

### 5.3 Low rank alone cannot certify a solve

Suppose $\widetilde A\approx A$ and each solves its own system. To first order, the solution error satisfies

$$
\frac{\|\delta x\|}{\|x\|}
\lesssim \kappa(A)\frac{\|\widetilde A-A\|}{\|A\|},
\qquad \kappa(A)=\|A\|\,\|A^{-1}\|.
$$

Thus a small matrix approximation error can be amplified by a large condition number. A small residual is also not automatically a small forward error.

There is a further issue specific to complex elimination. Let $\Pi=[I;-T]$ collect redundant and skeleton variables. The transpose-based pivot involves $\Pi^{\mathsf T}A\Pi$, whereas an adjoint-based transformation gives $\Pi^*A\Pi$. These are not the same when $T$ is complex. For the scalar example $A=I$ and $T=i$,

$$
\Pi^{\mathsf T}\Pi=0,\qquad \Pi^*\Pi=2.
$$

So even an identity matrix can produce a singular pivot under the first transformation. This does not say the transpose convention is algebraically invalid in general; it says block approximation accuracy is not enough to ensure pivot stability. The supplied high-wavenumber results make this an important unresolved implementation and research issue.

---

## 6. Paper 3: from complementary tiles to a butterfly factorization

### 6.1 Complementary means a sequence of tilings

A radix-2 butterfly uses a position tree and a frequency tree of depth $L$. At stage $\ell$, it pairs a position node at depth $\ell$ with a frequency node at depth $L-\ell$. As position blocks become narrower, frequency blocks become wider.

On the normalized grids used in the notes, each resulting tile has constant position–frequency area. Under appropriate phase regularity, these tiles can be low rank even when large ordinary off-diagonal blocks are not.

### 6.2 The exact step behind the heuristic

Let $K(x,\xi)=e^{2\pi i\Phi(x,\xi)}$ and choose tile centres $x_0,\xi_0$. Define the remainder

$$
R(x,\xi)=\Phi(x,\xi)-\Phi(x,\xi_0)-\Phi(x_0,\xi)+\Phi(x_0,\xi_0).
$$

Then, exactly,

$$
e^{2\pi i\Phi(x,\xi)}
=e^{2\pi i\Phi(x,\xi_0)}\,
e^{2\pi iR(x,\xi)}\,
e^{2\pi i(\Phi(x_0,\xi)-\Phi(x_0,\xi_0))}.
$$

The first and last factors depend on only one variable, so they are diagonal row and column scalings and do not change singular values. Only $e^{2\pi iR}$ needs compression.

If the tile has position width $w_A$, frequency width $w_B$, and $|\partial_x\partial_\xi\Phi|\le M$ there, the mixed-difference identity gives

$$
|R(x,\xi)|\le \frac{M}{4}w_Aw_B.
$$

This is the precise role of an area bound. For the bilinear phase $\Phi=x\xi$, $M=1$ and $R=(x-x_0)(\xi-\xi_0)$. The exponential series then gives

$$
e^{2\pi iR}=\sum_{q=0}^{\infty}\frac{(2\pi i)^q}{q!}(x-x_0)^q(\xi-\xi_0)^q,
$$

which is a sum of rank-one terms. Truncating it yields an explicit error bound through the exponential Taylor remainder.

For a general phase, additional regularity and approximation arguments are required. “One Heisenberg cell has rank one” is not the theorem; the useful statement is that bounded leftover phase can produce bounded numerical rank under stated assumptions.

### 6.3 Low-rank tiles still have to be assembled

The butterfly algorithm next constructs compatible bases across the tilings and collects transfers into $O(\log N)$ sparse factors. A merge with two old ranks $r_0,r_1$ and new rank $r_2$ stores at most $r_2(r_0+r_1)$ interpolation entries for that tile; zeros and identities can reduce this further. With bounded ranks and $N$ tiles per stage, each stage therefore has $O(N)$ entries, and the whole factor chain has $O(N\log N)$ storage and apply cost. Construction from entries, matrix–vector products, or a known phase has different costs.

Therefore two trees and complementary pairing identify the useful block structure; they do not by themselves supply the recursive bases, transfer factors, or an efficient construction.

---

## 7. Paper 4: what the FFT does—and does not—inherit from butterflies

Let $N=2^L$ and use the DFT convention

$$
F_{jk}=e^{2\pi ijk/N}.
$$

Choose a row residue class and a complementary column residue class:

$$
j=a+2^\ell j',\qquad k=b+2^{L-\ell}k'.
$$

Then

$$
\frac{jk}{N}
=\frac{ab}{N}
+\frac{ak'}{2^\ell}
+\frac{bj'}{2^{L-\ell}}
+j'k'.
$$

Since $j'k'$ is an integer, its exponential is $1$. Hence every tile factors exactly as

$$
F(A,B)=
\left(e^{2\pi iab/N}\right)
\underbrace{\left(e^{2\pi ibj'/2^{L-\ell}}\right)}_{\text{column vector}}
\underbrace{\left(e^{2\pi iak'/2^\ell}\right)^{\mathsf T}}_{\text{row vector}}.
$$

So the tile rank is **exactly one**. Bit reversal turns these residue classes into contiguous tile index sets. Reversing only one tree does not give the same statement. [examples.py](audit-sources/examples.py) checks all four middle-level $2\times2$ tiles of the $4\times4$ DFT numerically.

But rank one alone does not derive the FFT algorithm. Cooley–Tukey additionally splits even and odd inputs. With $\omega_N=e^{2\pi i/N}$,

$$
(F_Ng)_j
=\underbrace{\sum_{m=0}^{N/2-1}\omega_{N/2}^{jm}g_{2m}}_{\text{half-size DFT}}
+\omega_N^j
\underbrace{\sum_{m=0}^{N/2-1}\omega_{N/2}^{jm}g_{2m+1}}_{\text{half-size DFT}}.
$$

The identities

$$
(F_{N/2}g_{\rm even})_{j+N/2}=(F_{N/2}g_{\rm even})_j,
\qquad \omega_N^{j+N/2}=-\omega_N^j
$$

allow the two half transforms to be reused. Recursion gives $T(N)=2T(N/2)+O(N)=O(N\log N)$.

The accurate statement is:

> the DFT has an exact rank-one complementary representation in the correct pair of orderings; the FFT follows from that structure **together with** the radix recursion and twiddle-factor algebra.

---

## 8. Three worked updates from E6–E12

The later experiments sharpen three parts of the framework without changing its scope. The full claim-by-claim audit is in [update-review.md](update-review.md).

### 8.1 The FFT example now has an exact storage certificate

Section 7 proves that the DFT tiles have rank one in the complementary residue-class orderings. The generic ID butterfly makes one further measured statement: for the bit-reversed DFT at $N=256,512,1024,2048$, its stored entry count is exactly

$$
2N\log_2N+2N,
$$

with every tile rank one and apply errors between $3\times10^{-14}$ and $2\times10^{-13}$ in the independent check. This is a statement about this factor representation, not a count of FFT arithmetic operations and not a theorem that every butterfly implementation has identical storage. Rank-one tiles explain the compression; the Cooley–Tukey recursion still explains the FFT algorithm.

### 8.2 Proxy sampling is a representation hypothesis, not a smoothness switch

A proxy set is useful only if observations on it determine the relevant far fields closely enough. A circle works for the ordinary 2-D Laplace Green's function because that function satisfies the corresponding local equation away from the source box. A generic smooth or power-law kernel need not satisfy that equation.

E8 makes the distinction observable. For the tested hypersingular block, refining a flat circle from 64 to 256 points leaves the far-field error in the interval $7.7\times10^{-2}$ to $9.5\times10^{-2}$. Spreading samples over several latitudes reduces the error by orders of magnitude; an in-plane annulus also works. Thus the effective ingredient is a sufficiently rich sample set for the exterior field, not merely more points and not necessarily a lift for its own sake. The local-lift proof covers the integer-codimension cases only—essentially $s=1/2$ for the families swept in E8; the other fractional cases are numerical evidence, not the same theorem.

E11 gives the corresponding multi-order calculation for Matérn kernels. With

$$
f_\mu(r)=(\kappa r)^\mu K_\mu(\kappa r),
$$

the radial Bessel identities give, for $r>0$,

$$
(\kappa^2-\Delta_D)f_\mu
=\kappa^2(2\mu+D-2)f_{\mu-1}.
$$

The coefficient vanishes at $\mu=1-D/2$, which is why the chain terminates in a compatible dimension. In this construction, taking $\beta=\lceil\nu+1\rceil$ and $D=2(\beta-\nu)$ gives a finite integer-dimensional local lift exactly when $2\nu\in\mathbb Z$. Otherwise the multi-order lifted rows are an empirical truncation, not a representation in a fractional-dimensional Euclidean space. At the tested sizes, a plain ring with one extra kernel order sometimes matches the lifted proxy with less ID work; necessity and cost must therefore be measured separately.

### 8.3 The complex row step has a general congruence check

The scalar counterexample in §5.3 generalizes. Suppose $\operatorname{Herm}(W)=I$ and $\Pi=[I;-T]$. For the adjoint row step,

$$
\operatorname{Herm}(\Pi^*W\Pi)=I+T^*T\succeq I.
$$

No such lower bound follows for $\Pi^{\mathsf T}W\Pi$ with complex $T$. [examples.py](audit-sources/examples.py) now checks this over 2,000 random instances: the smallest eigenvalue of the Hermitian part reached $-15.54$ for the transpose expression and stayed at least $1$, to roundoff, for the adjoint expression. E5 supplies the corresponding solver evidence; E9 shows that the practical loss is kernel- and regime-dependent. Neither observation replaces residual, pivot, and forward-error checks for a new problem.

## 9. Where the framework stops

The common language should issue a warning rather than automatically select an algorithm when:

- the matrix is sparse and another solver family is more natural;
- touching or diagonal blocks contain singular quadrature terms;
- high-frequency blocks lose ordinary low rank;
- a proxy trick lacks its PDE/representation hypotheses;
- the matrix is available only through expensive matrix–vector products;
- an elimination pivot is ill-conditioned;
- phase regularity or complementary low rank fails;
- construction cost dominates the cheap apply;
- global error accumulation, not just one-block error, determines accuracy.

These are not rare exceptions to hide in footnotes. They determine which paper and which implementation is needed.

---

## 10. A defensible research statement

The broad four-dial framework is **not new as a classification**. Relevant precedents include the admissibility/shared-basis taxonomy in Martinsson–O'Neil, its oscillatory extension in Li–Liu, and a four-component generic matrix–vector framework with implementation in Agullo–Bramas–Coulaud–Gicquel. Yang's NUFFT-versus-butterfly work is also a direct precedent for using a cost/rank analysis to choose an algorithm.

The supplied project can still be interesting, but its claim should be narrower:

> We provide a beginner-readable map of several hierarchical fast algorithms, small reference implementations, and controlled measurements of when their block structures and compression strategies work. We state separately what is proved, what is heuristic, and what is measured.

To become a stronger research contribution, it should answer at least one falsifiable question that is not just a re-description. Three viable directions are:

1. **A validated selection rule:** from geometry, frequency, accuracy, and access model, predict representation, storage, apply cost, build cost, and failure probability on held-out kernels. Test against existing implementations, not only against other formats in the same code base.
2. **A stability result:** explain and control complex skeletonization pivots, including transpose versus adjoint conventions. Establish reproducible regimes, a theorem with correct hypotheses, and a serious prior-art search before calling it new.
3. **A fair cross-format study:** compare nested with nested representations, equal achieved error rather than merely equal nominal tolerance, and include near-field, construction, memory, and solve stability.

The high-wavenumber instability in the supplied experiments is promising because it has a simple algebraic counterexample and an implementable fix. At present it is evidence of a worthwhile question, not yet an established research result.

## 11. The one-paragraph mental model

A dense PDE interaction is expensive because every source talks to every target. A hierarchical method groups sources and targets, then replaces some group-to-group interactions by shorter descriptions. The physics and geometry determine whether such a description exists; the representation determines how expensive it is to build and move; the assembly determines how it is used; and a solver additionally needs stable elimination. FMM, FMM-LU, butterfly, and FFT share this broad accounting problem, but each solves a different missing piece of it. That is the sound version of “one framework.”

## Primary references used for this reconstruction

- Cheng et al., [A wideband fast multipole method for the Helmholtz equation in three dimensions](https://amath.colorado.edu/faculty/martinss/2014_CBMS/Refs/2006_rokhlin_wideband.pdf), JCP 216 (2006).
- Minden, Ho, Damle, Ying, [A recursive skeletonization factorization based on strong admissibility](https://arxiv.org/abs/1609.08130) (RS-S).
- Sushnikova, Greengard, O'Neil, Rachh, [FMM-LU: A fast direct solver for multiscale boundary integral equations in three dimensions](https://arxiv.org/abs/2201.07325).
- Candès, Demanet, Ying, [A Fast Butterfly Algorithm for the Computation of Fourier Integral Operators](https://arxiv.org/abs/0809.0719), especially the analytic assumptions and $\log^4(1/\varepsilon)$ bound in Theorem 3.1.
- Li, Yang, Martin, Ho, Ying, [Butterfly Factorization](https://arxiv.org/abs/1502.01379).
- Zheng, Riccietti, Gribonval, [Efficient identification of butterfly sparse matrix factorizations](https://arxiv.org/abs/2110.01230), especially Example 3.3 for the DFT recursion and Proposition 3.19 for complementary low rank.
- Engquist, Zhao, *Approximate Separability of the Green's Function of the Helmholtz Equation in the High Frequency Limit*, CPAM 71 (2018).
- Yang, [A Unified Framework for Oscillatory Integral Transform: When to use NUFFT or Butterfly Factorization?](https://arxiv.org/abs/1803.04128).
- Li, Liu, [Parallel Sparse and Data-Sparse Factorization-Based Linear Solvers](https://arxiv.org/abs/2602.14289).
- Agullo, Bramas, Coulaud, Gicquel, [Unified and Composable Rank-Structured Hierarchical Matrix–Vector Multiplication](https://inria.hal.science/hal-05568584).
