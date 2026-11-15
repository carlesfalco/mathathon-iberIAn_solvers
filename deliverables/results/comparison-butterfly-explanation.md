# Explaining butterfly factorization: paper outline versus our outline

**Paper:** Li, Yang, Martin, Ho, Ying, [Butterfly Factorization](https://arxiv.org/abs/1502.01379), Multiscale Model. Simul. 13(2), 2015.

**Purpose:** present the same algorithm as a guided explanation, not as a replacement for the paper's construction and proofs. Our ordering is designed for a reader who has not seen phase space, randomized SVDs, or hierarchical matrix formats.

## The two outlines

| Stage | Paper's route | Our route, and why |
|---|---|---|
| **1. The task** | Introduction: dense matrix–vector multiplication; complementary low-rank matrices; the sparse product and the two-stage construction. | Start with an oscillatory kernel $K(x,\xi)=e^{2\pi i\Phi(x,\xi)}$ and the direct $O(N^2)$ apply. Use the DFT as the simplest example. This keeps rows = positions and columns = frequencies, so the later two-tree construction has a physical meaning. |
| **2. The block geometry** | Complementary low-rank property and the pair of trees appear in §1. | Show the two trees before any factorization: position boxes become narrower as frequency boxes become wider. Contrasting one big off-diagonal block with several differently shaped tiles answers **which blocks are compressible** before asking how to compress them. |
| **3. Why tiles are low rank** | Complementary low rank is introduced as an assumption/the property used by the algorithm. | Prove the simplest version: peel off the one-variable phases and bound the leftover mixed phase. For the bilinear DFT phase, the exponential series gives an immediate rank bound. Only then name the extra analytic hypotheses required for a general FIO. This avoids treating “one Heisenberg cell” as a proof. |
| **4. First compression pass: the middle level** | §2 Randomized SVD tools; §3.1 Middle-level factorization: rank-r approximations of every middle-level block. | Compress each tile once. Then make the naive accounting explicit: there are $N$ middle tiles of size $\sqrt N\times\sqrt N$, so independent rank-$r$ factors take $O(rN^{3/2})$ storage and apply work. This is useful, but **not yet the fast algorithm**. |
| **5. Why the ladder is necessary** | §3.2 Recursive factorization of $U^h$ and $V^h$: factor each basis into a sequence of sparse matrices. | Explain why a single shared basis does not work: the peeled phase depends on the partner tile. Each tile needs its own factors. Then show that these factors themselves have complementary structure and can be compressed recursively. This motivates the ladder rather than presenting it as notation. With SVD bases, the nesting is approximate by projection; interpolative bases provide another route. |
| **6. What the final representation buys** | Product $U^L G^{L-1}\cdots G^hM^h(H^h)^*\cdots(H^{L-1})^*(V^L)^*$; each factor sparse. | Translate the product into the merge rule: a tile with new rank $r_2$ stores at most $r_2(r_0+r_1)$ entries, where $r_0,r_1$ are the two old ranks. Bounded ranks therefore give $O(N)$ work per stage and $O(N\log N)$ overall. Zeros and identities can reduce storage further. This is the payoff of nesting—not just an algebraic formula to memorize. |
| **7. How to construct it** | §2.1 randomized SVD via matvecs; §2.2 SVD via random entry sampling; §3.1 applies them to middle tiles. | Compare the two access models directly: fast black-box matvecs give $O(N^{3/2}\log N)$ construction; sampled entries give $O(N^{3/2})$ under the paper's incoherence assumption. Randomized-SVD details belong after the reader understands why those samples are needed. Mention that the code's simple full-entry construction is slower, not the paper's optimal method. |
| **8. Complexity and evidence** | §3.3 Complexity analysis; §4 Numerical examples (FIO and special transforms). | Keep three budgets separate: **build**, **store**, **apply**. Also distinguish nominal per-tile tolerance from achieved global error. Use one DFT-tile rank measurement and one full apply error, rather than treating a single rank plot as proof of the full algorithm. |
| **9. FFT as the exact special case** | The DFT/FFT connection appears as background and as a butterfly-structured matrix. | Derive it after the general ladder: residue-class DFT tiles are exactly rank one in the correct pair of orderings. Then derive the separate Cooley–Tukey recombination. Emphasize that **rank one is exact here, not merely small phase-space rank**, and that bit-reversing only one tree is insufficient. |
| **10. What transfers to other algorithms** | Conclusion and future work. | Close with the structural lesson: choose a useful tiling, prove its blocks are small, then build nested transfers so local compression becomes a fast global operator. Do not imply that tile rank alone gives the whole algorithm. |

## The main editorial change

The paper's order is

> **randomized compression tools → middle-level tiles → recursive basis factorization → complexity analysis → examples.**

Our order is

> **two trees → why area-one tiles work → why independent tiles are not enough → why nested factors are necessary → construction choices → complexity and examples.**

This exposes a subtle but central point early: **the middle-level split and the nested ladder solve different problems**. The first gives local compression; the second turns that local compression into a cheap operator.

## What each version is for

- **Use the paper for:** the precise construction, randomized-SVD details, recursive formulas, incoherence assumptions, and complexity proofs.
- **Use our outline for:** slides and first-pass documentation that explains why those constructions exist.
- **Use them together:** introduce our block geometry and cost motivation first, then treat the paper's §§2–3 as the explicit algorithmic appendix.

A suitable slide title for this outline is:

> **Tiles give low rank; nesting gives speed.**
