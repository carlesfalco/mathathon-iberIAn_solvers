# E9 hypotheses — written BEFORE any code was run (2026-11-14)

Problem: 2D Lippmann–Schwinger, u − k² ∫ G_k(x−y) q(y) u(y) dy = u_inc, on grid points filling the support
of q (disk / smooth bump). Matrix A = I + K D, K_ij = G_k(x_i, x_j) (i ≠ j), D = diag(−k² q_j h²).

Dials expected to stay unchanged (no code change):
1. Clustering: the same quadtree (`Tree`) already handles area points (P2 is an area problem).
2. Admissibility: Weak / Strong unchanged in code. Their *behaviour* changes (H1).
3. Representation: skeleton/ID (one ID per box, stacked rows+columns) unchanged.
5. Algebra: `rskel_factor` unchanged.

Dial expected to change: 4 (sampling) for the proxy compressor only — the kernel handed to it must be
G(x,y)·d(y) at unknowns but must NOT be scaled by d at proxy points (H3).

H1 (ranks). Strong admissibility: skeleton size per box bounded in N at fixed k, grows ~linearly in k·(box
width) (proxy circle needs ~kR modes). Weak admissibility on area data: top-level skeletons grow like √N
(the box touches its neighbours along an edge densely filled with points, rank ~ number of points along
the interface), worse than on a curve (where weak ranks are ~log N). Expect weak level-2 skel_max ∝ N^0.5.

H2 (one-sided contrast). The single stacked ID of M = [A_FB; A_BFᵀ] = [K_FB D_B; D_F K_FB] must satisfy two
column relations at once: K_FB D_B (column-scaled) and K_FB (row scaling by D_F doesn't change relations).
If d is constant on a box the two coincide (no penalty); if d varies on the box (smooth bump) the stacked
rank is larger than either block alone, by a factor in (1, 2]. Expected ratio ≈ 1.1–1.5 for a smooth bump.

H3 (proxy). Proxy with a kernel that scales by d at the proxy points ("naive": d(y) = −k² q(y) h², zero
outside the support) still works when d is constant on each box (constant disk, staircase grid), but fails
(error ≫ ε) for the variable bump, worst at upper levels where most of the proxy circle lies outside the
support. A "two-sided" proxy kernel (d at unknowns, 1 at proxy points) restores ε-accuracy in all cases.
Green's representation still holds because d only rescales columns (sources) and rows (targets).

H4 (Tᵀ vs Tᴴ, E5). Herm(A) is NOT ⪰ μI here once k²·q·a² is O(1) (strong scattering), so E5's theorem
does not apply to any variant. In the resolved regime (k·hw ≤ ~10 at the top level) we expect exact-ID
RS to be fine (E5 broke at k·hw ≳ 12–30). If we push k higher at fixed N (under-resolved, algebra only),
expect the exact (Tᵀ) variant to degrade like E5 and the realified (real T) variant to stay at ε as long as
A itself is well conditioned. Uncertain: whether indefinite Herm(A) breaks realified too.

H5 (verification). Dense LS solve converges to the Mie series for the constant disk at O(h) (staircase
boundary) and to the radial-ODE reference for the C² bump at O(h²). The discrete scattering coefficients
satisfy unitarity |1 + 2 b_n i⁻ⁿ| = 1 (lossless energy conservation) up to discretization error.
Fast solvers agree with the dense solve to ~ε·cond(A).
