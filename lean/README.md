# Lean proofs

Machine-checked statements of the Supplementary Information (SI) of "Fixation location in structured populations"
(Brewster, Lippner, Tkadlec, Nowak), with Lean 4 (v4.36.0-rc2) and Mathlib. Section 12 of the SI lists the exact
Lean statements next to the results they check.

Every theorem below depends only on Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`),
checked by `FixLoc/Audit.lean`.

| SI result | Lean name | Statement |
|---|---|---|
| Lemma S2, exact law, for real `z > 0` | `FixLoc.gf_head_advances` | If `F` satisfies the first-step equations `F 0 = 0`, `F N = 1`, `F (k+1) = α z F (k+2) + β F k`, then `F 1 = (2 α z)^(N-1) / P_N(1 - 4 α β z)`. Here `F x = E_x[z^(number of head advances) ; fixation]`, so `F 1 = ρ E[z^H]`. |
| Fixation probability | `FixLoc.fixation_probability` | For `r > 1`, `h 1 = (1 - r⁻¹) / (1 - r⁻ᴺ)` for every solution `h` of the first-step equations of the fixation probability. |
| Theorem S3, exact mean | `FixLoc.mean_head_advances` | For `r > 1`, `U 1 / h 1 = (N-1) r/(r-1) - (r+1)/(r-1)^2 + N(r+1)/((r-1)(r^N-1))`, where `U x = E_x[number of head advances ; fixation]`. |
| Proposition S4 (colonization, directed graphs) | `FixLoc.possible_fix_iff` | `t` is a possible fixation location under colonization exactly when `t ≠ u`, every other vertex is reachable from `u` by a path avoiding `t`, and `t` has an in-neighbor other than `t`. |
| Proposition S5 (colonization, undirected graphs) | `FixLoc.possible_fix_iff_not_cut` | For a symmetric, connected edge relation, `t` is a possible fixation location exactly when `t ≠ u` and `t` is not a cut vertex. |
| Theorem S1 (exact characterization) | `FixLoc.fixable_at_iff`, `FixLoc.bdstep_erase_univ_iff` | Fixation at `t` is possible exactly when the mutant set `V \ {t}` is reachable from `{s}` in the configuration graph and `t` has an in-neighbor other than `t`. |
| Corollary S1 (basic impossibility checks) | `FixLoc.not_fixable_of_unreachable`, `FixLoc.not_fixable_of_no_in` | A vertex unreachable from `s`, or no in-neighbor of `t` other than `t`, rules out fixation at `t` (no hypothesis `s ≠ t` needed). |
| Theorem S2 (refillability implies fixability) | `FixLoc.refillable_fixable` | A refillable path (structure `Refillable`, simple path given as a function on `{0, …, k}`, `k = 0` allowed) makes fixation at `t` possible (no hypothesis `s ≠ t` needed). |
| Proposition S3 (unidirectional bottlenecks) | `FixLoc.gateway`, `FixLoc.gateway_initial` | Conditions (i) and (ii) rule out `t` as a fixation location. Condition (ii) is `StaysMutant`, a statement about every trajectory ending in fixation. |
| Lemma S1 (single incoming neighbor) | `FixLoc.single_in` | For `t ≠ u`: if `t` has exactly one in-neighbor `x` and condition (ii) holds for `x`, then condition (ii) holds for `t`. |
| Lemma S2, Eq. (cycle-pgf), every complex `z` | `FixLoc.gf_head_advances_complex` | Every solution `F : ℕ → ℂ` of the first-step equations satisfies `F 1 = (2αz)^(N-1) / P_N(1 - 4αβz)`. |
| Lemma S2, Eq. (cycle-fourier) | `FixLoc.fourier_location` | For every distribution `q` of `H` on `ℕ`, `P(H mod N = k - 1) = (1/N) ∑_ℓ e^(-2πiℓ(k-1)/N) ∑_h q(h) e^(2πiℓh/N)` for `k = 1, …, N`. |
| Proposition S6 (neutral evolution) | `FixLoc.neutral_location_sharp`, `FixLoc.Cneu_lt`, `FixLoc.neutral_location_170` | Under the first-step equations at `r = 1` and the generating function at every `N`-th root of unity, `|p_k - 1/N| ≤ 16 C N^(-1/2) e^(-√(N/2))` for `N ≥ 2` and every vertex `k`, with `C = ∑_{ℓ≥1} √ℓ e^(1-√ℓ) < 10.6`, the constants of the SI proof, so the bound is at most `170 N^(-1/2) e^(-√(N/2))`. The earlier `FixLoc.neutral_location` (NeutralBound.lean) has the constant `3840 e`. |
| Proposition S6, the key lower bound | `FixLoc.PNc_lower` | `(1 + √(sin θ))^N - 1 ≤ 2√2 √(sin θ) |P_N(1 - e^(2iθ))|` for `0 < θ < π`. |
| Theorem S3, exact variance | `FixLoc.variance_head_advances`, `FixLoc.varRem_bounds` | For `r > 1`, `Var[H] = r(r+1)/(r-1)³ (N - 2(r+1)/(r-1)) + varRem r N` exactly, with `0 ≤ varRem r N ≤ r²(r+1)(r²+2r-1)/(r-1)⁵ N² r^(-N)` for `N ≥ 1`. The closed form of `E_x[H²; fixation]` comes from sympy (`checks/second_moment.py`, `checks/moments.py`). |
| Proposition S1 (factorization) | `FixLoc.factorization`, `FixLoc.factorization_strongly_connected` | On the jump chain of mutant sets, the probability of fixation at `v` equals the expected number of active steps starting with `V \ {v}` times `p_v = rθ_v/(rθ_v + 1[d⁺(v) ≥ 1])`, whenever absorption is reachable from every transient mutant set, in particular on every strongly connected graph. |
| Proposition S2 (final phase) | `FixLoc.final_phase` | `T^end(u) = (r(N-1)+1)/r ∑_v L(v | u) h(V \ {v}) / θ_v` when every vertex has an incoming and an outgoing edge. |
| Theorem S3, characteristic function | `FixLoc.clt_charFun` | For `r > 1` and every real `τ`, `e^(-iτE[H]/σ_N) E[e^(iτH/σ_N)] → e^(-τ²/2)` as `N → ∞`, with `σ_N² = Var[H]` and the law of `H` given through its generating function on the unit circle. |
| Theorem S3, central limit theorem | `FixLoc.clt_head_advances`, `FixLoc.clt_head_advances_of_steps` | For `r > 1`, the law of `(H - E[H])/√Var[H]` conditional on fixation converges in distribution to the standard normal law. The hypothesis is the generating function of Lemma S2 at `z = e^(iθ)` divided by `ρ`, or the first-step equations `GFStepsC` with `E[e^(iθH)] = F(1)/ρ`. |
| Corollary S2 (concentration of the fixation location) | `FixLoc.concentration_location` | For `r > 1` and `a > 0`, `liminf_N P(|H - m_N| ≤ a√N) ≥ 2Φ(a/σ_r) - 1`, with `m_N = (N-1)r/(r-1) - (r+1)/(r-1)²`, `σ_r² = r(r+1)/(r-1)³`, and `Φ = cdf (gaussianReal 0 1)`. |
| Lemma S2, the polynomial `P_N` | `FixLoc.PN`, `PN_succ_succ`, `PN_pos` | `P_N(y) = ∑ C(N, 2m+1) y^m` satisfies `P_{n+2} = 2 P_{n+1} - (1 - y) P_n`, `P_0 = 0`, `P_1 = 1`, and `P_N(y) > 0` for `N ≥ 1`, `y ≥ 0`. |

## Files

- `FixLoc/Poly.lean`: the polynomial `P_N` and its recurrence, first values, and positivity.
- `FixLoc/Walk.lean`: second-order recurrences with two boundary values (`canon`, `eq_mul_canon`, `bvp_unique`),
  the closed form of `canon` through `P_N`, the generating function, the fixation probability, and the exact mean.
  The closed forms `hClosed` and `UClosed` (with `q = r⁻¹`) are checked against the first-step equations, and
  uniqueness of the boundary value problem transfers them to every solution.
- `FixLoc/Variance.lean`: first-step equations of `E_x[H²; fixation]` (`SqSteps`), the closed form `WClosed`, the exact
  variance, and the bound on its remainder.
- `FixLoc/Factorization.lean`: finite absorbing chains (`AbsorbReach`, maximum principle `homogeneous_zero`,
  `factorization_chain`), the jump chain of the Birth-death process (`evw`, `actW`, `trW`, `jumpP`, `transient`, `theta`),
  SI Proposition S1, and reachability of absorption on strongly connected graphs.
- `FixLoc/FinalPhase.lean`: one time step of the Birth-death process (`stepP`, `totalFit`), `scale_chain`, and
  SI Proposition S2.
- `FixLoc/Fourier.lean`: `P_N`, `canon`, and the first-step equations over `ℂ` (`PNc`, `canonC`, `GFStepsC`), Lemma S2 for
  every complex `z`, the orthogonality of the powers of `ω = e^(2πi/N)`, and the Fourier formula.
- `FixLoc/Neutral.lean`: the identity `2x P_N(x²) = (1+x)^N - (1-x)^N`, the explicit square root
  `x = √(sin θ)(1 - i)e^(iθ/2)` of `1 - e^(2iθ)`, and the lower bound `PNc_lower` (about 5 minutes to build).
- `FixLoc/NeutralBound.lean`: the elementary inequalities, the bound of one term, the sum over the roots of unity,
  the neutral fixation probability `1/N`, and SI Proposition S6 (`neutral_location`).
- `FixLoc/NeutralConstant.lean`: the constant `C = ∑_{ℓ≥1} √ℓ e^(1-√ℓ)` (`Cneu`) and `C < 10.6` (`Cneu_lt`).
  The first 30 terms have certified bounds (`fC_le_1` to `fC_le_30`, generated by `checks/gen_cert.py`) from
  rational bounds on `√ℓ`, `2.71828 < e < 2.7182819`, and the Taylor polynomial of `exp` of degree 7. The
  remaining terms telescope against `G(x) = 2 e^(1-√x) (x + 2√x + 2)` (`fC_le_Gt_sub`, `tail_sum_le`), giving
  `C ≤ 9.55576 + 0.97638`. SI Proposition S6 with the constant `16 C` (`neutral_location_sharp`) and `170`
  (`neutral_location_170`).
- `FixLoc/Colonization.lean`: the colonization process as the inductive relation `ColReach` (a mutant at `x` places an
  offspring on an out-neighbor `y` occupied by a wild type), possible fixation locations (`PossibleFix`), the graph without
  `t` (`avoid`), cut vertices (`IsCutVertex`), and the two characterizations.
- `FixLoc/Fixability.lean`: Birth-death events as the edges of the configuration graph (`BDStep`), possible
  fixation locations at finite selection (`FixableAt`), trajectories indexed by time, refillable paths, condition (ii)
  (`StaysMutant`), and SI Theorem S1, Corollary S1, Theorem S2, Proposition S3, and Lemma S1.
- `FixLoc/CLTExpansion.lean`: the root `B(θ) = (1 - s(θ))/(2β)` of `βB² - B + αe^(iθ) = 0`, with `s` the principal square
  root of `1 - 4αβe^(iθ)`, its expansion `1 + rw/(r-1) - κw² + O(θ³)` in `w = iθ` (`Bf_expansion`), and the expansion
  `log B = rw/(r-1) + σ_r² w²/2 + O(θ³)` (`logBf_expansion`).
- `FixLoc/CLT.lean`: the factorization `E[e^(iθH)] = A B^N/(ρ(1 - R^N))` (`gf_factor`), the limits of `Var[H]/N` and of
  the exact mean, and the limit `e^(-τ²/2)` of the characteristic function of the standardized `H` (`clt_charFun`).
- `FixLoc/CLTLaw.lean`: the standardized law `stdLaw`, the standard normal law `stdNormal`, and convergence in
  distribution through Lévy's continuity theorem of Mathlib (`clt_head_advances`, `clt_head_advances_of_steps`).
- `FixLoc/Concentration.lean`: SI Corollary S2 (`concentration_location`) from the portmanteau theorem of Mathlib on the
  open intervals `(-c(k+1)/(k+2), c(k+1)/(k+2))` with `c = a/σ_r`, and `P(-c < Z < c) = 2Φ(c) - 1` (`gauss_Ioo`).
- `FixLoc/Islands.lean`: the neutral two-island chain on the mutant counts `(a, b)` (`up1`, `dn1`, `up2`, `dn2`, `stay`,
  `P`, `T`), the drift of the mutant counts (`drift1`, `drift2`), the harmonic function
  `h = (m₁₂ a + m₂₁ b)/(n (m₁₂ + m₂₁))` (`harmonic`), reachability of `(n, n)` (`reach`), the neutral fixation
  probability (`neutral_fixation`), the initial island (`neutral_initial_island`), and the algebra of the rare wild type
  approximations (`linearization`, `colonization_linearization`, `colonization_comparison`, `neutral_comparison`).
- `FixLoc/Audit.lean`: `#print axioms` for the headline theorems.

## The assumptions

As in the Mixed Star Lean project, the probabilities and expectations of the walk enter through their first-step
equations (structures `GFSteps`, `FixSteps`, `UpSteps`). The probability space of the walk is not constructed.
The true quantities satisfy the first-step equations by conditioning on the first step. The reduction from the
Birth-death process on the directed cycle to the walk of the block length (SI paragraph "The block of mutants")
is also not formalized.

Colonization (`r → ∞`) is modeled by its placements of positive probability (`ColReach`). Every placement of a mutant
offspring on a wild type has positive probability, so the mutant sets of positive probability are exactly the sets
reached by `ColReach`. The probability space of the colonization process is not constructed.

At finite fitness, every Birth-death event that changes the mutant set has positive probability, so the Birth-death
process is modeled by its events (`BDStep`, the configuration graph of the SI). Fixation at `t` has positive
probability exactly when some sequence of events realizes it (`FixableAt`). The probability space is not constructed.

The Lean proof of the generating function marks head advances with `z` directly. The SI proof marks all steps with
`s` and substitutes `n = 2H - N + 1`. The two routes give the same formula.

## Not covered

Lemma S1 for `t = u` (the SI proof covers only `t ≠ u`; an exhaustive search on all directed graphs with at most
6 vertices finds no counterexample for `t = u`).

## Building

Install [elan](https://github.com/leanprover/elan), the Lean toolchain manager. Then, in this folder:

    lake exe cache get
    lake build

The first command downloads the compiled Mathlib library. The second command checks every proof.
`FixLoc/Audit.lean` prints the axioms of the headline theorems during the build.
The files `checks/second_moment.py` and `checks/moments.py` need sympy. The file `checks/gen_cert.py` writes the
certified term bounds of `FixLoc/NeutralConstant.lean`.
