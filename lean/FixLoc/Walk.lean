import FixLoc.Poly

/-!
# The directed cycle: exact law and exact mean of the number of head advances

Machine-checked core of the SI section "Directed cycles" of "Fixation location in structured
populations" (Brewster, Lippner, Tkadlec, Nowak).

On the directed cycle `D_N`, the mutants form a block. At an active step the head advances with
probability `α` and the tail retreats with probability `β`. The block length performs a random walk on
`{0, …, N}`, absorbed at `0` (extinction) and at `N` (fixation). The fixation location is vertex
`1 + H` modulo `N`, with `H` the number of head advances.

As in the Mixed Star Lean project, probabilities and expectations of the walk enter through their
first-step equations (structures `GFSteps`, `FixSteps`, `UpSteps`). The probability space of the walk
is not constructed. The first-step equations hold for the true quantities by conditioning on the
first step.

* `gf_head_advances`: `E_1[z^H ; fixation] = (2αz)^(N-1) / P_N(1 - 4αβz)` (SI Lemma 2, for real `z > 0`).
* `fixation_probability`: `P_1(fixation) = (1 - r⁻¹) / (1 - r⁻ᴺ)` for `r > 1`.
* `mean_head_advances`: `E[H | fixation] = (N-1) r/(r-1) - (r+1)/(r-1)^2 + N(r+1)/((r-1)(r^N-1))`
  for `r > 1` (SI Theorem S3, exact mean).
-/

open Finset

namespace FixLoc

noncomputable section

/-! ## Second-order linear recurrences with two boundary conditions -/

/-- The solution of `u (k+2) = (u (k+1) - b u k) / a` with `u 0 = 0` and `u 1 = 1`. -/
def canon (a b : ℝ) : ℕ → ℝ
  | 0 => 0
  | 1 => 1
  | (k + 2) => (canon a b (k + 1) - b * canon a b k) / a

/-- Every solution of `G (k+1) = a G (k+2) + b G k` (for `k + 2 ≤ N`) with `G 0 = 0` is a multiple
of `canon a b` on `{0, …, N}`. -/
lemma eq_mul_canon (a b : ℝ) (ha : a ≠ 0) (N : ℕ) (G : ℕ → ℝ) (h0 : G 0 = 0)
    (hstep : ∀ k, k + 2 ≤ N → G (k + 1) = a * G (k + 2) + b * G k) :
    ∀ x, x ≤ N → G x = G 1 * canon a b x := by
  have pair : ∀ k, k + 1 ≤ N →
      G k = G 1 * canon a b k ∧ G (k + 1) = G 1 * canon a b (k + 1) := by
    intro k
    induction k with
    | zero =>
      intro _
      exact ⟨by simp [canon, h0], by simp [canon]⟩
    | succ k ih =>
      intro hk
      obtain ⟨h1, h2⟩ := ih (by omega)
      refine ⟨h2, ?_⟩
      have e := hstep k (by omega)
      rw [h1, h2] at e
      show G (k + 2) = G 1 * canon a b (k + 2)
      rw [show canon a b (k + 2) = (canon a b (k + 1) - b * canon a b k) / a from rfl,
        mul_div_assoc', eq_div_iff ha]
      linear_combination -e
  intro x hx
  rcases Nat.eq_zero_or_pos x with rfl | hpos
  · simp [canon, h0]
  · have := (pair (x - 1) (by omega)).2
    rwa [Nat.sub_add_cancel hpos] at this

/-- Closed form of `canon (α z) β` through `P_N`. -/
lemma canon_closed (α β z : ℝ) (hα : α ≠ 0) (hz : z ≠ 0) :
    ∀ x : ℕ, canon (α * z) β x = 2 * α * z * PN x (1 - 4 * α * β * z) / (2 * α * z) ^ x := by
  have hc : (2 * α * z) ≠ 0 := by
    have : (2 : ℝ) ≠ 0 := by norm_num
    exact mul_ne_zero (mul_ne_zero this hα) hz
  have pair : ∀ k : ℕ,
      canon (α * z) β k = 2 * α * z * PN k (1 - 4 * α * β * z) / (2 * α * z) ^ k ∧
      canon (α * z) β (k + 1) = 2 * α * z * PN (k + 1) (1 - 4 * α * β * z) / (2 * α * z) ^ (k + 1) := by
    intro k
    induction k with
    | zero =>
      refine ⟨by simp [canon, PN_zero], ?_⟩
      simp only [zero_add, pow_one, PN_one]
      rw [show canon (α * z) β 1 = 1 from rfl]
      field_simp
    | succ k ih =>
      obtain ⟨h1, h2⟩ := ih
      refine ⟨h2, ?_⟩
      show canon (α * z) β (k + 2) = _
      rw [show canon (α * z) β (k + 2) = (canon (α * z) β (k + 1) - β * canon (α * z) β k) / (α * z) from rfl,
        h1, h2, show k + 1 + 1 = k + 2 from rfl, PN_succ_succ k]
      field_simp
      ring
  exact fun x => (pair x).1

/-! ## The generating function of the number of head advances (SI Lemma 2) -/

/-- First-step equations of `F x = E_x[z^(number of up steps) ; the walk reaches N before 0]`.
From length `k + 1` the walk moves up with probability `α`, which multiplies by `z`, and down with
probability `β`. -/
structure GFSteps (α β z : ℝ) (N : ℕ) (F : ℕ → ℝ) : Prop where
  zero : F 0 = 0
  top : F N = 1
  step : ∀ k, k + 2 ≤ N → F (k + 1) = α * z * F (k + 2) + β * F k

/-- SI Lemma 2 for real `z > 0`: `E_1[z^H ; fixation] = (2αz)^(N-1) / P_N(1 - 4αβz)`. -/
theorem gf_head_advances (α β z : ℝ) (hα : 0 < α) (hz : 0 < z) (N : ℕ) (hN : 1 ≤ N)
    (F : ℕ → ℝ) (hF : GFSteps α β z N F) :
    F 1 = (2 * α * z) ^ (N - 1) / PN N (1 - 4 * α * β * z) := by
  have ha : α * z ≠ 0 := by positivity
  have hsol := eq_mul_canon (α * z) β ha N F hF.zero hF.step N le_rfl
  rw [hF.top, canon_closed α β z hα.ne' hz.ne' N] at hsol
  have hc : (2 * α * z) ≠ 0 := by positivity
  have hP : PN N (1 - 4 * α * β * z) ≠ 0 := by
    intro h
    rw [h] at hsol
    simp at hsol
  obtain ⟨M, rfl⟩ : ∃ M, N = M + 1 := ⟨N - 1, by omega⟩
  simp only [Nat.add_sub_cancel]
  rw [eq_div_iff hP]
  rw [pow_succ, mul_comm ((2 * α * z) ^ M) (2 * α * z), mul_div_mul_left _ _ hc, mul_div_assoc',
    eq_div_iff (pow_ne_zero _ hc), one_mul] at hsol
  exact hsol.symm

/-! ## Fixation probability and the mean number of head advances (SI Theorem S3) -/

/-- Probability of a head advance at an active step. -/
def alpha (r : ℝ) : ℝ := r / (1 + r)

/-- Probability of a tail retreat at an active step. -/
def beta (r : ℝ) : ℝ := 1 / (1 + r)

/-- First-step equations of `h x = P_x(the walk reaches N before 0)`. -/
structure FixSteps (r : ℝ) (N : ℕ) (h : ℕ → ℝ) : Prop where
  zero : h 0 = 0
  top : h N = 1
  step : ∀ k, k + 2 ≤ N → h (k + 1) = alpha r * h (k + 2) + beta r * h k

/-- First-step equations of `U x = E_x[number of up steps ; the walk reaches N before 0]`.
An up step from length `k + 1` adds `1` on the event of fixation, which has probability `h (k + 2)`
from the new length. -/
structure UpSteps (r : ℝ) (N : ℕ) (h U : ℕ → ℝ) : Prop where
  zero : U 0 = 0
  top : U N = 0
  step : ∀ k, k + 2 ≤ N → U (k + 1) = alpha r * (U (k + 2) + h (k + 2)) + beta r * U k

/-- The homogeneous boundary value problem has only the zero solution. -/
lemma bvp_unique (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (D : ℕ → ℝ) (h0 : D 0 = 0) (hN0 : D N = 0)
    (hstep : ∀ k, k + 2 ≤ N → D (k + 1) = alpha r * D (k + 2) + beta r * D k) :
    ∀ x, x ≤ N → D x = 0 := by
  have hr0 : 0 < r := by linarith
  have hα : 0 < alpha r := by unfold alpha; positivity
  have hβ : 0 < beta r := by unfold beta; positivity
  have hsol := eq_mul_canon (alpha r * 1) (beta r) (by positivity) N D h0
    (by intro k hk; rw [mul_one]; exact hstep k hk)
  have hcN : canon (alpha r * 1) (beta r) N ≠ 0 := by
    rw [canon_closed (alpha r) (beta r) 1 hα.ne' one_ne_zero N]
    have h1r : (1 + r) ≠ 0 := by positivity
    have hyeq : 1 - 4 * alpha r * beta r * 1 = ((r - 1) / (1 + r)) ^ 2 := by
      unfold alpha beta
      field_simp
      ring
    have hP := PN_pos N hN (1 - 4 * alpha r * beta r * 1) (by rw [hyeq]; positivity)
    positivity
  have hD1 : D 1 = 0 := by
    have e := hsol N le_rfl
    rw [hN0] at e
    rcases mul_eq_zero.mp e.symm with h | h
    · exact h
    · exact absurd h hcN
  intro x hx
  rw [hsol x hx, hD1, zero_mul]

/-- Closed form of the fixation probability, with `q = r⁻¹`. -/
def hClosed (r : ℝ) (N : ℕ) (x : ℕ) : ℝ := (1 - r⁻¹ ^ x) / (1 - r⁻¹ ^ N)

/-- Closed form of `U`, with `q = r⁻¹`:
`U x = (N (1 - q^x)(1 + q^(N+1)) - x (1 - q^N)(1 + q^(x+1))) / ((1 - q)(1 - q^N)^2)`. -/
def UClosed (r : ℝ) (N : ℕ) (x : ℕ) : ℝ :=
  ((N : ℝ) * (1 - r⁻¹ ^ x) * (1 + r⁻¹ * r⁻¹ ^ N) - (x : ℝ) * (1 - r⁻¹ ^ N) * (1 + r⁻¹ * r⁻¹ ^ x))
    / ((1 - r⁻¹) * (1 - r⁻¹ ^ N) ^ 2)

lemma alpha_eq (r : ℝ) (hr : 1 < r) : alpha r = 1 / (1 + r⁻¹) := by
  unfold alpha
  have h0 : 0 < r := by linarith
  have h1 : 1 + r ≠ 0 := by positivity
  have h2 : 1 + r⁻¹ ≠ 0 := by positivity
  rw [div_eq_div_iff h1 h2]
  field_simp
  ring

lemma beta_eq (r : ℝ) (hr : 1 < r) : beta r = r⁻¹ / (1 + r⁻¹) := by
  unfold beta
  have h0 : 0 < r := by linarith
  have h1 : 1 + r ≠ 0 := by positivity
  have h2 : 1 + r⁻¹ ≠ 0 := by positivity
  rw [div_eq_div_iff h1 h2]
  field_simp
  ring

lemma qN_lt_one (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) : r⁻¹ ^ N < 1 := by
  have h0 : 0 < r⁻¹ := by positivity
  have h1 : r⁻¹ < 1 := inv_lt_one_of_one_lt₀ hr
  exact pow_lt_one₀ h0.le h1 (by omega)

/-- Algebraic step identity for `hClosed`, with `t = q^k`. -/
lemma h_step_alg (q t Q : ℝ) (hq : 1 + q ≠ 0) (hQ : 1 - Q ≠ 0) :
    (1 - q * t) / (1 - Q) = 1 / (1 + q) * ((1 - q * (q * t)) / (1 - Q)) + q / (1 + q) * ((1 - t) / (1 - Q)) := by
  field_simp
  ring

/-- Algebraic step identity for `UClosed`, with `t = q^k`, `Q = q^N`, and `x = k`. -/
lemma U_step_alg (q t Q n x : ℝ) (hq : 1 + q ≠ 0) (hq1 : 1 - q ≠ 0) (hQ : 1 - Q ≠ 0) :
    (n * (1 - q * t) * (1 + q * Q) - (x + 1) * (1 - Q) * (1 + q * (q * t))) / ((1 - q) * (1 - Q) ^ 2)
      = 1 / (1 + q) * ((n * (1 - q * (q * t)) * (1 + q * Q) - (x + 2) * (1 - Q) * (1 + q * (q * (q * t))))
            / ((1 - q) * (1 - Q) ^ 2) + (1 - q * (q * t)) / (1 - Q))
        + q / (1 + q) * ((n * (1 - t) * (1 + q * Q) - x * (1 - Q) * (1 + q * t)) / ((1 - q) * (1 - Q) ^ 2)) := by
  field_simp
  ring

lemma hClosed_steps (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) : FixSteps r N (hClosed r N) := by
  have hr0 : 0 < r := by linarith
  have hQ : 1 - r⁻¹ ^ N ≠ 0 := (sub_pos.mpr (qN_lt_one r hr N hN)).ne'
  have hq : 1 + r⁻¹ ≠ 0 := by positivity
  refine ⟨by simp [hClosed], by unfold hClosed; exact div_self hQ, ?_⟩
  intro k _
  rw [alpha_eq r hr, beta_eq r hr]
  unfold hClosed
  have e := h_step_alg r⁻¹ (r⁻¹ ^ k) (r⁻¹ ^ N) hq hQ
  have p1 : r⁻¹ ^ (k + 1) = r⁻¹ * r⁻¹ ^ k := by ring
  have p2 : r⁻¹ ^ (k + 2) = r⁻¹ * (r⁻¹ * r⁻¹ ^ k) := by ring
  rw [p1, p2]
  exact e

lemma UClosed_steps (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) :
    UpSteps r N (hClosed r N) (UClosed r N) := by
  have hr0 : 0 < r := by linarith
  have hQ : 1 - r⁻¹ ^ N ≠ 0 := (sub_pos.mpr (qN_lt_one r hr N hN)).ne'
  have hq : 1 + r⁻¹ ≠ 0 := by positivity
  have hq1 : 1 - r⁻¹ ≠ 0 := (sub_pos.mpr (inv_lt_one_of_one_lt₀ hr)).ne'
  refine ⟨by simp [UClosed], by simp [UClosed], ?_⟩
  intro k _
  rw [alpha_eq r hr, beta_eq r hr]
  unfold UClosed hClosed
  have e := U_step_alg r⁻¹ (r⁻¹ ^ k) (r⁻¹ ^ N) (N : ℝ) (k : ℝ) hq hq1 hQ
  have p1 : r⁻¹ ^ (k + 1) = r⁻¹ * r⁻¹ ^ k := by ring
  have p2 : r⁻¹ ^ (k + 2) = r⁻¹ * (r⁻¹ * r⁻¹ ^ k) := by ring
  rw [p1, p2]
  push_cast
  linear_combination e

/-- The fixation probability of the walk started at `1`. -/
theorem fixation_probability (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (h : ℕ → ℝ)
    (hh : FixSteps r N h) : h 1 = (1 - r⁻¹) / (1 - r⁻¹ ^ N) := by
  have hc := hClosed_steps r hr N hN
  have := bvp_unique r hr N hN (fun x => h x - hClosed r N x) (by simp [hh.zero, hc.zero])
    (by simp [hh.top, hc.top])
    (by intro k hk; rw [hh.step k hk, hc.step k hk]; ring) 1 hN
  simp only [sub_eq_zero] at this
  rw [this]
  simp [hClosed]

/-- SI Theorem S3, exact mean: for `r > 1`,
`E[H | fixation] = U 1 / h 1 = (N-1) r/(r-1) - (r+1)/(r-1)^2 + N(r+1)/((r-1)(r^N-1))`. -/
theorem mean_head_advances (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (h U : ℕ → ℝ)
    (hh : FixSteps r N h) (hU : UpSteps r N h U) :
    U 1 / h 1 = ((N : ℝ) - 1) * r / (r - 1) - (r + 1) / (r - 1) ^ 2
      + (N : ℝ) * (r + 1) / ((r - 1) * (r ^ N - 1)) := by
  have hc := hClosed_steps r hr N hN
  have hCU := UClosed_steps r hr N hN
  -- h agrees with the closed form
  have hEq : ∀ x, x ≤ N → h x = hClosed r N x := by
    intro x hx
    have := bvp_unique r hr N hN (fun x => h x - hClosed r N x) (by simp [hh.zero, hc.zero])
      (by simp [hh.top, hc.top])
      (by intro k hk; rw [hh.step k hk, hc.step k hk]; ring) x hx
    linarith
  -- U agrees with the closed form
  have hUEq : U 1 = UClosed r N 1 := by
    have := bvp_unique r hr N hN (fun x => U x - UClosed r N x) (by simp [hU.zero, hCU.zero])
      (by simp [hU.top, hCU.top])
      (by
        intro k hk
        rw [hU.step k hk, hCU.step k hk, hEq (k + 2) hk]
        ring) 1 hN
    linarith
  rw [hUEq, hEq 1 hN]
  unfold UClosed hClosed
  have hr0 : r ≠ 0 := (by linarith : (0 : ℝ) < r).ne'
  have hr1 : r - 1 ≠ 0 := (by linarith : (0 : ℝ) < r - 1).ne'
  have hrN : r ^ N - 1 ≠ 0 := (sub_pos.mpr (one_lt_pow₀ hr (by omega))).ne'
  have hQ : 1 - r⁻¹ ^ N ≠ 0 := (sub_pos.mpr (qN_lt_one r hr N hN)).ne'
  have hrN0 : r ^ N ≠ 0 := pow_ne_zero _ hr0
  simp only [pow_one, inv_pow, Nat.cast_one]
  field_simp
  ring

end

end FixLoc
