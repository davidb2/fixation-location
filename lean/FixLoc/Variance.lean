import FixLoc.Walk

/-!
# The directed cycle: exact variance of the number of head advances (SI Theorem S3)

`W x = E_x[H² ; fixation]` satisfies first-step equations driven by `U` and `h`
(structure `SqSteps`). The closed form `WClosed` was found by solving the recurrence with sympy
(`checks/second_moment.py`) and is checked here by `field_simp` and
`ring`. Uniqueness of the boundary value problem transfers the closed form to every solution.

* `second_moment_head_advances`: `W = WClosed` on `{0, …, N}`.
* `variance_head_advances`: `Var[H | fixation] = r(r+1)/(r-1)³ (N - 2(r+1)/(r-1)) + varRem r N`
  exactly, for `r > 1`.
* `varRem_bounds`: `0 ≤ varRem r N ≤ K(r) N² r^(-N)` with `K(r) = r²(r+1)(r²+2r-1)/(r-1)⁵`, the
  remainder `O(N² r^(-N))` of SI Theorem S3.
-/

open Finset

namespace FixLoc

noncomputable section

/-- First-step equations of `W x = E_x[H² ; fixation]`. A head advance from position `k + 1` turns
`H` into `H + 1`, and `(H + 1)² = H² + 2H + 1`. -/
structure SqSteps (r : ℝ) (N : ℕ) (h U W : ℕ → ℝ) : Prop where
  zero : W 0 = 0
  top : W N = 0
  step : ∀ k, k + 2 ≤ N →
    W (k + 1) = alpha r * (W (k + 2) + 2 * U (k + 2) + h (k + 2)) + beta r * W k

/-- Coefficients of the closed form of `W`, with `q = 1/r`, `Q = q^N`, `n = N`. -/
def wA (q Q n : ℝ) : ℝ :=
  n * (n * Q ^ 2 * q ^ 3 - n * Q ^ 2 * q ^ 2 + n * Q * q ^ 3 + 3 * n * Q * q ^ 2 - 3 * n * Q * q
    - n * Q + n * q - n + Q ^ 2 * q ^ 2 + Q ^ 2 * q - q ^ 2 - q) / ((Q - 1) ^ 3 * (q - 1) ^ 3)

def wB (q Q n : ℝ) : ℝ :=
  (2 * n * Q * q ^ 2 - 2 * n * Q * q + 2 * n * q - 2 * n + Q * q ^ 2 + Q * q - q ^ 2 - q)
    / ((Q - 1) ^ 2 * (q - 1) ^ 3)

def wC (q Q n : ℝ) : ℝ :=
  q * (2 * n * Q * q ^ 2 - 2 * n * Q * q + 2 * n * q - 2 * n + Q * q + Q - q - 1)
    / ((Q - 1) ^ 2 * (q - 1) ^ 3)

/-- `W(x) = c₀ + c₁ t + c₂ x + c₃ x t + c₄ x² + c₅ x² t` with `t = q^x`. -/
def Wform (q t Q n x : ℝ) : ℝ :=
  -wA q Q n + wA q Q n * t - wB q Q n * x - wC q Q n * x * t
    - x ^ 2 / ((Q - 1) * (q - 1) ^ 2) + q ^ 2 * x ^ 2 * t / ((Q - 1) * (q - 1) ^ 2)

/-- The closed form of `U` in the same variables. -/
def Uform (q t Q n x : ℝ) : ℝ :=
  (n * (1 - t) * (1 + q * Q) - x * (1 - Q) * (1 + q * t)) / ((1 - q) * (1 - Q) ^ 2)

/-- Algebraic step identity for `Wform`, with `t = q^k`, `Q = q^N`, `x = k`. -/
lemma W_step_alg (q t Q n x : ℝ) (hq : 1 + q ≠ 0) (hq1 : q - 1 ≠ 0) (hQ : Q - 1 ≠ 0) :
    Wform q (q * t) Q n (x + 1)
      = 1 / (1 + q) * (Wform q (q * (q * t)) Q n (x + 2) + 2 * Uform q (q * (q * t)) Q n (x + 2)
          + (1 - q * (q * t)) / (1 - Q))
        + q / (1 + q) * Wform q t Q n x := by
  have hq1' : 1 - q ≠ 0 := fun h => hq1 (by linarith)
  have hQ' : 1 - Q ≠ 0 := fun h => hQ (by linarith)
  unfold Wform wA wB wC Uform
  field_simp
  ring

lemma W_top_alg (q Q n : ℝ) (hq1 : q - 1 ≠ 0) (hQ : Q - 1 ≠ 0) : Wform q Q Q n n = 0 := by
  unfold Wform wA wB wC
  field_simp
  ring

/-- Closed form of `W`, with `q = r⁻¹`. -/
def WClosed (r : ℝ) (N : ℕ) (x : ℕ) : ℝ := Wform r⁻¹ (r⁻¹ ^ x) (r⁻¹ ^ N) N x

lemma UClosed_eq_Uform (r : ℝ) (N x : ℕ) :
    UClosed r N x = Uform r⁻¹ (r⁻¹ ^ x) (r⁻¹ ^ N) N x := by
  unfold UClosed Uform
  ring

lemma WClosed_steps (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) :
    SqSteps r N (hClosed r N) (UClosed r N) (WClosed r N) := by
  have hr0 : 0 < r := by linarith
  have hQ : r⁻¹ ^ N - 1 ≠ 0 := (sub_neg.mpr (qN_lt_one r hr N hN)).ne
  have hq : 1 + r⁻¹ ≠ 0 := by positivity
  have hq1 : r⁻¹ - 1 ≠ 0 := (sub_neg.mpr (inv_lt_one_of_one_lt₀ hr)).ne
  refine ⟨?_, ?_, ?_⟩
  · unfold WClosed Wform
    simp
  · unfold WClosed
    exact W_top_alg r⁻¹ (r⁻¹ ^ N) N hq1 hQ
  · intro k _
    rw [alpha_eq r hr, beta_eq r hr, UClosed_eq_Uform]
    unfold WClosed hClosed
    have e := W_step_alg r⁻¹ (r⁻¹ ^ k) (r⁻¹ ^ N) (N : ℝ) (k : ℝ) hq hq1 hQ
    have p1 : r⁻¹ ^ (k + 1) = r⁻¹ * r⁻¹ ^ k := by ring
    have p2 : r⁻¹ ^ (k + 2) = r⁻¹ * (r⁻¹ * r⁻¹ ^ k) := by ring
    rw [p1, p2]
    push_cast
    linear_combination e

/-- `h` and `U` agree with their closed forms on `{0, …, N}`. -/
lemma h_U_closed (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (h U : ℕ → ℝ)
    (hh : FixSteps r N h) (hU : UpSteps r N h U) :
    (∀ x, x ≤ N → h x = hClosed r N x) ∧ (∀ x, x ≤ N → U x = UClosed r N x) := by
  have hc := hClosed_steps r hr N hN
  have hCU := UClosed_steps r hr N hN
  have hEq : ∀ x, x ≤ N → h x = hClosed r N x := by
    intro x hx
    have := bvp_unique r hr N hN (fun x => h x - hClosed r N x) (by simp [hh.zero, hc.zero])
      (by simp [hh.top, hc.top])
      (by intro k hk; rw [hh.step k hk, hc.step k hk]; ring) x hx
    linarith
  refine ⟨hEq, ?_⟩
  intro x hx
  have := bvp_unique r hr N hN (fun x => U x - UClosed r N x) (by simp [hU.zero, hCU.zero])
    (by simp [hU.top, hCU.top])
    (by
      intro k hk
      rw [hU.step k hk, hCU.step k hk, hEq (k + 2) hk]
      ring) x hx
  linarith

/-- The second moment: `W` agrees with `WClosed` on `{0, …, N}`. -/
theorem second_moment_head_advances (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (h U W : ℕ → ℝ)
    (hh : FixSteps r N h) (hU : UpSteps r N h U) (hW : SqSteps r N h U W) :
    ∀ x, x ≤ N → W x = WClosed r N x := by
  obtain ⟨hEq, hUEq⟩ := h_U_closed r hr N hN h U hh hU
  have hCW := WClosed_steps r hr N hN
  intro x hx
  have := bvp_unique r hr N hN (fun x => W x - WClosed r N x) (by simp [hW.zero, hCW.zero])
    (by simp [hW.top, hCW.top])
    (by
      intro k hk
      rw [hW.step k hk, hCW.step k hk, hEq (k + 2) hk, hUEq (k + 2) hk]
      ring) x hx
  linarith

/-- `E[H | fixation]` in the variables `q`, `Q`, `n`. -/
def EHform (q Q n : ℝ) : ℝ :=
  (Q * n * q ^ 2 - Q * n * q - Q * q ^ 2 - Q + n * q - n + q ^ 2 + 1) / ((Q - 1) * (q - 1) ^ 2)

/-- `E[H² | fixation]` in the variables `q`, `Q`, `n`. -/
def EH2form (q Q n : ℝ) : ℝ :=
  (Q ^ 2 * n ^ 2 * q ^ 4 - 2 * Q ^ 2 * n ^ 2 * q ^ 3 + Q ^ 2 * n ^ 2 * q ^ 2 - 2 * Q ^ 2 * n * q ^ 4
    + 3 * Q ^ 2 * n * q ^ 3 - 2 * Q ^ 2 * n * q ^ 2 + Q ^ 2 * n * q + Q ^ 2 * q ^ 4 - 2 * Q ^ 2 * q ^ 3
    - 2 * Q ^ 2 * q ^ 2 - 2 * Q ^ 2 * q + Q ^ 2 + Q * n ^ 2 * q ^ 4 + 2 * Q * n ^ 2 * q ^ 3
    - 6 * Q * n ^ 2 * q ^ 2 + 2 * Q * n ^ 2 * q + Q * n ^ 2 + 2 * Q * n * q ^ 4 - 4 * Q * n * q ^ 3
    + 4 * Q * n * q ^ 2 - 4 * Q * n * q + 2 * Q * n - 2 * Q * q ^ 4 + 4 * Q * q ^ 3 + 4 * Q * q ^ 2
    + 4 * Q * q - 2 * Q + n ^ 2 * q ^ 2 - 2 * n ^ 2 * q + n ^ 2 + n * q ^ 3 - 2 * n * q ^ 2
    + 3 * n * q - 2 * n + q ^ 4 - 2 * q ^ 3 - 2 * q ^ 2 - 2 * q + 1) / ((Q - 1) ^ 2 * (q - 1) ^ 4)

lemma U1_alg (q Q n : ℝ) (hq1 : q - 1 ≠ 0) (hQ : Q - 1 ≠ 0) :
    Uform q q Q n 1 = (1 - q) / (1 - Q) * EHform q Q n := by
  have hq1' : 1 - q ≠ 0 := fun h => hq1 (by linarith)
  have hQ' : 1 - Q ≠ 0 := fun h => hQ (by linarith)
  unfold Uform EHform
  field_simp
  ring

set_option maxRecDepth 20000 in
lemma W1_alg (q Q n : ℝ) (hq1 : q - 1 ≠ 0) (hQ : Q - 1 ≠ 0) :
    Wform q q Q n 1 = (1 - q) / (1 - Q) * EH2form q Q n := by
  have hq1' : 1 - q ≠ 0 := fun h => hq1 (by linarith)
  have hQ' : 1 - Q ≠ 0 := fun h => hQ (by linarith)
  unfold Wform wA wB wC EH2form
  field_simp
  ring

set_option maxRecDepth 20000 in
lemma var_alg (q Q n : ℝ) (hq1 : q - 1 ≠ 0) (hQ : Q - 1 ≠ 0) :
    EH2form q Q n - EHform q Q n ^ 2
      = q * (1 + q) / (1 - q) ^ 3 * (n - 2 * (1 + q) / (1 - q))
        + n * Q * (1 + q) * (n * (1 - q ^ 2) + 2 * q * (1 - Q)) / ((1 - Q) ^ 2 * (1 - q) ^ 3) := by
  have hq1' : 1 - q ≠ 0 := fun h => hq1 (by linarith)
  have hQ' : 1 - Q ≠ 0 := fun h => hQ (by linarith)
  unfold EH2form EHform
  field_simp
  ring

lemma lead_conv (r n : ℝ) (hr0 : r ≠ 0) (hr1 : r - 1 ≠ 0) (hq : 1 - r⁻¹ ≠ 0) :
    r⁻¹ * (1 + r⁻¹) / (1 - r⁻¹) ^ 3 * (n - 2 * (1 + r⁻¹) / (1 - r⁻¹))
      = r * (r + 1) / (r - 1) ^ 3 * (n - 2 * (r + 1) / (r - 1)) := by
  field_simp

lemma rem_conv (r n Q : ℝ) (hr0 : r ≠ 0) (hr1 : r - 1 ≠ 0) (hq : 1 - r⁻¹ ≠ 0) (hQ : 1 - Q ≠ 0) :
    n * Q * (1 + r⁻¹) * (n * (1 - r⁻¹ ^ 2) + 2 * r⁻¹ * (1 - Q)) / ((1 - Q) ^ 2 * (1 - r⁻¹) ^ 3)
      = n * Q * (r + 1) * (n * (r ^ 2 - 1) + 2 * r * (1 - Q)) / ((1 - Q) ^ 2 * (r - 1) ^ 3) := by
  field_simp

/-- The remainder of the variance, `O(N² r^(-N))`. -/
def varRem (r : ℝ) (N : ℕ) : ℝ :=
  (N : ℝ) * r⁻¹ ^ N * (r + 1) * ((N : ℝ) * (r ^ 2 - 1) + 2 * r * (1 - r⁻¹ ^ N))
    / ((1 - r⁻¹ ^ N) ^ 2 * (r - 1) ^ 3)

/-- SI Theorem S3, exact variance: for `r > 1`,
`Var[H | fixation] = r(r+1)/(r-1)³ (N - 2(r+1)/(r-1)) + varRem r N`. -/
theorem variance_head_advances (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) (h U W : ℕ → ℝ)
    (hh : FixSteps r N h) (hU : UpSteps r N h U) (hW : SqSteps r N h U W) :
    W 1 / h 1 - (U 1 / h 1) ^ 2
      = r * (r + 1) / (r - 1) ^ 3 * ((N : ℝ) - 2 * (r + 1) / (r - 1)) + varRem r N := by
  obtain ⟨hEq, hUEq⟩ := h_U_closed r hr N hN h U hh hU
  have hr0 : r ≠ 0 := (by linarith : (0 : ℝ) < r).ne'
  have hr1 : r - 1 ≠ 0 := (by linarith : (0 : ℝ) < r - 1).ne'
  have hQ : 1 - r⁻¹ ^ N ≠ 0 := (sub_pos.mpr (qN_lt_one r hr N hN)).ne'
  have hQ' : r⁻¹ ^ N - 1 ≠ 0 := fun h => hQ (by linarith)
  have hq1 : r⁻¹ - 1 ≠ 0 := (sub_neg.mpr (inv_lt_one_of_one_lt₀ hr)).ne
  have hq1' : 1 - r⁻¹ ≠ 0 := fun h => hq1 (by linarith)
  have hc : (1 - r⁻¹) / (1 - r⁻¹ ^ N) ≠ 0 := div_ne_zero hq1' hQ
  rw [second_moment_head_advances r hr N hN h U W hh hU hW 1 hN, hUEq 1 hN, hEq 1 hN,
    UClosed_eq_Uform]
  unfold WClosed hClosed varRem
  simp only [pow_one, Nat.cast_one]
  rw [W1_alg r⁻¹ (r⁻¹ ^ N) N hq1 hQ', U1_alg r⁻¹ (r⁻¹ ^ N) N hq1 hQ',
    mul_div_cancel_left₀ _ hc, mul_div_cancel_left₀ _ hc, var_alg r⁻¹ (r⁻¹ ^ N) N hq1 hQ',
    lead_conv r N hr0 hr1 hq1', rem_conv r N (r⁻¹ ^ N) hr0 hr1 hq1' hQ]

/-- The remainder is nonnegative and at most `K(r) N² r^(-N)`. -/
theorem varRem_bounds (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) :
    0 ≤ varRem r N ∧
      varRem r N ≤ r ^ 2 * (r + 1) * (r ^ 2 + 2 * r - 1) / (r - 1) ^ 5 * ((N : ℝ) ^ 2 * r⁻¹ ^ N) := by
  have hr0 : 0 < r := by linarith
  have hr1 : 0 < r - 1 := by linarith
  have hQ0 : 0 < r⁻¹ ^ N := by positivity
  have hQ1 : r⁻¹ ^ N ≤ r⁻¹ := by
    calc r⁻¹ ^ N ≤ r⁻¹ ^ 1 :=
          pow_le_pow_of_le_one (by positivity) (inv_lt_one_of_one_lt₀ hr).le hN
      _ = r⁻¹ := pow_one _
  have hgap : r - 1 ≤ r * (1 - r⁻¹ ^ N) := by
    have h1 : r * r⁻¹ = 1 := mul_inv_cancel₀ hr0.ne'
    have h2 : r * r⁻¹ ^ N ≤ r * r⁻¹ := mul_le_mul_of_nonneg_left hQ1 hr0.le
    nlinarith
  have hQlt : r⁻¹ ^ N < 1 := qN_lt_one r hr N hN
  have hNR : (1 : ℝ) ≤ N := by exact_mod_cast hN
  have hS0 : 0 ≤ (N : ℝ) * (r ^ 2 - 1) + 2 * r * (1 - r⁻¹ ^ N) := by
    have h1 : 0 ≤ r ^ 2 - 1 := by nlinarith
    have h2 : 0 ≤ (N : ℝ) * (r ^ 2 - 1) := mul_nonneg (Nat.cast_nonneg N) h1
    have h3 : 0 ≤ 2 * r * (1 - r⁻¹ ^ N) := mul_nonneg (by linarith) (by linarith)
    linarith
  have hQpos : 0 < 1 - r⁻¹ ^ N := by linarith
  have hden : 0 < (1 - r⁻¹ ^ N) ^ 2 * (r - 1) ^ 3 := mul_pos (pow_pos hQpos 2) (pow_pos hr1 3)
  have hfac : 0 ≤ (N : ℝ) * r⁻¹ ^ N * (r + 1) :=
    mul_nonneg (mul_nonneg (Nat.cast_nonneg N) hQ0.le) (by linarith)
  constructor
  · unfold varRem
    exact div_nonneg (mul_nonneg hfac hS0) hden.le
  · have e1 : varRem r N = ((N : ℝ) * r⁻¹ ^ N * (r + 1) / (r - 1) ^ 3)
        * (((N : ℝ) * (r ^ 2 - 1) + 2 * r * (1 - r⁻¹ ^ N)) / (1 - r⁻¹ ^ N) ^ 2) := by
      unfold varRem
      rw [div_mul_div_comm, mul_comm ((r - 1) ^ 3)]
    have e2 : r ^ 2 * (r + 1) * (r ^ 2 + 2 * r - 1) / (r - 1) ^ 5 * ((N : ℝ) ^ 2 * r⁻¹ ^ N)
        = ((N : ℝ) * r⁻¹ ^ N * (r + 1) / (r - 1) ^ 3)
          * ((N : ℝ) * r ^ 2 * (r ^ 2 + 2 * r - 1) / (r - 1) ^ 2) := by
      have hr1' : r - 1 ≠ 0 := hr1.ne'
      field_simp
    rw [e1, e2]
    apply mul_le_mul_of_nonneg_left _ (div_nonneg hfac (pow_nonneg hr1.le 3))
    rw [div_le_div_iff₀ (pow_pos hQpos 2) (pow_pos hr1 2)]
    have hA : (r - 1) ^ 2 ≤ r ^ 2 * (1 - r⁻¹ ^ N) ^ 2 := by
      have h := pow_le_pow_left₀ hr1.le hgap 2
      rw [mul_pow] at h
      exact h
    have hS : (N : ℝ) * (r ^ 2 - 1) + 2 * r * (1 - r⁻¹ ^ N) ≤ (N : ℝ) * (r ^ 2 + 2 * r - 1) := by
      have h1 : 1 - r⁻¹ ^ N ≤ (N : ℝ) := by linarith
      nlinarith [mul_le_mul_of_nonneg_left h1 (by linarith : (0 : ℝ) ≤ 2 * r)]
    have hK : 0 ≤ (N : ℝ) * (r ^ 2 + 2 * r - 1) := by
      have : 0 ≤ r ^ 2 + 2 * r - 1 := by nlinarith
      positivity
    have h1 := mul_le_mul_of_nonneg_right hS (sq_nonneg (r - 1))
    have h2 := mul_le_mul_of_nonneg_left hA hK
    nlinarith

end

end FixLoc
