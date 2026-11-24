import FixLoc.Neutral
import FixLoc.Variance

/-!
# Central limit theorem for the number of head advances (SI Theorem S3), part 1

On the unit circle `z = e^{iθ}`, the generating function of `H` factors through the root
`B(θ) = (1 - s(θ)) / (2β)` of `β B² - B + α e^{iθ} = 0`, with `s(θ)` the principal square root of
`1 - 4αβ e^{iθ}`. `B` is the generating function of the number of head advances in one passage of
the block length to the next level.

This file proves the expansion `B(θ) = 1 + m₁ w - κ w² + O(θ³)` with `w = iθ` (`Bf_expansion`)
without derivatives: a polynomial identity and the exponential series give
`β p² - p + α e^{iθ} = O(θ³)` for the quadratic `p`, and the quadratic equation transfers the bound
to `B - p`.
-/

open Complex Filter Topology
open scoped Real

namespace FixLoc

noncomputable section

/-- Mean number of head advances per level, `m₁ = r/(r-1)`. -/
def mOne (r : ℝ) : ℝ := r / (r - 1)

/-- Variance per level, `v = r(r+1)/(r-1)³`. -/
def vOne (r : ℝ) : ℝ := r * (r + 1) / (r - 1) ^ 3

/-- Second-order coefficient of the root in the variable `w = iθ`. -/
def kap (r : ℝ) : ℝ := -(r * (r ^ 2 + 1)) / (2 * (r - 1) ^ 3)

/-- `c = 4αβ`. -/
def cc (r : ℝ) : ℝ := 4 * alpha r * beta r

/-- `s₀ = (r-1)/(r+1)`, the square root of `1 - c`. -/
def s0 (r : ℝ) : ℝ := (r - 1) / (r + 1)

lemma alpha_pos' (r : ℝ) (hr : 1 < r) : 0 < alpha r := by
  unfold alpha
  exact div_pos (by linarith) (by linarith)

lemma beta_pos' (r : ℝ) (hr : 1 < r) : 0 < beta r := by
  unfold beta
  exact div_pos one_pos (by linarith)

lemma alpha_add_beta (r : ℝ) (hr : 1 < r) : alpha r + beta r = 1 := by
  unfold alpha beta
  have h : (1 : ℝ) + r ≠ 0 := (by linarith : (0 : ℝ) < 1 + r).ne'
  field_simp
  ring

lemma alpha_sub_beta_pos (r : ℝ) (hr : 1 < r) : 0 < alpha r - beta r := by
  unfold alpha beta
  rw [← sub_div]
  exact div_pos (by linarith) (by linarith)

lemma mOne_pos (r : ℝ) (hr : 1 < r) : 0 < mOne r := by
  unfold mOne
  exact div_pos (by linarith) (by linarith)

lemma s0_pos (r : ℝ) (hr : 1 < r) : 0 < s0 r := by
  unfold s0
  exact div_pos (by linarith) (by linarith)

lemma cc_eq (r : ℝ) (hr : 1 < r) : cc r = 1 - s0 r ^ 2 := by
  unfold cc alpha beta s0
  have h1 : r + 1 ≠ 0 := (by linarith : (0 : ℝ) < r + 1).ne'
  have h2 : (1 : ℝ) + r ≠ 0 := (by linarith : (0 : ℝ) < 1 + r).ne'
  field_simp
  try ring

/-- The principal square root of `1 - c e^{iθ}`. -/
def sq (r θ : ℝ) : ℂ := (1 - (cc r : ℂ) * exp (θ * I)) ^ ((2 : ℂ)⁻¹)

lemma sq_sq (r θ : ℝ) : sq r θ ^ 2 = 1 - (cc r : ℂ) * exp (θ * I) := by
  unfold sq
  exact cpow_ofNat_inv_pow _ 2

lemma sq_zero (r : ℝ) (hr : 1 < r) : sq r 0 = (s0 r : ℂ) := by
  have hs0 : 0 ≤ s0 r := (s0_pos r hr).le
  unfold sq
  have h1 : (1 - (cc r : ℂ) * exp (((0 : ℝ) : ℂ) * I)) = ((s0 r ^ 2 : ℝ) : ℂ) := by
    rw [Complex.ofReal_zero, zero_mul, Complex.exp_zero, mul_one, cc_eq r hr]
    push_cast
    ring
  have h2 : ((2 : ℂ)⁻¹) = (((2 : ℝ)⁻¹ : ℝ) : ℂ) := by simp
  rw [h1, h2, ← Complex.ofReal_cpow (by positivity)]
  congr 1
  rw [← one_div, ← Real.sqrt_eq_rpow, Real.sqrt_sq hs0]

lemma sq_continuousAt (r : ℝ) (hr : 1 < r) : ContinuousAt (sq r) 0 := by
  have hmem : (1 - (cc r : ℂ) * exp (((0 : ℝ) : ℂ) * I)) ∈ slitPlane := by
    rw [Complex.ofReal_zero, zero_mul, Complex.exp_zero, mul_one, cc_eq r hr]
    have e : (1 : ℂ) - ((1 - s0 r ^ 2 : ℝ) : ℂ) = ((s0 r ^ 2 : ℝ) : ℂ) := by
      push_cast
      ring
    rw [e]
    exact Complex.ofReal_mem_slitPlane.mpr (pow_pos (s0_pos r hr) 2)
  have hcont : ContinuousAt (fun θ : ℝ => (1 - (cc r : ℂ) * exp ((θ : ℂ) * I))) 0 := by
    fun_prop
  exact ContinuousAt.comp (g := fun x : ℂ => x ^ ((2 : ℂ)⁻¹))
    (f := fun θ : ℝ => (1 - (cc r : ℂ) * exp ((θ : ℂ) * I))) (continuousAt_cpow_const hmem) hcont

/-- The root `B(θ) = (1 - s(θ)) / (2β)`. -/
def Bf (r θ : ℝ) : ℂ := (1 - sq r θ) / (2 * (beta r : ℂ))

lemma Bf_zero (r : ℝ) (hr : 1 < r) : Bf r 0 = 1 := by
  unfold Bf
  rw [sq_zero r hr]
  unfold s0 beta
  have h1 : (r : ℂ) + 1 ≠ 0 := by
    exact_mod_cast (by linarith : (0 : ℝ) < r + 1).ne'
  have h2 : (1 : ℂ) + r ≠ 0 := by
    exact_mod_cast (by linarith : (0 : ℝ) < 1 + r).ne'
  push_cast
  field_simp
  try ring

lemma Bf_continuousAt (r : ℝ) (hr : 1 < r) : ContinuousAt (Bf r) 0 := by
  unfold Bf
  exact (continuousAt_const.sub (sq_continuousAt r hr)).div_const _

/-- `B` solves `β B² - B + α e^{iθ} = 0`. -/
lemma Bf_quad (r θ : ℝ) (hr : 1 < r) :
    (beta r : ℂ) * Bf r θ ^ 2 - Bf r θ + (alpha r : ℂ) * exp (θ * I) = 0 := by
  have hb : (beta r : ℂ) ≠ 0 := by exact_mod_cast (beta_pos' r hr).ne'
  have h := sq_sq r θ
  unfold Bf
  have e1 : (beta r : ℂ) * ((1 - sq r θ) / (2 * beta r)) ^ 2 - (1 - sq r θ) / (2 * beta r)
      = (sq r θ ^ 2 - 1) / (4 * beta r) := by
    field_simp
    try ring
  rw [e1, h]
  unfold cc
  push_cast
  field_simp
  try ring

/-- The quadratic approximation of `B`, in the variable `w = iθ`. -/
def pB (r : ℝ) (w : ℂ) : ℂ := 1 + (mOne r : ℂ) * w - (kap r : ℂ) * w ^ 2

/-- `β p² - p + α (1 + w + w²/2)` has no terms of degree below `3`. -/
lemma poly_identity (r : ℝ) (hr : 1 < r) (w : ℂ) :
    (beta r : ℂ) * pB r w ^ 2 - pB r w + (alpha r : ℂ) * (1 + w + w ^ 2 / 2)
      = (beta r : ℂ) * (((-2 * mOne r * kap r : ℝ) : ℂ) * w ^ 3 + ((kap r ^ 2 : ℝ) : ℂ) * w ^ 4) := by
  have h1 : (r : ℂ) - 1 ≠ 0 := by exact_mod_cast (sub_pos.mpr hr).ne'
  have h2 : (1 : ℂ) + r ≠ 0 := by
    exact_mod_cast (by linarith : (0 : ℝ) < 1 + r).ne'
  unfold pB mOne kap alpha beta
  push_cast
  field_simp
  try ring

lemma norm_ofReal_mul_pow (a : ℝ) (w : ℂ) (k : ℕ) : ‖(a : ℂ) * w ^ k‖ = |a| * ‖w‖ ^ k := by
  rw [norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs]

lemma norm_ofReal_mul_I (θ : ℝ) : ‖(θ : ℂ) * I‖ = |θ| := by
  rw [norm_mul, Complex.norm_real, Complex.norm_I, mul_one, Real.norm_eq_abs]

/-- The exponential series to third order on the imaginary axis. -/
lemma exp_taylor3 (θ : ℝ) (hθ : |θ| ≤ 1) :
    ‖exp ((θ : ℂ) * I) - (1 + θ * I + (θ * I) ^ 2 / 2)‖ ≤ |θ| ^ 3 := by
  have hw := norm_ofReal_mul_I θ
  have hb := Complex.exp_bound (x := (θ : ℂ) * I) (by rw [hw]; exact hθ) (n := 3) (by norm_num)
  have hsum : ∑ m ∈ Finset.range 3, ((θ : ℂ) * I) ^ m / (m.factorial : ℂ)
      = 1 + θ * I + (θ * I) ^ 2 / 2 := by
    simp [Finset.sum_range_succ, Nat.factorial]
  rw [hsum, hw] at hb
  refine hb.trans ?_
  apply mul_le_of_le_one_right (by positivity)
  norm_num [Nat.factorial]

/-- `β p² - p + α e^{iθ} = O(θ³)`. -/
lemma F_bound (r : ℝ) (hr : 1 < r) (θ : ℝ) (hθ : |θ| ≤ 1) :
    ‖(beta r : ℂ) * pB r (θ * I) ^ 2 - pB r (θ * I) + (alpha r : ℂ) * exp (θ * I)‖
      ≤ (beta r * (|(-2 * mOne r * kap r)| + kap r ^ 2) + alpha r) * |θ| ^ 3 := by
  have hw := norm_ofReal_mul_I θ
  have ha := alpha_pos' r hr
  have hb := beta_pos' r hr
  have e : (beta r : ℂ) * pB r (θ * I) ^ 2 - pB r (θ * I) + (alpha r : ℂ) * exp (θ * I)
      = (beta r : ℂ) * (((-2 * mOne r * kap r : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 3
          + ((kap r ^ 2 : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 4)
        + (alpha r : ℂ) * (exp ((θ : ℂ) * I) - (1 + θ * I + (θ * I) ^ 2 / 2)) := by
    rw [← poly_identity r hr ((θ : ℂ) * I)]
    ring
  rw [e]
  have h3 : |θ| ^ 4 ≤ |θ| ^ 3 := by
    have h0 : 0 ≤ |θ| := abs_nonneg θ
    calc |θ| ^ 4 = |θ| ^ 3 * |θ| := by ring
      _ ≤ |θ| ^ 3 * 1 := mul_le_mul_of_nonneg_left hθ (by positivity)
      _ = |θ| ^ 3 := by ring
  have hpoly : ‖((-2 * mOne r * kap r : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 3
      + ((kap r ^ 2 : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 4‖
      ≤ (|(-2 * mOne r * kap r)| + kap r ^ 2) * |θ| ^ 3 := by
    calc _ ≤ ‖((-2 * mOne r * kap r : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 3‖
          + ‖((kap r ^ 2 : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 4‖ := norm_add_le _ _
      _ = |(-2 * mOne r * kap r)| * |θ| ^ 3 + |kap r ^ 2| * |θ| ^ 4 := by
          rw [norm_ofReal_mul_pow, norm_ofReal_mul_pow, hw]
      _ ≤ |(-2 * mOne r * kap r)| * |θ| ^ 3 + kap r ^ 2 * |θ| ^ 3 := by
          rw [abs_of_nonneg (sq_nonneg (kap r))]
          have := mul_le_mul_of_nonneg_left h3 (sq_nonneg (kap r))
          linarith
      _ = (|(-2 * mOne r * kap r)| + kap r ^ 2) * |θ| ^ 3 := by ring
  have hexp := exp_taylor3 θ hθ
  calc _ ≤ ‖(beta r : ℂ) * (((-2 * mOne r * kap r : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 3
          + ((kap r ^ 2 : ℝ) : ℂ) * ((θ : ℂ) * I) ^ 4)‖
        + ‖(alpha r : ℂ) * (exp ((θ : ℂ) * I) - (1 + θ * I + (θ * I) ^ 2 / 2))‖ := norm_add_le _ _
    _ ≤ beta r * ((|(-2 * mOne r * kap r)| + kap r ^ 2) * |θ| ^ 3) + alpha r * |θ| ^ 3 := by
        rw [norm_mul, norm_mul, Complex.norm_real, Complex.norm_real, Real.norm_eq_abs,
          Real.norm_eq_abs, abs_of_pos hb, abs_of_pos ha]
        gcongr
    _ = (beta r * (|(-2 * mOne r * kap r)| + kap r ^ 2) + alpha r) * |θ| ^ 3 := by ring

/-- SI Theorem S3, expansion of the root: `B(θ) = p(iθ) + O(θ³)` near `θ = 0`. -/
theorem Bf_expansion (r : ℝ) (hr : 1 < r) :
    ∃ δ > 0, ∃ C : ℝ, ∀ θ : ℝ, |θ| ≤ δ → ‖Bf r θ - pB r (θ * I)‖ ≤ C * |θ| ^ 3 := by
  have ha := alpha_pos' r hr
  have hb := beta_pos' r hr
  have hb' : beta r ≠ 0 := hb.ne'
  have hab := alpha_sub_beta_pos r hr
  have hab' : alpha r - beta r ≠ 0 := hab.ne'
  have hsum := alpha_add_beta r hr
  have hm := mOne_pos r hr
  set CF := beta r * (|(-2 * mOne r * kap r)| + kap r ^ 2) + alpha r with hCF
  -- continuity of the error at `0`
  have hcont : ContinuousAt (fun θ : ℝ => Bf r θ - pB r ((θ : ℂ) * I)) 0 := by
    have h2 : ContinuousAt (fun θ : ℝ => pB r ((θ : ℂ) * I)) 0 := by
      unfold pB
      fun_prop
    exact (Bf_continuousAt r hr).sub h2
  have he0 : Bf r 0 - pB r (((0 : ℝ) : ℂ) * I) = 0 := by
    rw [Bf_zero r hr]
    unfold pB
    simp
  set ε := (alpha r - beta r) / (4 * beta r) with hε
  have hε0 : 0 < ε := by positivity
  obtain ⟨δ₂, hδ₂, hδ₂'⟩ := Metric.continuousAt_iff.mp hcont ε hε0
  set K := mOne r + |kap r| + 1 with hKdef
  have hK : 0 < K := by positivity
  have hK' : K ≠ 0 := hK.ne'
  set δ₁ := min 1 ((alpha r - beta r) / (8 * beta r * K)) with hδ₁
  have hδ₁0 : 0 < δ₁ := by positivity
  refine ⟨min δ₁ (δ₂ / 2), by positivity, 2 * CF / (alpha r - beta r), ?_⟩
  intro θ hθ
  have hθ1 : |θ| ≤ δ₁ := hθ.trans (min_le_left _ _)
  have hθone : |θ| ≤ 1 := hθ1.trans (min_le_left _ _)
  have hθsmall : |θ| ≤ (alpha r - beta r) / (8 * beta r * K) := hθ1.trans (min_le_right _ _)
  have hθ2 : |θ| < δ₂ := by
    have := hθ.trans (min_le_right _ _)
    linarith
  have hq := Bf_quad r θ hr
  have hF := F_bound r hr θ hθone
  have hwn := norm_ofReal_mul_I θ
  have heps := hδ₂' (show dist θ 0 < δ₂ by rw [Real.dist_eq, sub_zero]; exact hθ2)
  rw [dist_eq_norm, he0, sub_zero] at heps
  set w : ℂ := (θ : ℂ) * I with hwdef
  set e := Bf r θ - pB r w with hedef
  -- the quadratic equation for the error
  have hrel : e * (2 * (beta r : ℂ) * pB r w - 1 + beta r * e)
      = -((beta r : ℂ) * pB r w ^ 2 - pB r w + (alpha r : ℂ) * exp w) := by
    have hB : Bf r θ = pB r w + e := by rw [hedef]; ring
    rw [hB] at hq
    linear_combination hq
  -- the linear part of the factor is small
  set Y := (2 * (beta r : ℂ)) * (((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2) with hY
  have hlin : ‖Y‖ ≤ (alpha r - beta r) / 4 := by
    have h1 : ‖((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2‖ ≤ K * |θ| := by
      calc _ ≤ ‖((mOne r : ℝ) : ℂ) * w‖ + ‖((kap r : ℝ) : ℂ) * w ^ 2‖ := norm_sub_le _ _
        _ = |mOne r| * |θ| + |kap r| * |θ| ^ 2 := by
            rw [norm_ofReal_mul_pow (kap r) w 2, hwn, norm_mul, Complex.norm_real,
              Real.norm_eq_abs, hwn]
        _ ≤ K * |θ| := by
            rw [abs_of_pos hm, hKdef]
            have h0 : 0 ≤ |θ| := abs_nonneg θ
            have hk : 0 ≤ |kap r| := abs_nonneg _
            have hθsq : |θ| ^ 2 ≤ |θ| := by nlinarith
            have := mul_le_mul_of_nonneg_left hθsq hk
            nlinarith
    have h2 : ‖(2 * (beta r : ℂ))‖ = 2 * beta r := by
      rw [norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hb]
      norm_num
    rw [hY, norm_mul, h2]
    calc 2 * beta r * ‖((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2‖
        ≤ 2 * beta r * (K * |θ|) := by gcongr
      _ ≤ 2 * beta r * (K * ((alpha r - beta r) / (8 * beta r * K))) := by gcongr
      _ = (alpha r - beta r) / 4 := by
          field_simp
          try ring
  set Z := (beta r : ℂ) * e with hZ
  have hbe : ‖Z‖ ≤ (alpha r - beta r) / 4 := by
    rw [hZ, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hb]
    calc beta r * ‖e‖ ≤ beta r * ε := by gcongr
      _ = (alpha r - beta r) / 4 := by
          rw [hε]
          field_simp
          try ring
  set X := -((alpha r - beta r : ℝ) : ℂ) with hX
  have hconst : ‖X‖ = alpha r - beta r := by
    rw [hX, norm_neg, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hab]
  have hsplit : 2 * (beta r : ℂ) * pB r w - 1 + beta r * e = X + Y + Z := by
    have hs : ((alpha r : ℝ) : ℂ) + (beta r : ℂ) = 1 := by exact_mod_cast hsum
    rw [hX, hY, hZ]
    unfold pB
    push_cast
    linear_combination hs
  have tri : ‖X‖ ≤ ‖X + Y + Z‖ + ‖Y‖ + ‖Z‖ := by
    calc ‖X‖ = ‖(X + Y + Z) - Y - Z‖ := by congr 1; ring
      _ ≤ ‖(X + Y + Z) - Y‖ + ‖Z‖ := norm_sub_le _ _
      _ ≤ ‖X + Y + Z‖ + ‖Y‖ + ‖Z‖ := by
          have := norm_sub_le (X + Y + Z) Y
          linarith
  have hfac : (alpha r - beta r) / 2 ≤ ‖2 * (beta r : ℂ) * pB r w - 1 + beta r * e‖ := by
    rw [hsplit]
    rw [hconst] at tri
    linarith
  -- conclude
  have hprod : ‖e‖ * ((alpha r - beta r) / 2) ≤ CF * |θ| ^ 3 := by
    calc ‖e‖ * ((alpha r - beta r) / 2)
        ≤ ‖e‖ * ‖2 * (beta r : ℂ) * pB r w - 1 + beta r * e‖ :=
          mul_le_mul_of_nonneg_left hfac (norm_nonneg e)
      _ = ‖e * (2 * (beta r : ℂ) * pB r w - 1 + beta r * e)‖ := (norm_mul _ _).symm
      _ = ‖(beta r : ℂ) * pB r w ^ 2 - pB r w + (alpha r : ℂ) * exp w‖ := by
          rw [hrel, norm_neg]
      _ ≤ CF * |θ| ^ 3 := hF
  have hpos : 0 < (alpha r - beta r) / 2 := by positivity
  calc ‖Bf r θ - pB r w‖ = ‖e‖ := rfl
    _ ≤ CF * |θ| ^ 3 / ((alpha r - beta r) / 2) := by
        rw [le_div_iff₀ hpos]
        exact hprod
    _ = 2 * CF / (alpha r - beta r) * |θ| ^ 3 := by
        field_simp
        try ring


/-! ## The logarithm of the root -/

/-- The quadratic approximation of `log B`: `m₁ w + (v/2) w²` with `w = iθ`. -/
def qL (r : ℝ) (w : ℂ) : ℂ := ((mOne r : ℝ) : ℂ) * w + ((vOne r / 2 : ℝ) : ℂ) * w ^ 2

lemma kap_rel (r : ℝ) (hr : 1 < r) : kap r = -(mOne r ^ 2 / 2) - vOne r / 2 := by
  unfold kap mOne vOne
  have h1 : r - 1 ≠ 0 := (sub_pos.mpr hr).ne'
  field_simp
  try ring

lemma logTaylor_three (z : ℂ) : Complex.logTaylor 3 z = z - z ^ 2 / 2 := by
  simp [Complex.logTaylor, Finset.sum_range_succ]
  try ring

/-- The logarithm against its second Taylor polynomial. -/
lemma log_taylor2_bound (b : ℂ) (M t : ℝ) (hbM : ‖b‖ ≤ M * t) (hhalf : ‖b‖ ≤ 1 / 2)
    (hM : 0 ≤ M) (ht : 0 ≤ t) :
    ‖log (1 + b) - (b - b ^ 2 / 2)‖ ≤ 2 / 3 * M ^ 3 * t ^ 3 := by
  have hlt : ‖b‖ < 1 := by linarith
  have hlog := Complex.norm_log_sub_logTaylor_le 2 hlt
  rw [show (2 : ℕ) + 1 = 3 from rfl, logTaylor_three] at hlog
  refine hlog.trans ?_
  have hb0 : 0 ≤ ‖b‖ := norm_nonneg _
  have hinv : (1 - ‖b‖)⁻¹ ≤ 2 := by
    rw [inv_le_comm₀ (by linarith) (by norm_num)]
    linarith
  have hb3 : ‖b‖ ^ 3 ≤ (M * t) ^ 3 := pow_le_pow_left₀ hb0 hbM 3
  have hpos : 0 ≤ (1 - ‖b‖)⁻¹ := inv_nonneg.mpr (by linarith)
  calc ‖b‖ ^ 3 * (1 - ‖b‖)⁻¹ / ((2 : ℕ) + 1 : ℝ)
      = ‖b‖ ^ 3 * (1 - ‖b‖)⁻¹ / 3 := by norm_num
    _ ≤ (M * t) ^ 3 * 2 / 3 := by
        apply div_le_div_of_nonneg_right _ (by norm_num)
        exact mul_le_mul hb3 hinv hpos (by positivity)
    _ = 2 / 3 * M ^ 3 * t ^ 3 := by ring

/-- The algebraic remainder of the logarithm expansion. -/
lemma log_rem_bound (e w : ℂ) (m κ C₀ t : ℝ) (hm : 0 < m) (hwn : ‖w‖ = t) (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) (he : ‖e‖ ≤ |C₀| * t ^ 3) :
    ‖e - (κ : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2 + (m : ℂ) * (κ : ℂ) * w ^ 3
        - (m : ℂ) * w * e + (κ : ℂ) * w ^ 2 * e‖
      ≤ (|C₀| + κ ^ 2 / 2 + C₀ ^ 2 / 2 + m * |κ| + m * |C₀| + |κ| * |C₀|) * t ^ 3 := by
  have n1 : ‖(κ : ℂ) ^ 2 * w ^ 4 / 2‖ = κ ^ 2 / 2 * t ^ 4 := by
    rw [norm_div, norm_mul, norm_pow, norm_pow, Complex.norm_real, Real.norm_eq_abs, hwn, sq_abs]
    norm_num
    try ring
  have n2 : ‖e ^ 2 / 2‖ = ‖e‖ ^ 2 / 2 := by
    rw [norm_div, norm_pow]
    norm_num
  have n3 : ‖(m : ℂ) * (κ : ℂ) * w ^ 3‖ = m * |κ| * t ^ 3 := by
    rw [norm_mul, norm_mul, norm_pow, Complex.norm_real, Complex.norm_real, Real.norm_eq_abs,
      Real.norm_eq_abs, abs_of_pos hm, hwn]
  have n4 : ‖(m : ℂ) * w * e‖ = m * t * ‖e‖ := by
    rw [norm_mul, norm_mul, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hm, hwn]
  have n5 : ‖(κ : ℂ) * w ^ 2 * e‖ = |κ| * t ^ 2 * ‖e‖ := by
    rw [norm_mul, norm_mul, norm_pow, Complex.norm_real, Real.norm_eq_abs, hwn]
  have hen : 0 ≤ ‖e‖ := norm_nonneg e
  have hk0 : 0 ≤ |κ| := abs_nonneg _
  have hc0 : 0 ≤ |C₀| := abs_nonneg _
  have ht3 : 0 ≤ t ^ 3 := by positivity
  have h4 : t ^ 4 ≤ t ^ 3 := pow_le_pow_of_le_one ht0 ht1 (by norm_num)
  have h6 : t ^ 6 ≤ t ^ 3 := pow_le_pow_of_le_one ht0 ht1 (by norm_num)
  have ht2 : t ^ 2 ≤ 1 := pow_le_one₀ ht0 ht1
  have he2 : ‖e‖ ^ 2 ≤ C₀ ^ 2 * t ^ 3 := by
    have h1 : ‖e‖ ^ 2 ≤ (|C₀| * t ^ 3) ^ 2 := pow_le_pow_left₀ hen he 2
    have h2 : (|C₀| * t ^ 3) ^ 2 = C₀ ^ 2 * t ^ 6 := by rw [mul_pow, sq_abs]; ring
    have h3 : C₀ ^ 2 * t ^ 6 ≤ C₀ ^ 2 * t ^ 3 := mul_le_mul_of_nonneg_left h6 (sq_nonneg C₀)
    linarith only [h1, h2, h3]
  have m4 : m * t * ‖e‖ ≤ m * |C₀| * t ^ 3 := by
    have h1 : t * ‖e‖ ≤ 1 * (|C₀| * t ^ 3) := mul_le_mul ht1 he hen (by norm_num)
    have h2 := mul_le_mul_of_nonneg_left h1 hm.le
    linarith only [h2]
  have m5 : |κ| * t ^ 2 * ‖e‖ ≤ |κ| * |C₀| * t ^ 3 := by
    have h1 : t ^ 2 * ‖e‖ ≤ 1 * (|C₀| * t ^ 3) := mul_le_mul ht2 he hen (by norm_num)
    have h2 := mul_le_mul_of_nonneg_left h1 hk0
    linarith only [h2]
  have m1 : κ ^ 2 / 2 * t ^ 4 ≤ κ ^ 2 / 2 * t ^ 3 :=
    mul_le_mul_of_nonneg_left h4 (by positivity)
  have t1 := norm_add_le (e - (κ : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2 + (m : ℂ) * (κ : ℂ) * w ^ 3
    - (m : ℂ) * w * e) ((κ : ℂ) * w ^ 2 * e)
  have t2 := norm_sub_le (e - (κ : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2 + (m : ℂ) * (κ : ℂ) * w ^ 3)
    ((m : ℂ) * w * e)
  have t3 := norm_add_le (e - (κ : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2) ((m : ℂ) * (κ : ℂ) * w ^ 3)
  have t4 := norm_sub_le (e - (κ : ℂ) ^ 2 * w ^ 4 / 2) (e ^ 2 / 2)
  have t5 := norm_sub_le e ((κ : ℂ) ^ 2 * w ^ 4 / 2)
  rw [n1] at t5
  rw [n2] at t4
  rw [n3] at t3
  rw [n4] at t2
  rw [n5] at t1
  linarith only [t1, t2, t3, t4, t5, he, he2, m1, m4, m5]

/-- SI Theorem S3, expansion of the logarithm of the root:
`log B(θ) = m₁ (iθ) + (v/2)(iθ)² + O(θ³)` near `θ = 0`. -/
theorem logBf_expansion (r : ℝ) (hr : 1 < r) :
    ∃ δ > 0, ∃ C : ℝ, ∀ θ : ℝ, |θ| ≤ δ →
      ‖Bf r θ - 1‖ ≤ 1 / 2 ∧ ‖log (Bf r θ) - qL r (θ * I)‖ ≤ C * |θ| ^ 3 := by
  obtain ⟨δ₀, hδ₀, C₀, hC₀⟩ := Bf_expansion r hr
  have hm := mOne_pos r hr
  set M := mOne r + |kap r| + |C₀| + 1 with hM
  have hMpos : 0 < M := by positivity
  set C₂ := |C₀| + kap r ^ 2 / 2 + C₀ ^ 2 / 2 + mOne r * |kap r| + mOne r * |C₀| + |kap r| * |C₀|
    with hC₂
  refine ⟨min (min δ₀ 1) (1 / (2 * M)), by positivity, 2 / 3 * M ^ 3 + C₂, ?_⟩
  intro θ hθ
  have hθ0 : |θ| ≤ δ₀ := hθ.trans ((min_le_left _ _).trans (min_le_left _ _))
  have hθ1 : |θ| ≤ 1 := hθ.trans ((min_le_left _ _).trans (min_le_right _ _))
  have hθM : |θ| ≤ 1 / (2 * M) := hθ.trans (min_le_right _ _)
  have ht0 : 0 ≤ |θ| := abs_nonneg θ
  have he := hC₀ θ hθ0
  have hwn := norm_ofReal_mul_I θ
  have hk := kap_rel r hr
  set w : ℂ := (θ : ℂ) * I with hwdef
  set e := Bf r θ - pB r w with hedef
  have he' : ‖e‖ ≤ |C₀| * |θ| ^ 3 := by
    have h3 : 0 ≤ |θ| ^ 3 := by positivity
    have := mul_le_mul_of_nonneg_right (le_abs_self C₀) h3
    linarith only [he, this]
  have hb_eq : Bf r θ - 1 = (((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2) + e := by
    rw [hedef]
    unfold pB
    ring
  -- size of `B - 1`
  have hlinn : ‖((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2‖
      ≤ mOne r * |θ| + |kap r| * |θ| ^ 2 := by
    calc _ ≤ ‖((mOne r : ℝ) : ℂ) * w‖ + ‖((kap r : ℝ) : ℂ) * w ^ 2‖ := norm_sub_le _ _
      _ = mOne r * |θ| + |kap r| * |θ| ^ 2 := by
          rw [norm_ofReal_mul_pow (kap r) w 2, hwn, norm_mul, Complex.norm_real,
            Real.norm_eq_abs, hwn, abs_of_pos hm]
  have hbM : ‖Bf r θ - 1‖ ≤ M * |θ| := by
    rw [hb_eq]
    have h2 : |θ| ^ 2 ≤ |θ| := by
      have := pow_le_pow_of_le_one ht0 hθ1 (show 1 ≤ 2 by norm_num)
      simpa using this
    have h3 : |θ| ^ 3 ≤ |θ| := by
      have := pow_le_pow_of_le_one ht0 hθ1 (show 1 ≤ 3 by norm_num)
      simpa using this
    have hk0 : 0 ≤ |kap r| := abs_nonneg _
    have hc0 : 0 ≤ |C₀| := abs_nonneg _
    have m2 := mul_le_mul_of_nonneg_left h2 hk0
    have m3 := mul_le_mul_of_nonneg_left h3 hc0
    have tri := norm_add_le (((mOne r : ℝ) : ℂ) * w - ((kap r : ℝ) : ℂ) * w ^ 2) e
    have hMe : M * |θ| = mOne r * |θ| + |kap r| * |θ| + |C₀| * |θ| + |θ| := by rw [hM]; ring
    linarith only [tri, hlinn, he', m2, m3, hMe, ht0]
  have hbhalf : ‖Bf r θ - 1‖ ≤ 1 / 2 := by
    calc ‖Bf r θ - 1‖ ≤ M * |θ| := hbM
      _ ≤ M * (1 / (2 * M)) := by gcongr
      _ = 1 / 2 := by field_simp
  refine ⟨hbhalf, ?_⟩
  have hT1 := log_taylor2_bound (Bf r θ - 1) M |θ| hbM hbhalf hMpos.le ht0
  rw [show (1 : ℂ) + (Bf r θ - 1) = Bf r θ by ring] at hT1
  have hkc : ((kap r : ℝ) : ℂ) = -(((mOne r : ℝ) : ℂ) ^ 2 / 2) - ((vOne r : ℝ) : ℂ) / 2 := by
    rw [hk]
    push_cast
    ring
  have key : log (Bf r θ) - qL r w
      = (log (Bf r θ) - ((Bf r θ - 1) - (Bf r θ - 1) ^ 2 / 2))
        + (e - ((kap r : ℝ) : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2
          + ((mOne r : ℝ) : ℂ) * ((kap r : ℝ) : ℂ) * w ^ 3
          - ((mOne r : ℝ) : ℂ) * w * e + ((kap r : ℝ) : ℂ) * w ^ 2 * e) := by
    rw [hb_eq]
    unfold qL
    push_cast
    linear_combination (-w ^ 2) * hkc
  have hT2 := log_rem_bound e w (mOne r) (kap r) C₀ |θ| hm hwn ht0 hθ1 he'
  rw [key]
  calc _ ≤ ‖log (Bf r θ) - ((Bf r θ - 1) - (Bf r θ - 1) ^ 2 / 2)‖
        + ‖e - ((kap r : ℝ) : ℂ) ^ 2 * w ^ 4 / 2 - e ^ 2 / 2
          + ((mOne r : ℝ) : ℂ) * ((kap r : ℝ) : ℂ) * w ^ 3
          - ((mOne r : ℝ) : ℂ) * w * e + ((kap r : ℝ) : ℂ) * w ^ 2 * e‖ := norm_add_le _ _
    _ ≤ 2 / 3 * M ^ 3 * |θ| ^ 3 + C₂ * |θ| ^ 3 := add_le_add hT1 hT2
    _ = (2 / 3 * M ^ 3 + C₂) * |θ| ^ 3 := by ring

end

end FixLoc
