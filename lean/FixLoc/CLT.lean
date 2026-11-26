import FixLoc.CLTExpansion

/-!
# Central limit theorem for the number of head advances (SI Theorem S3), part 2

The generating function of `H` at `z = e^{iθ}` equals
`(A(θ)/A(0)) · (1 - R(0)^N)/(1 - R(θ)^N) · B(θ)^N` (`gf_factor`, `rhoH_eq`). With
`θ = τ/σ_N`, the first two factors tend to `1`, and `N log B(τ/σ_N) - iτμ_N/σ_N → -τ²/2` by the
expansion of `log B` and the exact mean and variance. Hence the characteristic function of
`(H - μ_N)/σ_N` tends to `e^{-τ²/2}` (`clt_charFun`).
-/

open Complex Filter Topology
open scoped Real

namespace FixLoc

noncomputable section

/-- `A(θ) = s(θ) / (α e^{iθ})`. -/
def Af (r θ : ℝ) : ℂ := sq r θ / ((alpha r : ℂ) * exp (θ * I))

/-- `R(θ) = (1 - s(θ)) / (1 + s(θ))`. -/
def Rf (r θ : ℝ) : ℂ := (1 - sq r θ) / (1 + sq r θ)

/-- The generating function through the root: for `N = M + 1`,
`(2αz)^M / P_N(1 - 4αβz) = A B^N / (1 - R^N)`. -/
lemma gf_factor (r θ : ℝ) (hr : 1 < r) (M : ℕ) (hs : sq r θ ≠ 0) (hs1 : 1 + sq r θ ≠ 0)
    (hR : 1 - Rf r θ ^ (M + 1) ≠ 0) :
    (2 * (alpha r : ℂ) * exp (θ * I)) ^ M
        / PNc (M + 1) (1 - 4 * (alpha r : ℂ) * beta r * exp (θ * I))
      = Af r θ * Bf r θ ^ (M + 1) / (1 - Rf r θ ^ (M + 1)) := by
  have ha : (alpha r : ℂ) ≠ 0 := by exact_mod_cast (alpha_pos' r hr).ne'
  have hb : (beta r : ℂ) ≠ 0 := by exact_mod_cast (beta_pos' r hr).ne'
  have hz : exp ((θ : ℂ) * I) ≠ 0 := Complex.exp_ne_zero _
  have hs2 : sq r θ ^ 2 = 1 - 4 * (alpha r : ℂ) * beta r * exp ((θ : ℂ) * I) := by
    rw [sq_sq]
    unfold cc
    push_cast
    ring
  unfold Af Bf
  unfold Rf at hR ⊢
  set s := sq r θ with hsdef
  set z := exp ((θ : ℂ) * I) with hzdef
  have hprod : (1 - s) * (1 + s) = 4 * (alpha r : ℂ) * beta r * z := by
    linear_combination -hs2
  have hs1' : 1 - s ≠ 0 := by
    intro h
    have h0 : (4 : ℂ) * (alpha r : ℂ) * beta r * z = 0 := by rw [← hprod, h, zero_mul]
    exact (mul_ne_zero (mul_ne_zero (mul_ne_zero (by norm_num) ha) hb) hz) h0
  have hP : PNc (M + 1) (1 - 4 * (alpha r : ℂ) * beta r * z)
      = ((1 + s) ^ (M + 1) - (1 - s) ^ (M + 1)) / (2 * s) := by
    rw [← hs2, eq_div_iff (mul_ne_zero two_ne_zero hs)]
    have := two_mul_PNc_sq s (M + 1)
    linear_combination this
  have hpow : (1 + s) ^ (M + 1) ≠ 0 := pow_ne_zero _ hs1
  have hD : (1 + s) ^ (M + 1) - (1 - s) ^ (M + 1) ≠ 0 := by
    intro h
    apply hR
    rw [div_pow, sub_eq_zero, eq_comm, div_eq_one_iff_eq hpow]
    linear_combination -h
  have haz : (alpha r : ℂ) * z = (1 - s) * (1 + s) / (4 * beta r) := by
    rw [hprod]
    field_simp
    try ring
  have h2az : 2 * (alpha r : ℂ) * z = (1 - s) * (1 + s) / (2 * beta r) := by
    rw [mul_assoc, haz]
    field_simp
    try ring
  rw [hP, h2az, haz]
  simp only [div_pow, mul_pow]
  field_simp
  ring

/-- Values at `θ = 0`. -/
lemma Af_zero (r : ℝ) (hr : 1 < r) : Af r 0 = ((1 - r⁻¹ : ℝ) : ℂ) := by
  unfold Af
  rw [sq_zero r hr, Complex.ofReal_zero, zero_mul, Complex.exp_zero, mul_one]
  unfold s0 alpha
  have h1 : (r : ℂ) + 1 ≠ 0 := by exact_mod_cast (by linarith : (0 : ℝ) < r + 1).ne'
  have h2 : (1 : ℂ) + r ≠ 0 := by exact_mod_cast (by linarith : (0 : ℝ) < 1 + r).ne'
  have h3 : (r : ℂ) ≠ 0 := by exact_mod_cast (by linarith : (0 : ℝ) < r).ne'
  push_cast
  field_simp
  ring

lemma Rf_zero (r : ℝ) (hr : 1 < r) : Rf r 0 = ((r⁻¹ : ℝ) : ℂ) := by
  unfold Rf
  rw [sq_zero r hr]
  unfold s0
  have h1 : (r : ℂ) + 1 ≠ 0 := by exact_mod_cast (by linarith : (0 : ℝ) < r + 1).ne'
  have h3 : (r : ℂ) ≠ 0 := by exact_mod_cast (by linarith : (0 : ℝ) < r).ne'
  push_cast
  field_simp
  ring_nf
  simp [h3]

/-- `ρ = A(0) / (1 - R(0)^N)`. -/
def rhoH (r : ℝ) (N : ℕ) : ℝ := (1 - r⁻¹) / (1 - r⁻¹ ^ N)

lemma rhoH_eq (r : ℝ) (hr : 1 < r) (N : ℕ) :
    (rhoH r N : ℂ) = Af r 0 / (1 - Rf r 0 ^ N) := by
  rw [Af_zero r hr, Rf_zero r hr]
  unfold rhoH
  push_cast
  try ring

lemma Af_continuousAt (r : ℝ) (hr : 1 < r) : ContinuousAt (Af r) 0 := by
  unfold Af
  have ha : (alpha r : ℂ) * exp (((0 : ℝ) : ℂ) * I) ≠ 0 := by
    rw [Complex.ofReal_zero, zero_mul, Complex.exp_zero, mul_one]
    exact_mod_cast (alpha_pos' r hr).ne'
  exact (sq_continuousAt r hr).div (by fun_prop) ha

lemma Rf_continuousAt (r : ℝ) (hr : 1 < r) : ContinuousAt (Rf r) 0 := by
  unfold Rf
  have h : 1 + sq r 0 ≠ 0 := by
    rw [sq_zero r hr]
    exact_mod_cast (by linarith [s0_pos r hr] : (0 : ℝ) < 1 + s0 r).ne'
  exact (continuousAt_const.sub (sq_continuousAt r hr)).div
    (continuousAt_const.add (sq_continuousAt r hr)) h

/-- `E[H | fixation]` (SI Theorem S3, exact mean). -/
def meanH (r : ℝ) (N : ℕ) : ℝ :=
  ((N : ℝ) - 1) * r / (r - 1) - (r + 1) / (r - 1) ^ 2 + (N : ℝ) * (r + 1) / ((r - 1) * (r ^ N - 1))

/-- `Var[H | fixation]` (SI Theorem S3, exact variance). -/
def varH (r : ℝ) (N : ℕ) : ℝ :=
  r * (r + 1) / (r - 1) ^ 3 * ((N : ℝ) - 2 * (r + 1) / (r - 1)) + varRem r N


lemma vOne_pos (r : ℝ) (hr : 1 < r) : 0 < vOne r := by
  unfold vOne
  exact div_pos (mul_pos (by linarith) (by linarith)) (pow_pos (by linarith) 3)

lemma varH_ge (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) :
    vOne r * ((N : ℝ) - 2 * (r + 1) / (r - 1)) ≤ varH r N := by
  have := (varRem_bounds r hr N hN).1
  unfold varH vOne
  linarith

lemma varH_tendsto (r : ℝ) (hr : 1 < r) : Tendsto (fun N : ℕ => varH r N) atTop atTop := by
  have hv := vOne_pos r hr
  have h0 : Tendsto (fun N : ℕ => (N : ℝ) + -(2 * (r + 1) / (r - 1))) atTop atTop :=
    tendsto_atTop_add_const_right atTop _ tendsto_natCast_atTop_atTop
  have h1 : Tendsto (fun N : ℕ => vOne r * ((N : ℝ) - 2 * (r + 1) / (r - 1))) atTop atTop := by
    apply Tendsto.const_mul_atTop hv
    exact h0.congr (fun N => by ring)
  exact tendsto_atTop_mono' atTop ((eventually_ge_atTop 1).mono fun N hN => varH_ge r hr N hN) h1

lemma varRem_div_tendsto (r : ℝ) (hr : 1 < r) :
    Tendsto (fun N : ℕ => varRem r N / N) atTop (𝓝 0) := by
  set K := r ^ 2 * (r + 1) * (r ^ 2 + 2 * r - 1) / (r - 1) ^ 5 with hK
  have hq : |r⁻¹| < 1 := by
    rw [abs_of_pos (inv_pos.mpr (by linarith))]
    exact inv_lt_one_of_one_lt₀ hr
  have h0 := (tendsto_pow_const_mul_const_pow_of_abs_lt_one 1 hq).const_mul K
  rw [mul_zero] at h0
  apply squeeze_zero' _ _ h0
  · filter_upwards [eventually_ge_atTop 1] with N hN
    exact div_nonneg (varRem_bounds r hr N hN).1 (Nat.cast_nonneg N)
  · filter_upwards [eventually_ge_atTop 1] with N hN
    have hb := (varRem_bounds r hr N hN).2
    have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
    rw [div_le_iff₀ hNpos]
    calc varRem r N ≤ K * ((N : ℝ) ^ 2 * r⁻¹ ^ N) := hb
      _ = K * ((N : ℝ) ^ 1 * r⁻¹ ^ N) * N := by ring

lemma N_div_varH_tendsto (r : ℝ) (hr : 1 < r) :
    Tendsto (fun N : ℕ => (N : ℝ) / varH r N) atTop (𝓝 (vOne r)⁻¹) := by
  have hv := vOne_pos r hr
  have hr1 : r - 1 ≠ 0 := (sub_pos.mpr hr).ne'
  have h2 : Tendsto (fun N : ℕ => vOne r - vOne r * (2 * (r + 1) / (r - 1)) * (1 / (N : ℝ))
      + varRem r N / N) atTop (𝓝 (vOne r - vOne r * (2 * (r + 1) / (r - 1)) * 0 + 0)) :=
    (tendsto_const_nhds.sub (tendsto_one_div_atTop_nhds_zero_nat.const_mul _)).add
      (varRem_div_tendsto r hr)
  rw [mul_zero, sub_zero, add_zero] at h2
  have h1 : Tendsto (fun N : ℕ => varH r N / N) atTop (𝓝 (vOne r)) := by
    apply h2.congr'
    filter_upwards [eventually_ge_atTop 1] with N hN
    have hNpos : (N : ℝ) ≠ 0 := by exact_mod_cast (by omega : N ≠ 0)
    unfold varH vOne
    field_simp
    try ring
  have h3 := h1.inv₀ hv.ne'
  apply h3.congr'
  filter_upwards with N
  rw [inv_div]

lemma mean_bounded (r : ℝ) (hr : 1 < r) :
    ∃ K : ℝ, ∀ N : ℕ, 1 ≤ N → |(N : ℝ) * mOne r - meanH r N| ≤ K := by
  refine ⟨r / (r - 1) + 2 * ((r + 1) / (r - 1) ^ 2), ?_⟩
  intro N hN
  have hr1 : 0 < r - 1 := by linarith
  have hr1' : r - 1 ≠ 0 := hr1.ne'
  have hNR : (1 : ℝ) ≤ N := by exact_mod_cast hN
  have hpow : (N : ℝ) * (r - 1) ≤ r ^ N - 1 := by
    have h := one_add_mul_le_pow (show (-2 : ℝ) ≤ r - 1 by linarith) N
    rw [show (1 : ℝ) + (r - 1) = r by ring] at h
    linarith
  have hrN : 0 < r ^ N - 1 := by nlinarith
  have hrN' : r ^ N - 1 ≠ 0 := hrN.ne'
  have hx0 : 0 ≤ (N : ℝ) / (r ^ N - 1) := div_nonneg (by positivity) hrN.le
  have hx1 : (N : ℝ) / (r ^ N - 1) ≤ 1 / (r - 1) := by
    rw [div_le_div_iff₀ hrN hr1]
    linarith
  have e : (N : ℝ) * mOne r - meanH r N
      = r / (r - 1) + (r + 1) / (r - 1) ^ 2 - (r + 1) / (r - 1) * ((N : ℝ) / (r ^ N - 1)) := by
    unfold mOne meanH
    field_simp
    ring
  rw [e]
  have hA : 0 < r / (r - 1) := div_pos (by linarith) hr1
  have hB : 0 < (r + 1) / (r - 1) ^ 2 := div_pos (by linarith) (pow_pos hr1 2)
  have hC : 0 < (r + 1) / (r - 1) := div_pos (by linarith) hr1
  have hCx : (r + 1) / (r - 1) * ((N : ℝ) / (r ^ N - 1)) ≤ (r + 1) / (r - 1) ^ 2 := by
    calc (r + 1) / (r - 1) * ((N : ℝ) / (r ^ N - 1)) ≤ (r + 1) / (r - 1) * (1 / (r - 1)) :=
          mul_le_mul_of_nonneg_left hx1 hC.le
      _ = (r + 1) / (r - 1) ^ 2 := by field_simp
  have hCx0 : 0 ≤ (r + 1) / (r - 1) * ((N : ℝ) / (r ^ N - 1)) := mul_nonneg hC.le hx0
  rw [abs_le]
  constructor <;> linarith

lemma qL_eq (r θ : ℝ) :
    qL r ((θ : ℂ) * I) = ((mOne r * θ : ℝ) : ℂ) * I - ((vOne r / 2 * θ ^ 2 : ℝ) : ℂ) := by
  unfold qL
  push_cast
  linear_combination ((vOne r : ℂ) / 2 * (θ : ℂ) ^ 2) * Complex.I_sq

/-- SI Theorem S3, central limit theorem: the characteristic function of `(H - μ_N)/σ_N` tends to
`e^{-τ²/2}` for every real `τ`. The hypothesis `hgf` is the generating function of the law `q N` of
`H` on the unit circle (SI Lemma S2, `gf_head_advances_complex`). -/
theorem clt_charFun (r : ℝ) (hr : 1 < r) (q : ℕ → ℕ → ℝ)
    (hgf : ∀ N, 2 ≤ N → ∀ θ : ℝ, ∑' n, (q N n : ℂ) * exp ((θ : ℂ) * n * I)
        = (2 * (alpha r : ℂ) * exp ((θ : ℂ) * I)) ^ (N - 1)
          / ((rhoH r N : ℂ) * PNc N (1 - 4 * (alpha r : ℂ) * beta r * exp ((θ : ℂ) * I))))
    (τ : ℝ) :
    Tendsto (fun N : ℕ => exp (((-(τ * meanH r N / √(varH r N))) : ℝ) * I)
        * ∑' n, (q N n : ℂ) * exp (((τ / √(varH r N)) : ℝ) * n * I))
      atTop (𝓝 (exp ((-(τ ^ 2 / 2) : ℝ) : ℂ))) := by
  obtain ⟨δ, hδ, C, hC⟩ := logBf_expansion r hr
  obtain ⟨Kμ, hKμ⟩ := mean_bounded r hr
  have hv := vOne_pos r hr
  have hs0 := s0_pos r hr
  have hrinv : r⁻¹ < 1 := inv_lt_one_of_one_lt₀ hr
  have hrinv0 : 0 < r⁻¹ := inv_pos.mpr (by linarith)
  have hσ : Tendsto (fun N : ℕ => √(varH r N)) atTop atTop :=
    Real.tendsto_sqrt_atTop.comp (varH_tendsto r hr)
  have hσinv : Tendsto (fun N : ℕ => (√(varH r N))⁻¹) atTop (𝓝 0) := hσ.inv_tendsto_atTop
  have hθ : Tendsto (fun N : ℕ => τ / √(varH r N)) atTop (𝓝 0) := by
    have := hσinv.const_mul τ
    rw [mul_zero] at this
    exact this.congr (fun N => by rw [div_eq_mul_inv])
  have hpos_ev := (varH_tendsto r hr).eventually_gt_atTop 0
  -- neighborhoods of `θ = 0`
  obtain ⟨δR, hδR, hRball⟩ := Metric.continuousAt_iff.mp (Rf_continuousAt r hr) ((1 - r⁻¹) / 2)
    (by linarith)
  obtain ⟨δs, hδs, hsball⟩ := Metric.continuousAt_iff.mp (sq_continuousAt r hr) (s0 r / 2)
    (by linarith)
  have hsmall : ∀ᶠ N : ℕ in atTop, |τ / √(varH r N)| < min δ (min δR δs) := by
    have := (Metric.tendsto_nhds.mp hθ) (min δ (min δR δs)) (by positivity)
    filter_upwards [this] with N hN
    rwa [Real.dist_eq, sub_zero] at hN
  have hRbound : ∀ θ : ℝ, |θ| < min δ (min δR δs) → ‖Rf r θ‖ ≤ (1 + r⁻¹) / 2 := by
    intro θ hθ'
    have h := hRball (show dist θ 0 < δR by
      rw [Real.dist_eq, sub_zero]
      exact hθ'.trans_le ((min_le_right _ _).trans (min_le_left _ _)))
    rw [dist_eq_norm, Rf_zero r hr] at h
    have h2 := norm_sub_norm_le (Rf r θ) ((r⁻¹ : ℝ) : ℂ)
    rw [Complex.norm_real, Real.norm_eq_abs, abs_of_pos hrinv0] at h2
    linarith
  have hsbound : ∀ θ : ℝ, |θ| < min δ (min δR δs) → sq r θ ≠ 0 ∧ 1 + sq r θ ≠ 0 := by
    intro θ hθ'
    have h := hsball (show dist θ 0 < δs by
      rw [Real.dist_eq, sub_zero]
      exact hθ'.trans_le ((min_le_right _ _).trans (min_le_right _ _)))
    rw [dist_eq_norm, sq_zero r hr] at h
    constructor
    · intro h0
      rw [h0, zero_sub, norm_neg, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hs0] at h
      linarith
    · intro h0
      have h1 : sq r θ = -1 := by linear_combination h0
      rw [h1] at h
      have h3 : ‖(-1 : ℂ) - (s0 r : ℂ)‖ = 1 + s0 r := by
        rw [show (-1 : ℂ) - (s0 r : ℂ) = ((-(1 + s0 r) : ℝ) : ℂ) by push_cast; ring,
          Complex.norm_real, Real.norm_eq_abs, abs_neg, abs_of_pos (by linarith)]
      linarith
  -- factor 1
  have hA0 : Af r 0 ≠ 0 := by
    rw [Af_zero r hr]
    exact_mod_cast (sub_pos.mpr hrinv).ne'
  have hF1 : Tendsto (fun N : ℕ => Af r (τ / √(varH r N)) / Af r 0) atTop (𝓝 1) := by
    have h := ((Af_continuousAt r hr).tendsto.comp hθ).div_const (Af r 0)
    rw [div_self hA0] at h
    exact h
  -- factor 2
  have hR0pow : Tendsto (fun N : ℕ => Rf r 0 ^ N) atTop (𝓝 0) := by
    apply tendsto_pow_atTop_nhds_zero_of_norm_lt_one
    rw [Rf_zero r hr, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hrinv0]
    exact hrinv
  have hRpow : Tendsto (fun N : ℕ => Rf r (τ / √(varH r N)) ^ N) atTop (𝓝 0) := by
    have hρ0 : (0 : ℝ) ≤ (1 + r⁻¹) / 2 := by positivity
    have hρ1 : (1 + r⁻¹) / 2 < 1 := by linarith
    apply squeeze_zero_norm' _ (tendsto_pow_atTop_nhds_zero_of_lt_one hρ0 hρ1)
    filter_upwards [hsmall] with N hN
    rw [norm_pow]
    exact pow_le_pow_left₀ (norm_nonneg _) (hRbound _ hN) N
  have hF2 : Tendsto (fun N : ℕ => (1 - Rf r 0 ^ N) / (1 - Rf r (τ / √(varH r N)) ^ N))
      atTop (𝓝 1) := by
    have h := ((tendsto_const_nhds (x := (1 : ℂ))).sub hR0pow).div
      ((tendsto_const_nhds (x := (1 : ℂ))).sub hRpow) (by norm_num)
    simp only [sub_zero, div_one] at h
    exact h
  -- factor 3: the argument of the exponential
  have ha0 : Tendsto (fun N : ℕ =>
      (N : ℝ) * mOne r * (τ / √(varH r N)) - τ * meanH r N / √(varH r N)) atTop (𝓝 0) := by
    have hbound : Tendsto (fun N : ℕ => |τ| * Kμ * (√(varH r N))⁻¹) atTop (𝓝 0) := by
      have := hσinv.const_mul (|τ| * Kμ)
      rwa [mul_zero] at this
    apply squeeze_zero_norm' _ hbound
    filter_upwards [eventually_ge_atTop 1, hpos_ev] with N hN hpos
    have hσpos : 0 < √(varH r N) := Real.sqrt_pos.mpr hpos
    have e : (N : ℝ) * mOne r * (τ / √(varH r N)) - τ * meanH r N / √(varH r N)
        = τ * ((N : ℝ) * mOne r - meanH r N) * (√(varH r N))⁻¹ := by ring
    rw [e, Real.norm_eq_abs, abs_mul, abs_mul, abs_inv, abs_of_pos hσpos]
    have h := hKμ N hN
    gcongr
  have hb0 : Tendsto (fun N : ℕ => (N : ℝ) * (τ / √(varH r N)) ^ 2) atTop
      (𝓝 (τ ^ 2 * (vOne r)⁻¹)) := by
    have h := (N_div_varH_tendsto r hr).const_mul (τ ^ 2)
    apply h.congr'
    filter_upwards [hpos_ev] with N hpos
    rw [div_pow, Real.sq_sqrt hpos.le]
    ring
  have hE0 : Tendsto (fun N : ℕ => (N : ℂ) * (log (Bf r (τ / √(varH r N)))
      - qL r (((τ / √(varH r N) : ℝ) : ℂ) * I))) atTop (𝓝 0) := by
    have hbound : Tendsto (fun N : ℕ => |C| * |τ| ^ 3 * ((N : ℝ) / varH r N) * (√(varH r N))⁻¹)
        atTop (𝓝 0) := by
      have h1 := (N_div_varH_tendsto r hr).const_mul (|C| * |τ| ^ 3)
      have h2 := h1.mul hσinv
      rwa [mul_zero] at h2
    apply squeeze_zero_norm' _ hbound
    filter_upwards [hsmall, hpos_ev] with N hN hpos
    have hσpos : 0 < √(varH r N) := Real.sqrt_pos.mpr hpos
    have hθδ : |τ / √(varH r N)| ≤ δ := hN.le.trans (min_le_left _ _)
    have hlog := (hC _ hθδ).2
    have hcube : √(varH r N) ^ 3 = varH r N * √(varH r N) := by
      rw [pow_succ, Real.sq_sqrt hpos.le]
    rw [norm_mul, Complex.norm_natCast]
    calc (N : ℝ) * ‖log (Bf r (τ / √(varH r N))) - qL r (((τ / √(varH r N) : ℝ) : ℂ) * I)‖
        ≤ (N : ℝ) * (C * |τ / √(varH r N)| ^ 3) := by gcongr
      _ ≤ (N : ℝ) * (|C| * |τ / √(varH r N)| ^ 3) := by
          have h3 : 0 ≤ |τ / √(varH r N)| ^ 3 := by positivity
          have := mul_le_mul_of_nonneg_right (le_abs_self C) h3
          have hN0 : (0 : ℝ) ≤ N := Nat.cast_nonneg N
          exact mul_le_mul_of_nonneg_left this hN0
      _ = |C| * |τ| ^ 3 * ((N : ℝ) / varH r N) * (√(varH r N))⁻¹ := by
          rw [abs_div, abs_of_pos hσpos, div_pow, hcube]
          field_simp
  have harg : Tendsto (fun N : ℕ => (N : ℂ) * log (Bf r (τ / √(varH r N)))
      + (((-(τ * meanH r N / √(varH r N))) : ℝ) : ℂ) * I) atTop
      (𝓝 ((-(τ ^ 2 / 2) : ℝ) : ℂ)) := by
    have hdecomp : ∀ N : ℕ, (N : ℂ) * log (Bf r (τ / √(varH r N)))
        + (((-(τ * meanH r N / √(varH r N))) : ℝ) : ℂ) * I
        = I * ((((N : ℝ) * mOne r * (τ / √(varH r N)) - τ * meanH r N / √(varH r N)) : ℝ) : ℂ)
          - (((vOne r / 2) * ((N : ℝ) * (τ / √(varH r N)) ^ 2) : ℝ) : ℂ)
          + (N : ℂ) * (log (Bf r (τ / √(varH r N)))
              - qL r (((τ / √(varH r N) : ℝ) : ℂ) * I)) := by
      intro N
      rw [qL_eq]
      push_cast
      ring
    have hlim : Tendsto (fun N : ℕ =>
        I * ((((N : ℝ) * mOne r * (τ / √(varH r N)) - τ * meanH r N / √(varH r N)) : ℝ) : ℂ)
          - (((vOne r / 2) * ((N : ℝ) * (τ / √(varH r N)) ^ 2) : ℝ) : ℂ)
          + (N : ℂ) * (log (Bf r (τ / √(varH r N)))
              - qL r (((τ / √(varH r N) : ℝ) : ℂ) * I)))
        atTop (𝓝 (I * ((0 : ℝ) : ℂ) - (((vOne r / 2) * (τ ^ 2 * (vOne r)⁻¹) : ℝ) : ℂ) + 0)) := by
      apply Tendsto.add
      · apply Tendsto.sub
        · exact tendsto_const_nhds.mul ((Complex.continuous_ofReal.tendsto _).comp ha0)
        · exact (Complex.continuous_ofReal.tendsto _).comp (hb0.const_mul (vOne r / 2))
      · exact hE0
    have hval : I * ((0 : ℝ) : ℂ) - (((vOne r / 2) * (τ ^ 2 * (vOne r)⁻¹) : ℝ) : ℂ) + 0
        = ((-(τ ^ 2 / 2) : ℝ) : ℂ) := by
      have hv0 : (vOne r : ℂ) ≠ 0 := by exact_mod_cast hv.ne'
      push_cast
      field_simp
      try ring
    rw [hval] at hlim
    exact hlim.congr (fun N => (hdecomp N).symm)
  -- the eventual factorization
  have hevent : ∀ᶠ N : ℕ in atTop,
      Af r (τ / √(varH r N)) / Af r 0 * ((1 - Rf r 0 ^ N) / (1 - Rf r (τ / √(varH r N)) ^ N))
        * exp ((N : ℂ) * log (Bf r (τ / √(varH r N)))
          + (((-(τ * meanH r N / √(varH r N))) : ℝ) : ℂ) * I)
      = exp (((-(τ * meanH r N / √(varH r N))) : ℝ) * I)
        * ∑' n, (q N n : ℂ) * exp (((τ / √(varH r N)) : ℝ) * n * I) := by
    filter_upwards [eventually_ge_atTop 2, hsmall] with N hN2 hN
    have hθδ : |τ / √(varH r N)| ≤ δ := hN.le.trans (min_le_left _ _)
    obtain ⟨hB1, -⟩ := hC _ hθδ
    have hB0 : Bf r (τ / √(varH r N)) ≠ 0 := by
      intro h
      rw [h, zero_sub, norm_neg, norm_one] at hB1
      norm_num at hB1
    obtain ⟨hs, hs1⟩ := hsbound _ hN
    have hRθ := hRbound _ hN
    rw [hgf N hN2]
    obtain ⟨M, rfl⟩ : ∃ M, N = M + 1 := ⟨N - 1, by omega⟩
    have hRne : 1 - Rf r (τ / √(varH r (M + 1))) ^ (M + 1) ≠ 0 := by
      intro h
      have h1 : Rf r (τ / √(varH r (M + 1))) ^ (M + 1) = 1 := by linear_combination -h
      have h2 : ‖Rf r (τ / √(varH r (M + 1))) ^ (M + 1)‖ < 1 := by
        rw [norm_pow]
        calc ‖Rf r (τ / √(varH r (M + 1)))‖ ^ (M + 1) ≤ ((1 + r⁻¹) / 2) ^ (M + 1) :=
              pow_le_pow_left₀ (norm_nonneg _) hRθ _
          _ < 1 := pow_lt_one₀ (by positivity) (by linarith) (by omega)
      rw [h1, norm_one] at h2
      exact lt_irrefl _ h2
    have hR0ne : 1 - Rf r 0 ^ (M + 1) ≠ 0 := by
      rw [Rf_zero r hr]
      intro h
      have h1 : ((r⁻¹ : ℝ) : ℂ) ^ (M + 1) = 1 := by linear_combination -h
      have h2 : ‖((r⁻¹ : ℝ) : ℂ) ^ (M + 1)‖ = r⁻¹ ^ (M + 1) := by
        rw [norm_pow, Complex.norm_real, Real.norm_eq_abs, abs_of_pos hrinv0]
      rw [h1, norm_one] at h2
      have h3 : r⁻¹ ^ (M + 1) < 1 := pow_lt_one₀ hrinv0.le hrinv (by omega)
      linarith
    have hfac := gf_factor r (τ / √(varH r (M + 1))) hr M hs hs1 hRne
    rw [Nat.add_sub_cancel, rhoH_eq r hr (M + 1), mul_comm (Af r 0 / (1 - Rf r 0 ^ (M + 1))),
      ← div_div, hfac, Complex.exp_add, Complex.exp_nat_mul, Complex.exp_log hB0]
    field_simp
  have hlim : Tendsto (fun N : ℕ =>
      Af r (τ / √(varH r N)) / Af r 0 * ((1 - Rf r 0 ^ N) / (1 - Rf r (τ / √(varH r N)) ^ N))
        * exp ((N : ℂ) * log (Bf r (τ / √(varH r N)))
          + (((-(τ * meanH r N / √(varH r N))) : ℝ) : ℂ) * I))
      atTop (𝓝 (1 * 1 * exp ((-(τ ^ 2 / 2) : ℝ) : ℂ))) :=
    (hF1.mul hF2).mul ((Complex.continuous_exp.tendsto _).comp harg)
  rw [one_mul, one_mul] at hlim
  exact hlim.congr' hevent

end

end FixLoc
