import FixLoc.CLTLaw
import Mathlib.MeasureTheory.Measure.Portmanteau
import Mathlib.Probability.CDF

/-!
# Concentration of the fixation location on the directed cycle (SI Corollary S2)

Let `m_N = (N-1) r/(r-1) - (r+1)/(r-1)²` and `σ_r² = r(r+1)/(r-1)³`. The probability of the event
`|H - m_N| ≤ a√N` has limit inferior at least `2Φ(a/σ_r) - 1`, where `Φ` is the standard normal
distribution function (`concentration_location`). The proof combines the central limit theorem
`clt_head_advances` with the portmanteau theorem for open sets. The open sets are the intervals
`(-c(k+1)/(k+2), c(k+1)/(k+2))` with `c = a/σ_r`, which increase to `(-c, c)`.
-/

open Filter Topology MeasureTheory ProbabilityTheory
open scoped Real ENNReal

namespace FixLoc

noncomputable section

/-- The center `m_N = (N-1) r/(r-1) - (r+1)/(r-1)²` of SI Corollary S2. -/
def mN (r : ℝ) (N : ℕ) : ℝ := ((N : ℝ) - 1) * r / (r - 1) - (r + 1) / (r - 1) ^ 2

/-- The exact mean differs from `m_N` by at most `(r+1)/(r-1)²`. -/
lemma meanH_sub_mN (r : ℝ) (hr : 1 < r) (N : ℕ) (hN : 1 ≤ N) :
    |meanH r N - mN r N| ≤ (r + 1) / (r - 1) ^ 2 := by
  have hr1 : 0 < r - 1 := by linarith
  have hpow : (N : ℝ) * (r - 1) ≤ r ^ N - 1 := by
    have h := one_add_mul_le_pow (show (-2 : ℝ) ≤ r - 1 by linarith) N
    rw [show (1 : ℝ) + (r - 1) = r by ring] at h
    linarith
  have hNR : (1 : ℝ) ≤ N := by exact_mod_cast hN
  have hrN : 0 < r ^ N - 1 := by nlinarith
  have e : meanH r N - mN r N = (N : ℝ) * (r + 1) / ((r - 1) * (r ^ N - 1)) := by
    unfold meanH mN
    ring
  have hc0 : 0 ≤ (N : ℝ) * (r + 1) / ((r - 1) * (r ^ N - 1)) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg N) (by linarith)) (mul_pos hr1 hrN).le
  have hc1 : (N : ℝ) * (r + 1) / ((r - 1) * (r ^ N - 1)) ≤ (r + 1) / (r - 1) ^ 2 := by
    rw [div_le_div_iff₀ (mul_pos hr1 hrN) (pow_pos hr1 2)]
    have h := mul_le_mul_of_nonneg_left hpow (mul_pos (by linarith : (0 : ℝ) < r + 1) hr1).le
    nlinarith [h]
  rw [e, abs_of_nonneg hc0]
  exact hc1

/-- `σ_N/√N → σ_r`. -/
lemma sqrt_varH_div_tendsto (r : ℝ) (hr : 1 < r) :
    Tendsto (fun N : ℕ => √(varH r N) / √(N : ℝ)) atTop (𝓝 (√(vOne r))) := by
  have hv := vOne_pos r hr
  have h1 : Tendsto (fun N : ℕ => varH r N / N) atTop (𝓝 (vOne r)) := by
    have h := (N_div_varH_tendsto r hr).inv₀ (inv_ne_zero hv.ne')
    rw [inv_inv] at h
    exact h.congr (fun N => inv_div _ _)
  have h2 := (Real.continuous_sqrt.tendsto (vOne r)).comp h1
  exact h2.congr (fun N => Real.sqrt_div' _ (Nat.cast_nonneg N))

/-- The open interval `(-c(k+1)/(k+2), c(k+1)/(k+2))`. -/
def shrink (c : ℝ) (k : ℕ) : Set ℝ :=
  Set.Ioo (-(c * (((k : ℝ) + 1) / ((k : ℝ) + 2)))) (c * (((k : ℝ) + 1) / ((k : ℝ) + 2)))

lemma frac_mono {k l : ℕ} (h : k ≤ l) :
    ((k : ℝ) + 1) / ((k : ℝ) + 2) ≤ ((l : ℝ) + 1) / ((l : ℝ) + 2) := by
  have hkl : (k : ℝ) ≤ l := by exact_mod_cast h
  rw [div_le_div_iff₀ (by positivity) (by positivity)]
  nlinarith

lemma shrink_mono (c : ℝ) (hc : 0 ≤ c) : Monotone (shrink c) := by
  intro k l hkl
  have h := mul_le_mul_of_nonneg_left (frac_mono hkl) hc
  show shrink c k ⊆ shrink c l
  unfold shrink
  exact Set.Ioo_subset_Ioo (by linarith) h

lemma iUnion_shrink (c : ℝ) (hc : 0 < c) : (⋃ k, shrink c k) = Set.Ioo (-c) c := by
  ext x
  simp only [Set.mem_iUnion, shrink, Set.mem_Ioo]
  constructor
  · rintro ⟨k, hl, hu⟩
    have h1 : ((k : ℝ) + 1) / ((k : ℝ) + 2) < 1 := by
      rw [div_lt_one (by positivity)]
      linarith
    have h2 := mul_le_mul_of_nonneg_left h1.le hc.le
    rw [mul_one] at h2
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨hl, hu⟩
    have hx : |x| < c := abs_lt.mpr ⟨hl, hu⟩
    have hd : 0 < c - |x| := by linarith
    obtain ⟨k, hk⟩ := exists_nat_gt (c / (c - |x|))
    have hk2 : c / (c - |x|) < (k : ℝ) + 2 := by linarith
    have h3 : c < ((k : ℝ) + 2) * (c - |x|) := by
      rw [div_lt_iff₀ hd] at hk2
      linarith
    have h4 : |x| < c * (((k : ℝ) + 1) / ((k : ℝ) + 2)) := by
      rw [show c * (((k : ℝ) + 1) / ((k : ℝ) + 2)) = c * ((k : ℝ) + 1) / ((k : ℝ) + 2) by ring,
        lt_div_iff₀ (by positivity)]
      linarith
    exact ⟨k, by linarith [neg_abs_le x], by linarith [le_abs_self x]⟩

/-- `2Φ(c) - 1` is the standard normal probability of `(-c, c)`. -/
lemma gauss_Ioo (c : ℝ) (hc : 0 < c) :
    gaussianReal 0 1 (Set.Ioo (-c) c) = ENNReal.ofReal (2 * cdf (gaussianReal 0 1) c - 1) := by
  have : NullSingletonClass (gaussianReal 0 1) := nullSingletonClass_gaussianReal one_ne_zero
  have hneg : (gaussianReal 0 1).map (fun x : ℝ => -x) = gaussianReal 0 1 := by
    rw [gaussianReal_map_neg, neg_zero]
  have hsym : (gaussianReal 0 1).real (Set.Iic (-c)) = (gaussianReal 0 1).real (Set.Ici c) := by
    have hpre : (fun x : ℝ => -x) ⁻¹' Set.Ici c = Set.Iic (-c) := by
      ext x
      simp only [Set.mem_preimage, Set.mem_Ici, Set.mem_Iic]
      constructor <;> intro h <;> linarith
    rw [← hpre, measureReal_def, measureReal_def,
      ← Measure.map_apply (measurable_neg : Measurable (fun x : ℝ => -x)) measurableSet_Ici, hneg]
  have hunion : (gaussianReal 0 1).real (Set.Iio c)
      = (gaussianReal 0 1).real (Set.Iic (-c)) + (gaussianReal 0 1).real (Set.Ioo (-c) c) := by
    rw [← Set.Iic_union_Ioo_eq_Iio (by linarith : -c < c)]
    exact measureReal_union
      (Disjoint.mono_right Set.Ioo_subset_Ioi_self (Set.Iic_disjoint_Ioi le_rfl)) measurableSet_Ioo
  have hae : Set.Iio c =ᵐ[gaussianReal 0 1] Set.Iic c := Iio_ae_eq_Iic
  have hIio : (gaussianReal 0 1).real (Set.Iio c) = cdf (gaussianReal 0 1) c := by
    rw [cdf_eq_real, measureReal_congr hae]
  have hIci : (gaussianReal 0 1).real (Set.Ici c) = 1 - cdf (gaussianReal 0 1) c := by
    rw [← Set.compl_Iio, probReal_compl_eq_one_sub measurableSet_Iio, hIio]
  have hreal : (gaussianReal 0 1).real (Set.Ioo (-c) c) = 2 * cdf (gaussianReal 0 1) c - 1 := by
    linarith
  rw [← hreal, ofReal_measureReal (measure_ne_top (gaussianReal 0 1) (Set.Ioo (-c) c))]

lemma stdLaw_apply (r : ℝ) (N : ℕ) (p : PMF ℕ) (S : Set ℝ) (hS : MeasurableSet S) :
    (stdLaw r N p : Measure ℝ) S
      = p.toMeasure ((fun n : ℕ => ((n : ℝ) - meanH r N) / √(varH r N)) ⁻¹' S) := by
  show (p.toMeasure.map (fun n : ℕ => ((n : ℝ) - meanH r N) / √(varH r N))) S = _
  exact Measure.map_apply measurable_from_nat hS

/-- SI Corollary S2, concentration of the fixation location. Let `m_N = (N-1)r/(r-1) - (r+1)/(r-1)²`
and `σ_r² = r(r+1)/(r-1)³`. The probability of `|H - m_N| ≤ a√N` has limit inferior at least
`2Φ(a/σ_r) - 1`. The event `|H - m_N| ≤ a√N` places the fixation location `1 + H` within `a√N`
vertices of vertex `1 + m_N` (modulo `N`). -/
theorem concentration_location (r : ℝ) (hr : 1 < r) (a : ℝ) (ha : 0 < a) (p : ℕ → PMF ℕ)
    (hgf : ∀ N, 2 ≤ N → ∀ θ : ℝ,
      ∑' n, (((p N n).toReal : ℝ) : ℂ) * Complex.exp ((θ : ℂ) * n * Complex.I)
        = (2 * (alpha r : ℂ) * Complex.exp ((θ : ℂ) * Complex.I)) ^ (N - 1)
          / ((rhoH r N : ℂ) * PNc N
            (1 - 4 * (alpha r : ℂ) * beta r * Complex.exp ((θ : ℂ) * Complex.I)))) :
    ENNReal.ofReal (2 * cdf (gaussianReal 0 1) (a / √(vOne r)) - 1)
      ≤ liminf (fun N : ℕ =>
          (p N).toMeasure {n : ℕ | |(n : ℝ) - mN r N| ≤ a * √(N : ℝ)}) atTop := by
  have hv := vOne_pos r hr
  have hsv : 0 < √(vOne r) := Real.sqrt_pos.mpr hv
  have hc0 : 0 < a / √(vOne r) := div_pos ha hsv
  rw [← gauss_Ioo _ hc0, ← iUnion_shrink _ hc0, (shrink_mono _ hc0.le).measure_iUnion]
  refine iSup_le (fun k => ?_)
  set c := a / √(vOne r) with hc
  have hfrac : ((k : ℝ) + 1) / ((k : ℝ) + 2) < 1 := by
    rw [div_lt_one (by positivity)]
    linarith
  -- `c (k+1)/(k+2) σ_r < a`
  have hsv' : √(vOne r) ≠ 0 := hsv.ne'
  have hcv : c * √(vOne r) = a := by
    rw [hc]
    field_simp
  have hck_lt : c * (((k : ℝ) + 1) / ((k : ℝ) + 2)) * √(vOne r) < a := by
    rw [mul_right_comm, hcv]
    calc a * (((k : ℝ) + 1) / ((k : ℝ) + 2)) < a * 1 := mul_lt_mul_of_pos_left hfrac ha
      _ = a := mul_one a
  -- `(c (k+1)/(k+2) σ_N + (r+1)/(r-1)²)/√N → c (k+1)/(k+2) σ_r`
  have hT : Tendsto (fun N : ℕ => (c * (((k : ℝ) + 1) / ((k : ℝ) + 2)) * √(varH r N)
      + (r + 1) / (r - 1) ^ 2) / √(N : ℝ)) atTop
      (𝓝 (c * (((k : ℝ) + 1) / ((k : ℝ) + 2)) * √(vOne r))) := by
    have h1 := (sqrt_varH_div_tendsto r hr).const_mul (c * (((k : ℝ) + 1) / ((k : ℝ) + 2)))
    have h2 : Tendsto (fun N : ℕ => (r + 1) / (r - 1) ^ 2 * (√(N : ℝ))⁻¹) atTop
        (𝓝 ((r + 1) / (r - 1) ^ 2 * 0)) :=
      ((Real.tendsto_sqrt_atTop.comp tendsto_natCast_atTop_atTop).inv_tendsto_atTop).const_mul _
    rw [mul_zero] at h2
    have h3 := h1.add h2
    rw [add_zero] at h3
    exact h3.congr (fun N => by ring)
  have hev := hT.eventually (gt_mem_nhds hck_lt)
  -- portmanteau theorem for the open interval `shrink c k`
  have hlim := clt_head_advances r hr p hgf
  have h1 := ProbabilityMeasure.le_liminf_measure_open_of_tendsto hlim
    (isOpen_Ioo : IsOpen (shrink c k))
  refine le_trans (le_of_eq (by rw [stdNormal_coe]; rfl)) (le_trans h1 (liminf_le_liminf ?_))
  filter_upwards [hev, eventually_ge_atTop 1, (varH_tendsto r hr).eventually_gt_atTop 0]
    with N hN hN1 hvar
  have hσ : 0 < √(varH r N) := Real.sqrt_pos.mpr hvar
  have hNpos : (0 : ℝ) < N := by
    have : (1 : ℝ) ≤ N := by exact_mod_cast hN1
    linarith
  have hsN : 0 < √(N : ℝ) := Real.sqrt_pos.mpr hNpos
  rw [stdLaw_apply r N (p N) _ (measurableSet_Ioo : MeasurableSet (shrink c k))]
  apply measure_mono
  intro n hn
  simp only [Set.mem_preimage, Set.mem_Ioo] at hn
  obtain ⟨hl, hu⟩ := hn
  have e1 := (lt_div_iff₀ hσ).mp hl
  have e2 := (div_lt_iff₀ hσ).mp hu
  have e3 := (div_lt_iff₀ hsN).mp hN
  obtain ⟨e4, e5⟩ := abs_le.mp (meanH_sub_mN r hr N hN1)
  show |(n : ℝ) - mN r N| ≤ a * √(N : ℝ)
  rw [abs_le]
  constructor <;> linarith

end

end FixLoc
