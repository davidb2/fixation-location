import FixLoc.Fourier

/-!
# The directed cycle under neutral evolution (SI Proposition S6)

Under neutral evolution the fixation location probabilities differ from `1/N` by
`∑_{ℓ=1}^{N-1} ω^(-ℓk) / P_N(1 - ω^ℓ)`. This file proves the lower bound on `|P_N(1 - ω^ℓ)|` of the
SI proof of Proposition S6.

* `two_mul_PNc_sq`: `2x P_N(x²) = (1+x)^N - (1-x)^N`.
* `sqrtOneSub`: an explicit square root `x` of `1 - e^(2iθ)` with `Re x ≥ √(sin θ)` and `|x|² = 2 sin θ`.
* `PNc_lower`: `(1 + √(sin θ))^N - 1 ≤ 2√2 √(sin θ) |P_N(1 - e^(2iθ))|` for `0 < θ < π`.
-/

open Finset Complex
open scoped Real

namespace FixLoc

noncomputable section

/-- `2x P_N(x²) = (1+x)^N - (1-x)^N`. -/
lemma two_mul_PNc_sq (x : ℂ) (N : ℕ) :
    2 * x * PNc N (x ^ 2) = (1 + x) ^ N - (1 - x) ^ N := by
  have pair : ∀ n : ℕ, (2 * x * PNc n (x ^ 2) = (1 + x) ^ n - (1 - x) ^ n) ∧
      (2 * x * PNc (n + 1) (x ^ 2) = (1 + x) ^ (n + 1) - (1 - x) ^ (n + 1)) := by
    intro n
    induction n with
    | zero =>
      constructor
      · simp [PNc_zero]
      · simp only [zero_add, pow_one, PNc_one]
        ring
    | succ n ih =>
      refine ⟨ih.2, ?_⟩
      have e : 2 * x * PNc (n + 1 + 1) (x ^ 2)
          = 2 * (2 * x * PNc (n + 1) (x ^ 2)) - (1 - x ^ 2) * (2 * x * PNc n (x ^ 2)) := by
        rw [show n + 1 + 1 = n + 2 from rfl, PNc_succ_succ]
        ring
      rw [e, ih.1, ih.2]
      ring
  exact (pair N).1

/-- A square root of `1 - e^(2iθ)`. -/
def sqrtOneSub (θ : ℝ) : ℂ :=
  ⟨√(Real.sin θ) * (Real.cos (θ / 2) + Real.sin (θ / 2)),
    √(Real.sin θ) * (Real.sin (θ / 2) - Real.cos (θ / 2))⟩

lemma sqrtOneSub_sq (θ : ℝ) (hθ : 0 ≤ Real.sin θ) :
    sqrtOneSub θ ^ 2 = 1 - exp ((2 * θ : ℝ) * I) := by
  have hs2 : √(Real.sin θ) * √(Real.sin θ) = Real.sin θ := Real.mul_self_sqrt hθ
  have hsin : Real.sin θ = 2 * Real.sin (θ / 2) * Real.cos (θ / 2) := by
    have h := Real.sin_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    exact h
  have hcos : Real.cos θ = Real.cos (θ / 2) ^ 2 - Real.sin (θ / 2) ^ 2 := by
    have h := Real.cos_two_mul (θ / 2)
    rw [show 2 * (θ / 2) = θ by ring] at h
    have h1 := Real.sin_sq_add_cos_sq (θ / 2)
    rw [h]
    linear_combination h1
  have hcos2 : Real.cos (2 * θ) = 1 - 2 * Real.sin θ ^ 2 := by
    have h1 := Real.sin_sq_add_cos_sq θ
    rw [Real.cos_two_mul θ]
    linear_combination 2 * h1
  have hsin2 : Real.sin (2 * θ) = 2 * Real.sin θ * Real.cos θ := Real.sin_two_mul θ
  apply Complex.ext
  · simp only [sqrtOneSub, sq, Complex.mul_re, Complex.sub_re, Complex.one_re,
      Complex.exp_ofReal_mul_I_re]
    linear_combination (4 * Real.cos (θ / 2) * Real.sin (θ / 2)) * hs2
      - (2 * Real.sin θ) * hsin + hcos2
  · simp only [sqrtOneSub, sq, Complex.mul_im, Complex.sub_im, Complex.one_im,
      Complex.exp_ofReal_mul_I_im]
    linear_combination (2 * (Real.sin (θ / 2) ^ 2 - Real.cos (θ / 2) ^ 2)) * hs2 + hsin2
      + (2 * Real.sin θ) * hcos

lemma sqrtOneSub_re_ge (θ : ℝ) (h0 : 0 ≤ θ) (hπ : θ ≤ π) :
    √(Real.sin θ) ≤ (sqrtOneSub θ).re := by
  have hc0 : 0 ≤ Real.cos (θ / 2) :=
    Real.cos_nonneg_of_mem_Icc ⟨by linarith [Real.pi_pos], by linarith⟩
  have hd0 : 0 ≤ Real.sin (θ / 2) :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith [Real.pi_pos])
  have hc1 : Real.cos (θ / 2) ≤ 1 := Real.cos_le_one _
  have hd1 : Real.sin (θ / 2) ≤ 1 := Real.sin_le_one _
  have h1 := Real.sin_sq_add_cos_sq (θ / 2)
  have hsum : 1 ≤ Real.cos (θ / 2) + Real.sin (θ / 2) := by nlinarith
  have hs0 : 0 ≤ √(Real.sin θ) := Real.sqrt_nonneg _
  show √(Real.sin θ) ≤ √(Real.sin θ) * (Real.cos (θ / 2) + Real.sin (θ / 2))
  nlinarith

lemma sqrtOneSub_normSq (θ : ℝ) (hθ : 0 ≤ Real.sin θ) :
    Complex.normSq (sqrtOneSub θ) = 2 * Real.sin θ := by
  rw [Complex.normSq_apply]
  simp only [sqrtOneSub]
  have hs2 : √(Real.sin θ) * √(Real.sin θ) = Real.sin θ := Real.mul_self_sqrt hθ
  have h1 := Real.sin_sq_add_cos_sq (θ / 2)
  linear_combination 2 * hs2 + (2 * (√(Real.sin θ) * √(Real.sin θ))) * h1

/-- The lower bound of the SI proof of Proposition S6:
`(1 + s)^N - 1 ≤ 2√2 s |P_N(1 - e^(2iθ))|` with `s = √(sin θ)`, for `0 < θ < π`. -/
lemma PNc_lower (N : ℕ) (θ : ℝ) (h0 : 0 < θ) (hπ : θ < π) :
    (1 + √(Real.sin θ)) ^ N - 1
      ≤ 2 * √2 * √(Real.sin θ) * ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖ := by
  have hsin : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi h0 hπ
  have hsin1 : Real.sin θ ≤ 1 := Real.sin_le_one θ
  set s := √(Real.sin θ) with hsdef
  have hs0 : 0 ≤ s := Real.sqrt_nonneg _
  have hs1 : s ≤ 1 := by
    calc s = √(Real.sin θ) := hsdef
      _ ≤ √1 := Real.sqrt_le_sqrt hsin1
      _ = 1 := Real.sqrt_one
  have hss : s * s = Real.sin θ := Real.mul_self_sqrt hsin.le
  set x := sqrtOneSub θ with hxdef
  have hx2 : x ^ 2 = 1 - exp ((2 * θ : ℝ) * I) := sqrtOneSub_sq θ hsin.le
  have hre : s ≤ x.re := sqrtOneSub_re_ge θ h0.le hπ.le
  have hnsq : Complex.normSq x = 2 * Real.sin θ := sqrtOneSub_normSq θ hsin.le
  -- `‖x‖ = √2 s`
  have hxnorm : ‖x‖ = √2 * s := by
    have h2 : ‖x‖ ^ 2 = (√2 * s) ^ 2 := by
      rw [Complex.sq_norm, hnsq, mul_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2), sq, hss]
    have hp : 0 ≤ √2 * s := mul_nonneg (Real.sqrt_nonneg 2) hs0
    rw [← Real.sqrt_sq (norm_nonneg x), h2, Real.sqrt_sq hp]
  -- `‖1 + x‖ ≥ 1 + s`
  have hplus : 1 + s ≤ ‖1 + x‖ := by
    have := Complex.re_le_norm (1 + x)
    simp only [Complex.add_re, Complex.one_re] at this
    linarith
  -- `‖1 - x‖ ≤ 1`
  have hminus : ‖1 - x‖ ≤ 1 := by
    have hsq : ‖1 - x‖ ^ 2 ≤ 1 := by
      rw [Complex.sq_norm, Complex.normSq_apply]
      simp only [Complex.sub_re, Complex.one_re, Complex.sub_im, Complex.one_im]
      have hn := hnsq
      rw [Complex.normSq_apply] at hn
      nlinarith [mul_nonneg hs0 (sub_nonneg.mpr hs1)]
    nlinarith [norm_nonneg (1 - x)]
  have key := two_mul_PNc_sq x N
  rw [hx2] at key
  have hlow : ‖1 + x‖ ^ N - ‖1 - x‖ ^ N ≤ ‖(1 + x) ^ N - (1 - x) ^ N‖ := by
    have := norm_sub_norm_le ((1 + x) ^ N) ((1 - x) ^ N)
    rwa [norm_pow, norm_pow] at this
  have hp1 : (1 + s) ^ N ≤ ‖1 + x‖ ^ N := pow_le_pow_left₀ (by linarith) hplus N
  have hp2 : ‖1 - x‖ ^ N ≤ 1 := pow_le_one₀ (norm_nonneg _) hminus
  have hnorm2 : ‖(1 + x) ^ N - (1 - x) ^ N‖
      = 2 * √2 * s * ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖ := by
    have h2 : ‖(2 : ℂ)‖ = 2 := by simp
    rw [← key, norm_mul, norm_mul, hxnorm, h2]
    ring
  linarith

end

end FixLoc
