import FixLoc.Neutral

/-!
# The directed cycle under neutral evolution: the bound of SI Proposition S6

`Neutral.lean` proves the lower bound on `|P_N(1 - e^(2iθ))|`. This file proves the elementary
inequalities of the SI proof, bounds the sum over the nontrivial roots of unity, and assembles
SI Proposition S6 (`neutral_location`) with the explicit constant `3840 e`.
-/

open Finset Complex
open scoped Real

namespace FixLoc

noncomputable section

/-! ## Elementary inequalities -/

/-- `e^(s/2) ≤ 1 + s` for `0 ≤ s ≤ 1`. -/
lemma exp_half_le (s : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) : Real.exp (s / 2) ≤ 1 + s := by
  have habs : |s / 2| ≤ 1 := by rw [abs_of_nonneg (by linarith)]; linarith
  have h := (abs_le.mp (Real.abs_exp_sub_one_sub_id_le habs)).2
  nlinarith

/-- `e^(Ns/2) / 2 ≤ (1 + s)^N - 1` for `0 ≤ s ≤ 1` and `N s ≥ 1`. -/
lemma pow_sub_one_ge (N : ℕ) (s : ℝ) (hs0 : 0 ≤ s) (hs1 : s ≤ 1) (hNs : 1 ≤ N * s) :
    Real.exp (N * s / 2) / 2 ≤ (1 + s) ^ N - 1 := by
  have h1 : Real.exp (N * s / 2) ≤ (1 + s) ^ N := by
    rw [show (N : ℝ) * s / 2 = N * (s / 2) by ring, Real.exp_nat_mul]
    exact pow_le_pow_left₀ (Real.exp_pos _).le (exp_half_le s hs0 hs1) N
  have h2 : 2 ≤ (1 + s) ^ N := by
    have := one_add_mul_le_pow (show (-2 : ℝ) ≤ s by linarith) N
    linarith
  linarith

/-- `s e^(-Ns/2)` decreases in `s` for `N s ≥ 2`. -/
lemma mul_exp_antitone (N s₀ s : ℝ) (h0 : 0 < s₀) (hN : 2 ≤ N * s₀) (hs : s₀ ≤ s) :
    s * Real.exp (-(N * s / 2)) ≤ s₀ * Real.exp (-(N * s₀ / 2)) := by
  have hstep : s ≤ s₀ * Real.exp (N * (s - s₀) / 2) := by
    have h1 := Real.add_one_le_exp (N * (s - s₀) / 2)
    have h2 : s ≤ s₀ * (N * (s - s₀) / 2 + 1) := by
      nlinarith [mul_nonneg (sub_nonneg.mpr hs) (sub_nonneg.mpr hN)]
    calc s ≤ s₀ * (N * (s - s₀) / 2 + 1) := h2
      _ ≤ s₀ * Real.exp (N * (s - s₀) / 2) := mul_le_mul_of_nonneg_left h1 h0.le
  have he : Real.exp (N * (s - s₀) / 2) * Real.exp (-(N * s / 2)) = Real.exp (-(N * s₀ / 2)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  calc s * Real.exp (-(N * s / 2))
      ≤ s₀ * Real.exp (N * (s - s₀) / 2) * Real.exp (-(N * s / 2)) :=
        mul_le_mul_of_nonneg_right hstep (Real.exp_pos _).le
    _ = s₀ * Real.exp (-(N * s₀ / 2)) := by rw [mul_assoc, he]

/-- The bound of one term of the SI proof of Proposition S6:
`1/|P_N(1 - e^(2iθ))| ≤ 4√2 s₀ e^(-N s₀/2)` whenever `0 < θ < π`, `N s₀ ≥ 2`, and `s₀ ≤ √(sin θ)`. -/
lemma inv_norm_PNc_le (N : ℕ) (θ s₀ : ℝ) (h0 : 0 < θ) (hπ : θ < π) (hs₀ : 0 < s₀)
    (hNs : 2 ≤ N * s₀) (hle : s₀ ≤ √(Real.sin θ)) :
    1 / ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖ ≤ 4 * √2 * s₀ * Real.exp (-(N * s₀ / 2)) := by
  have hsin : 0 < Real.sin θ := Real.sin_pos_of_pos_of_lt_pi h0 hπ
  have hlow := PNc_lower N θ h0 hπ
  set P := ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖ with hPdef
  set s := √(Real.sin θ) with hsdef
  have hs0 : 0 < s := Real.sqrt_pos.mpr hsin
  have hs1 : s ≤ 1 := by
    calc s = √(Real.sin θ) := hsdef
      _ ≤ √1 := Real.sqrt_le_sqrt (Real.sin_le_one θ)
      _ = 1 := Real.sqrt_one
  have hNs' : 1 ≤ (N : ℝ) * s := by nlinarith
  have hexp := pow_sub_one_ge N s hs0.le hs1 hNs'
  have hchain : Real.exp (N * s / 2) / 2 ≤ 2 * √2 * s * P := hexp.trans hlow
  have hP : 0 < P := by
    by_contra hneg
    have hP0 : P ≤ 0 := not_lt.mp hneg
    have : 2 * √2 * s * P ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (by positivity) hP0
    have : 0 < Real.exp (N * s / 2) / 2 := by positivity
    linarith
  have hEE : Real.exp (-(N * s / 2)) * Real.exp (N * s / 2) = 1 := by
    rw [← Real.exp_add]
    simp
  have h1 : 1 / P ≤ 4 * √2 * s * Real.exp (-(N * s / 2)) := by
    rw [div_le_iff₀ hP]
    have := mul_le_mul_of_nonneg_left hchain (Real.exp_pos (-(N * s / 2))).le
    nlinarith
  have h2 := mul_exp_antitone N s₀ s hs₀ hNs hle
  have h3 := mul_le_mul_of_nonneg_left h2 (by positivity : (0 : ℝ) ≤ 4 * √2)
  calc 1 / P ≤ 4 * √2 * s * Real.exp (-(N * s / 2)) := h1
    _ ≤ 4 * √2 * s₀ * Real.exp (-(N * s₀ / 2)) := by linarith

/-- `√m e^(-√m) ≤ 120 / m²` for `m ≥ 1`. -/
lemma sqrt_mul_exp_neg_sqrt_le (m : ℝ) (hm : 1 ≤ m) : √m * Real.exp (-√m) ≤ 120 / m ^ 2 := by
  have hb : 0 ≤ √m := Real.sqrt_nonneg m
  have hbsq : √m ^ 2 = m := Real.sq_sqrt (by linarith)
  have h5 := Real.pow_div_factorial_le_exp (√m) hb 5
  have hfac : ((Nat.factorial 5 : ℕ) : ℝ) = 120 := by norm_num [Nat.factorial]
  rw [hfac] at h5
  have h5' : m ^ 2 * √m / 120 ≤ Real.exp √m := by
    have e : √m ^ 5 = m ^ 2 * √m := by
      rw [show (5 : ℕ) = 2 * 2 + 1 from rfl, pow_succ, pow_mul, hbsq]
    rwa [e] at h5
  have hm2 : 0 < m ^ 2 := by positivity
  rw [Real.exp_neg, ← div_eq_mul_inv, div_le_div_iff₀ (Real.exp_pos _) hm2]
  linarith

/-- `∑_{j<M} 1/(j+1)² ≤ 2`. -/
lemma sum_inv_sq_le (M : ℕ) : ∑ j ∈ range M, 1 / ((j : ℝ) + 1) ^ 2 ≤ 2 := by
  have key : ∀ M : ℕ, ∑ j ∈ range (M + 1), 1 / ((j : ℝ) + 1) ^ 2 ≤ 2 - 1 / ((M : ℝ) + 1) := by
    intro M
    induction M with
    | zero => norm_num
    | succ M ih =>
      rw [Finset.sum_range_succ]
      have h : 1 / ((M : ℝ) + 1 + 1) ^ 2 ≤ 1 / ((M : ℝ) + 1) - 1 / ((M : ℝ) + 1 + 1) := by
        rw [div_sub_div _ _ (by positivity) (by positivity),
          div_le_div_iff₀ (by positivity) (by positivity)]
        nlinarith
      push_cast
      linarith
  rcases M with _ | M
  · simp
  · have hpos : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
    linarith [key M]

/-! ## Assembly: SI Proposition S6 -/

/-- Under neutral evolution the fixation probability of the walk of the block length is `1/N`. -/
lemma fixation_probability_neutral (N : ℕ) (hN : 1 ≤ N) (h : ℕ → ℝ) (hh : FixSteps 1 N h) :
    h 1 = 1 / N := by
  have hα : alpha 1 = 1 / 2 := by unfold alpha; norm_num
  have hβ : beta 1 = 1 / 2 := by unfold beta; norm_num
  have pair : ∀ k, k + 1 ≤ N → h k = (k : ℝ) * h 1 ∧ h (k + 1) = ((k : ℝ) + 1) * h 1 := by
    intro k
    induction k with
    | zero =>
      intro _
      exact ⟨by simp [hh.zero], by simp⟩
    | succ k ih =>
      intro hk
      obtain ⟨h1, h2⟩ := ih (by omega)
      refine ⟨by push_cast; exact h2, ?_⟩
      have e := hh.step k (by omega)
      rw [hα, hβ, h1, h2] at e
      show h (k + 2) = (((k + 1 : ℕ) : ℝ) + 1) * h 1
      push_cast
      linarith
  obtain ⟨M, rfl⟩ : ∃ M, N = M + 1 := ⟨N - 1, by omega⟩
  have hM := (pair M le_rfl).2
  rw [hh.top] at hM
  push_cast
  rw [eq_div_iff (by positivity)]
  linarith

/-- `P_N(0) = N`. -/
lemma PNc_at_zero (N : ℕ) : PNc N 0 = N := by
  unfold PNc
  rw [Finset.sum_range_succ']
  simp

/-- `2 min(ℓ, N - ℓ) / N ≤ sin(πℓ/N)` for `0 ≤ ℓ ≤ N`. -/
lemma sin_ge (N ℓ : ℕ) (hN : 1 ≤ N) (hℓ : ℓ ≤ N) :
    2 * (min ℓ (N - ℓ) : ℕ) / (N : ℝ) ≤ Real.sin (π * ℓ / N) := by
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  rcases le_total ℓ (N - ℓ) with hle | hle
  · rw [min_eq_left hle]
    have hx0 : 0 ≤ π * ℓ / N := by positivity
    have hx1 : π * ℓ / N ≤ π / 2 := by
      rw [div_le_div_iff₀ hNpos (by norm_num)]
      have : (2 * ℓ : ℝ) ≤ N := by exact_mod_cast (by omega : 2 * ℓ ≤ N)
      nlinarith [Real.pi_pos]
    have h := Real.mul_le_sin hx0 hx1
    have e : 2 / π * (π * ℓ / N) = 2 * (ℓ : ℝ) / N := by
      field_simp
      try ring
    linarith
  · rw [min_eq_right hle]
    have hsym : Real.sin (π * ℓ / N) = Real.sin (π * ((N - ℓ : ℕ) : ℝ) / N) := by
      rw [← Real.sin_pi_sub]
      congr 1
      push_cast [Nat.cast_sub hℓ]
      field_simp
      try ring
    rw [hsym]
    have hx0 : 0 ≤ π * ((N - ℓ : ℕ) : ℝ) / N := by positivity
    have hx1 : π * ((N - ℓ : ℕ) : ℝ) / N ≤ π / 2 := by
      rw [div_le_div_iff₀ hNpos (by norm_num)]
      have : (2 * (N - ℓ : ℕ) : ℝ) ≤ N := by exact_mod_cast (by omega : 2 * (N - ℓ) ≤ N)
      nlinarith [Real.pi_pos]
    have h := Real.mul_le_sin hx0 hx1
    have e : 2 / π * (π * ((N - ℓ : ℕ) : ℝ) / N) = 2 * ((N - ℓ : ℕ) : ℝ) / N := by
      field_simp
      try ring
    linarith

/-- One term of the sum: `1/|P_N(1 - ω^ℓ)| ≤ (4√2/a) e^(-a) · 120 e / m²` with `a = √(N/2)` and
`m = min(ℓ, N - ℓ)`. -/
lemma term_le (N ℓ : ℕ) (hN : 2 ≤ N) (hℓ1 : 1 ≤ ℓ) (hℓN : ℓ < N) :
    1 / ‖PNc N (1 - exp (2 * π * I * ℓ / N))‖
      ≤ 4 * √2 / √(N / 2) * Real.exp (-√(N / 2))
        * (120 * Real.exp 1 / ((min ℓ (N - ℓ) : ℕ) : ℝ) ^ 2) := by
  set m : ℕ := min ℓ (N - ℓ) with hmdef
  have hm1 : 1 ≤ m := by rw [hmdef]; omega
  have hmR : (1 : ℝ) ≤ m := by exact_mod_cast hm1
  have hNR : (2 : ℝ) ≤ N := by exact_mod_cast hN
  have hNpos : (0 : ℝ) < N := by linarith
  have hN0 : (N : ℝ) ≠ 0 := hNpos.ne'
  have hℓR : (0 : ℝ) < ℓ := by exact_mod_cast hℓ1
  set a := √((N : ℝ) / 2) with hadef
  set b := √(m : ℝ) with hbdef
  have ha2 : a ^ 2 = N / 2 := Real.sq_sqrt (by positivity)
  have hb2 : b ^ 2 = m := Real.sq_sqrt (by positivity)
  have ha1 : 1 ≤ a := by
    rw [hadef, show (1 : ℝ) = √1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt (by linarith)
  have hb1 : 1 ≤ b := by
    rw [hbdef, show (1 : ℝ) = √1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt hmR
  have ha0 : 0 < a := by linarith
  have ha0' : a ≠ 0 := ha0.ne'
  set θ : ℝ := π * ℓ / N with hθdef
  have hθ0 : 0 < θ := div_pos (mul_pos Real.pi_pos hℓR) hNpos
  have hθπ : θ < π := by
    rw [hθdef, div_lt_iff₀ hNpos]
    have : (ℓ : ℝ) < N := by exact_mod_cast hℓN
    nlinarith [Real.pi_pos]
  have hz : exp (2 * π * I * ℓ / N) = exp ((2 * θ : ℝ) * I) := by
    congr 1
    rw [hθdef]
    push_cast
    ring
  rw [hz]
  -- the parameter s₀ = b / a
  have hs₀ : 0 < b / a := by positivity
  have hN2a : (N : ℝ) = 2 * a ^ 2 := by linarith
  have hNb : (N : ℝ) * (b / a) = 2 * a * b := by
    rw [hN2a, ← mul_div_assoc, div_eq_iff ha0']
    ring
  have hNs : 2 ≤ (N : ℝ) * (b / a) := by
    rw [hNb]
    nlinarith [mul_le_mul ha1 hb1 zero_le_one (by linarith : (0 : ℝ) ≤ a)]
  have hsin := sin_ge N ℓ (by omega) hℓN.le
  rw [← hmdef] at hsin
  have hle : b / a ≤ √(Real.sin θ) := by
    rw [show b / a = √((b / a) ^ 2) from (Real.sqrt_sq hs₀.le).symm]
    apply Real.sqrt_le_sqrt
    rw [div_pow, hb2, ha2]
    calc (m : ℝ) / (N / 2) = 2 * m / N := by rw [div_div_eq_mul_div]; ring
      _ ≤ Real.sin θ := hsin
  have hterm := inv_norm_PNc_le N θ (b / a) hθ0 hθπ hs₀ hNs hle
  -- `e^(-N s₀/2) = e^(-ab) ≤ e^(-a) e^(1-b)`
  have hNsa : (N : ℝ) * (b / a) / 2 = a * b := by
    rw [hNb]
    ring
  have hexp : Real.exp (-((N : ℝ) * (b / a) / 2)) ≤ Real.exp (-a) * Real.exp (1 - b) := by
    rw [hNsa, ← Real.exp_add]
    apply Real.exp_le_exp.mpr
    nlinarith [mul_nonneg (sub_nonneg.mpr ha1) (sub_nonneg.mpr hb1)]
  have hbe : b * Real.exp (1 - b) ≤ 120 * Real.exp 1 / (m : ℝ) ^ 2 := by
    have h := sqrt_mul_exp_neg_sqrt_le (m : ℝ) hmR
    rw [← hbdef] at h
    have e1 : Real.exp (1 - b) = Real.exp 1 * Real.exp (-b) := by
      rw [sub_eq_add_neg, Real.exp_add]
    rw [e1]
    have := mul_le_mul_of_nonneg_left h (Real.exp_pos 1).le
    calc b * (Real.exp 1 * Real.exp (-b)) = Real.exp 1 * (b * Real.exp (-b)) := by ring
      _ ≤ Real.exp 1 * (120 / (m : ℝ) ^ 2) := this
      _ = 120 * Real.exp 1 / (m : ℝ) ^ 2 := by ring
  have hpos1 : 0 ≤ 4 * √2 / a := by positivity
  calc 1 / ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖
      ≤ 4 * √2 * (b / a) * Real.exp (-((N : ℝ) * (b / a) / 2)) := hterm
    _ ≤ 4 * √2 * (b / a) * (Real.exp (-a) * Real.exp (1 - b)) := by
        apply mul_le_mul_of_nonneg_left hexp
        positivity
    _ = 4 * √2 / a * Real.exp (-a) * (b * Real.exp (1 - b)) := by
        ring
    _ ≤ 4 * √2 / a * Real.exp (-a) * (120 * Real.exp 1 / (m : ℝ) ^ 2) := by
        apply mul_le_mul_of_nonneg_left hbe
        positivity

/-- `∑_{ℓ=1}^{N-1} 1/min(ℓ, N-ℓ)² ≤ 4`. -/
lemma sum_inv_min_sq_le (N : ℕ) :
    ∑ i ∈ range (N - 1), 1 / ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) ^ 2 ≤ 4 := by
  have hterm : ∀ i ∈ range (N - 1), 1 / ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) ^ 2
      ≤ 1 / ((i : ℝ) + 1) ^ 2 + 1 / (((N - 1 - 1 - i : ℕ) : ℝ) + 1) ^ 2 := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    have h1 : 0 ≤ 1 / ((i : ℝ) + 1) ^ 2 := by positivity
    have h2 : 0 ≤ 1 / (((N - 1 - 1 - i : ℕ) : ℝ) + 1) ^ 2 := by positivity
    rcases le_total (i + 1) (N - (i + 1)) with hle | hle
    · rw [min_eq_left hle]
      push_cast
      linarith
    · rw [min_eq_right hle]
      have e : ((N - (i + 1) : ℕ) : ℝ) = ((N - 1 - 1 - i : ℕ) : ℝ) + 1 := by
        rw [show N - (i + 1) = (N - 1 - 1 - i) + 1 by omega, Nat.cast_add, Nat.cast_one]
      rw [e]
      linarith
  calc ∑ i ∈ range (N - 1), 1 / ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) ^ 2
      ≤ ∑ i ∈ range (N - 1), (1 / ((i : ℝ) + 1) ^ 2 + 1 / (((N - 1 - 1 - i : ℕ) : ℝ) + 1) ^ 2) :=
        Finset.sum_le_sum hterm
    _ = ∑ i ∈ range (N - 1), 1 / ((i : ℝ) + 1) ^ 2
        + ∑ i ∈ range (N - 1), 1 / (((N - 1 - 1 - i : ℕ) : ℝ) + 1) ^ 2 := by
        rw [Finset.sum_add_distrib]
    _ = ∑ i ∈ range (N - 1), 1 / ((i : ℝ) + 1) ^ 2 + ∑ i ∈ range (N - 1), 1 / ((i : ℝ) + 1) ^ 2 := by
        rw [Finset.sum_range_reflect (fun j => 1 / ((j : ℝ) + 1) ^ 2) (N - 1)]
    _ ≤ 2 + 2 := add_le_add (sum_inv_sq_le _) (sum_inv_sq_le _)
    _ = 4 := by norm_num

/-- SI Proposition S6 (neutral evolution), with an explicit constant. Let `q` be the distribution of
the number `H` of head advances conditional on fixation, and let `h` solve the first-step equations
of the fixation probability at `r = 1`. Suppose that at every `N`-th root of unity `z` the
generating function `∑ q(n) z^n` equals `F(1) / h(1)` for a solution `F` of the first-step equations
with `α = β = 1/2`. Then every fixation location probability differs from `1/N` by at most
`3840 e N^(-1/2) e^(-√(N/2))`. -/
theorem neutral_location (N : ℕ) (hN : 2 ≤ N) (q : ℕ → ℝ) (hq : ∀ n, 0 ≤ q n) (hs : Summable q)
    (h : ℕ → ℝ) (hh : FixSteps 1 N h)
    (hgf : ∀ ℓ < N, ∃ F : ℕ → ℂ, GFStepsC (1 / 2) (1 / 2) (exp (2 * π * I * ℓ / N)) N F ∧
      ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N) = F 1 / (h 1 : ℂ))
    (k : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N|
      ≤ 3840 * Real.exp 1 / √N * Real.exp (-√(N / 2)) := by
  have hN1 : 1 ≤ N := by omega
  have hNR : (0 : ℝ) < N := by positivity
  have hNc : (N : ℂ) ≠ 0 := by exact_mod_cast (by omega : N ≠ 0)
  have hρ := fixation_probability_neutral N hN1 h hh
  -- the generating function at every root of unity
  have hE : ∀ ℓ < N, ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N)
      = N * exp (2 * π * I * ℓ / N) ^ (N - 1) / PNc N (1 - exp (2 * π * I * ℓ / N)) := by
    intro ℓ hℓ
    obtain ⟨F, hF, hFE⟩ := hgf ℓ hℓ
    have hgfF := gf_head_advances_complex (1 / 2) (1 / 2) (exp (2 * π * I * ℓ / N)) (by norm_num)
      N hN1 F hF
    rw [hFE, hgfF, hρ]
    have e1 : (2 : ℂ) * (1 / 2) * exp (2 * π * I * ℓ / N) = exp (2 * π * I * ℓ / N) := by ring
    have e2 : (1 : ℂ) - 4 * (1 / 2) * (1 / 2) * exp (2 * π * I * ℓ / N)
        = 1 - exp (2 * π * I * ℓ / N) := by ring
    rw [e1, e2]
    push_cast
    rw [div_div_eq_mul_div, div_one]
    ring
  have hF := fourier_location N hN1 q hq hs k hk1 hkN
  set p : ℝ := ∑' n, if n % N = k - 1 then q n else 0 with hpdef
  -- split off the term `ℓ = 0`
  obtain ⟨M, hMN⟩ : ∃ M, N = M + 1 := ⟨N - 1, by omega⟩
  have hsplit := Finset.sum_range_succ' (fun ℓ : ℕ => exp (-(2 * π * I * ℓ * (k - 1) / N)) *
    ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N)) M
  rw [← hMN] at hsplit
  rw [hsplit] at hF
  beta_reduce at hF
  have h0term : exp (-(2 * π * I * ((0 : ℕ) : ℂ) * (k - 1) / N)) *
      ∑' n, (q n : ℂ) * exp (2 * π * I * ((0 : ℕ) : ℂ) * n / N) = 1 := by
    have := hE 0 (by omega)
    simp only [Nat.cast_zero, mul_zero, zero_mul, zero_div, neg_zero, Complex.exp_zero, one_mul,
      one_pow, sub_self, PNc_at_zero] at this ⊢
    rw [this]
    field_simp
  rw [h0term] at hF
  push_cast at hF
  -- `p - 1/N` as a sum over `ℓ = 1, …, N-1`
  have hdiff : ((p - 1 / N : ℝ) : ℂ) = ∑ i ∈ range M,
      exp (-(2 * π * I * ((i + 1 : ℕ) : ℂ) * (k - 1) / N)) *
        (exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N) ^ (N - 1)
          / PNc N (1 - exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N))) := by
    push_cast
    rw [hF, mul_add, Finset.mul_sum]
    have hsum : ∀ i ∈ range M, 1 / (N : ℂ) * (exp (-(2 * π * I * ((i : ℂ) + 1) * (k - 1) / N)) *
        ∑' n, (q n : ℂ) * exp (2 * π * I * ((i : ℂ) + 1) * n / N))
        = exp (-(2 * π * I * ((i : ℂ) + 1) * (k - 1) / N)) *
          (exp (2 * π * I * ((i : ℂ) + 1) / N) ^ (N - 1)
            / PNc N (1 - exp (2 * π * I * ((i : ℂ) + 1) / N))) := by
      intro i hi
      have hi' := Finset.mem_range.mp hi
      have := hE (i + 1) (by omega)
      push_cast at this
      rw [this]
      field_simp
    rw [Finset.sum_congr rfl hsum]
    ring
  -- norms
  have habs : |p - 1 / N| ≤ ∑ i ∈ range M, 1 / ‖PNc N (1 - exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N))‖ := by
    rw [← Real.norm_eq_abs, ← Complex.norm_real, hdiff]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun i _ => ?_)
    rw [norm_mul, norm_div, norm_pow]
    have hc : ‖exp (-(2 * π * I * ((i + 1 : ℕ) : ℂ) * (k - 1) / N))‖ = 1 := by
      have : -(2 * π * I * ((i + 1 : ℕ) : ℂ) * (k - 1) / N)
          = ((-(2 * π * ((i + 1 : ℕ) : ℝ) * ((k : ℝ) - 1) / N) : ℝ) : ℂ) * I := by
        push_cast
        ring
      rw [this, Complex.norm_exp_ofReal_mul_I]
    have hz : ‖exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N)‖ = 1 := by
      have : 2 * π * I * ((i + 1 : ℕ) : ℂ) / N = ((2 * π * ((i + 1 : ℕ) : ℝ) / N : ℝ) : ℂ) * I := by
        push_cast
        ring
      rw [this, Complex.norm_exp_ofReal_mul_I]
    rw [hc, hz, one_pow, one_mul, one_div]
  -- the term bounds and the sum
  have hterms : ∀ i ∈ range M, 1 / ‖PNc N (1 - exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N))‖
      ≤ 4 * √2 / √(N / 2) * Real.exp (-√(N / 2)) *
        (120 * Real.exp 1 * (1 / ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) ^ 2)) := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    have := term_le N (i + 1) hN (by omega) (by omega)
    calc _ ≤ _ := this
      _ = _ := by ring
  have hsumle := Finset.sum_le_sum hterms
  rw [← Finset.mul_sum, ← Finset.mul_sum] at hsumle
  have hmin := sum_inv_min_sq_le N
  have hNM : N - 1 = M := by omega
  rw [hNM] at hmin
  have hcoef : 4 * √2 / √((N : ℝ) / 2) = 8 / √N := by
    have h2 : √2 * √2 = (2 : ℝ) := Real.mul_self_sqrt (by norm_num)
    rw [Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 2), div_div_eq_mul_div, mul_assoc, h2]
    norm_num
  calc |p - 1 / N| ≤ _ := habs
    _ ≤ _ := hsumle
    _ ≤ 4 * √2 / √((N : ℝ) / 2) * Real.exp (-√(N / 2)) * (120 * Real.exp 1 * 4) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        apply mul_le_mul_of_nonneg_left hmin (by positivity)
    _ = 3840 * Real.exp 1 / √N * Real.exp (-√(N / 2)) := by
        rw [hcoef]
        ring

end

end FixLoc
