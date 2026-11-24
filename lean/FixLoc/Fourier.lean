import FixLoc.Walk

/-!
# The directed cycle: SI Lemma 2 for complex `z` and the Fourier formula

`Walk.lean` proves the generating function of the number `H` of head advances for real `z > 0`.
This file proves the same formula for every complex `z` (`gf_head_advances_complex`) and the finite
Fourier formula for the fixation location probabilities (`fourier_location`).

* `gf_head_advances_complex`: every solution `F` of the first-step equations with complex `z`
  satisfies `F 1 = (2αz)^(N-1) / P_N(1 - 4αβz)`.
* `fourier_location`: for every distribution `q` of `H` on `ℕ` and every vertex `k ∈ {1, …, N}`,
  `P(H mod N = k - 1) = (1/N) ∑_ℓ e^(-2πiℓ(k-1)/N) E[e^(2πiℓH/N)]`.
-/

open Finset Complex
open scoped Real

namespace FixLoc

noncomputable section

/-! ## The polynomial `P_N` with complex argument -/

/-- `P_N(y) = ∑_m C(N, 2m+1) y^m` for complex `y`. -/
def PNc (N : ℕ) (y : ℂ) : ℂ :=
  ∑ m ∈ range (N + 1), (N.choose (2 * m + 1) : ℂ) * y ^ m

lemma PNc_eq_sum_range (N R : ℕ) (hR : N + 1 ≤ R) (y : ℂ) :
    PNc N y = ∑ m ∈ range R, (N.choose (2 * m + 1) : ℂ) * y ^ m := by
  unfold PNc
  apply Finset.sum_subset (fun m hm => Finset.mem_range.mpr (by have := Finset.mem_range.mp hm; omega))
  intro m _ hmN
  have hlt : N < 2 * m + 1 := by
    simp only [Finset.mem_range, not_lt] at hmN
    omega
  simp [Nat.choose_eq_zero_of_lt hlt]

lemma choose_two_step_c (n m : ℕ) :
    ((n + 2).choose (2 * m + 3) : ℂ)
      = 2 * ((n + 1).choose (2 * m + 3) : ℂ) - (n.choose (2 * m + 3) : ℂ)
        + (n.choose (2 * m + 1) : ℂ) := by
  have h1 : (n + 2).choose (2 * m + 3) = (n + 1).choose (2 * m + 2) + (n + 1).choose (2 * m + 3) := by
    have := Nat.choose_succ_succ (n + 1) (2 * m + 2)
    simpa [Nat.add_assoc] using this
  have h2 : (n + 1).choose (2 * m + 2) = n.choose (2 * m + 1) + n.choose (2 * m + 2) := by
    have := Nat.choose_succ_succ n (2 * m + 1)
    simpa [Nat.add_assoc] using this
  have h3 : (n + 1).choose (2 * m + 3) = n.choose (2 * m + 2) + n.choose (2 * m + 3) := by
    have := Nat.choose_succ_succ n (2 * m + 2)
    simpa [Nat.add_assoc] using this
  rw [h1, h2, h3]
  push_cast
  ring

lemma PNc_succ_succ (n : ℕ) (y : ℂ) :
    PNc (n + 2) y = 2 * PNc (n + 1) y - (1 - y) * PNc n y := by
  rw [PNc_eq_sum_range (n + 2) ((n + 2) + 1) le_rfl,
    PNc_eq_sum_range (n + 1) ((n + 2) + 1) (by omega), PNc_eq_sum_range n ((n + 2) + 1) (by omega)]
  have key : ∑ m ∈ range ((n + 2) + 1),
        (((n + 2).choose (2 * m + 1) : ℂ) - 2 * ((n + 1).choose (2 * m + 1) : ℂ)
          + (n.choose (2 * m + 1) : ℂ)) * y ^ m
      = ∑ m ∈ range ((n + 2) + 1), ((n.choose (2 * m + 1) : ℂ) * y ^ m) * y := by
    conv_lhs => rw [Finset.sum_range_succ']
    conv_rhs => rw [Finset.sum_range_succ]
    have hz : (n.choose (2 * (n + 2) + 1) : ℂ) = 0 := by
      rw [Nat.choose_eq_zero_of_lt (by omega)]
      simp
    have h0 : (((n + 2).choose (2 * 0 + 1) : ℂ) - 2 * ((n + 1).choose (2 * 0 + 1) : ℂ)
          + (n.choose (2 * 0 + 1) : ℂ)) = 0 := by
      simp only [Nat.mul_zero, Nat.zero_add, Nat.choose_one_right]
      push_cast
      ring
    rw [hz, h0, zero_mul, zero_mul, zero_mul, add_zero, add_zero]
    apply Finset.sum_congr rfl
    intro m _
    have e : 2 * (m + 1) + 1 = 2 * m + 3 := by ring
    rw [e, choose_two_step_c n m, pow_succ]
    ring
  have expand : ∑ m ∈ range ((n + 2) + 1), ((n + 2).choose (2 * m + 1) : ℂ) * y ^ m
      = 2 * ∑ m ∈ range ((n + 2) + 1), ((n + 1).choose (2 * m + 1) : ℂ) * y ^ m
        - ∑ m ∈ range ((n + 2) + 1), (n.choose (2 * m + 1) : ℂ) * y ^ m
        + ∑ m ∈ range ((n + 2) + 1), ((n.choose (2 * m + 1) : ℂ) * y ^ m) * y := by
    rw [← key, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro m _
    ring
  rw [expand, ← Finset.sum_mul]
  ring

lemma PNc_zero (y : ℂ) : PNc 0 y = 0 := by
  simp [PNc]

lemma PNc_one (y : ℂ) : PNc 1 y = 1 := by
  simp [PNc, Finset.sum_range_succ, Nat.choose_eq_zero_of_lt (show 1 < 3 by norm_num)]

/-! ## Second-order recurrences over `ℂ` -/

/-- The solution of `u (k+2) = (u (k+1) - b u k) / a` with `u 0 = 0` and `u 1 = 1`. -/
def canonC (a b : ℂ) : ℕ → ℂ
  | 0 => 0
  | 1 => 1
  | (k + 2) => (canonC a b (k + 1) - b * canonC a b k) / a

lemma eq_mul_canonC (a b : ℂ) (ha : a ≠ 0) (N : ℕ) (G : ℕ → ℂ) (h0 : G 0 = 0)
    (hstep : ∀ k, k + 2 ≤ N → G (k + 1) = a * G (k + 2) + b * G k) :
    ∀ x, x ≤ N → G x = G 1 * canonC a b x := by
  have pair : ∀ k, k + 1 ≤ N →
      G k = G 1 * canonC a b k ∧ G (k + 1) = G 1 * canonC a b (k + 1) := by
    intro k
    induction k with
    | zero =>
      intro _
      exact ⟨by simp [canonC, h0], by simp [canonC]⟩
    | succ k ih =>
      intro hk
      obtain ⟨h1, h2⟩ := ih (by omega)
      refine ⟨h2, ?_⟩
      have e := hstep k (by omega)
      rw [h1, h2] at e
      show G (k + 2) = G 1 * canonC a b (k + 2)
      rw [show canonC a b (k + 2) = (canonC a b (k + 1) - b * canonC a b k) / a from rfl,
        mul_div_assoc', eq_div_iff ha]
      linear_combination -e
  intro x hx
  rcases Nat.eq_zero_or_pos x with rfl | hpos
  · simp [canonC, h0]
  · have := (pair (x - 1) (by omega)).2
    rwa [Nat.sub_add_cancel hpos] at this

lemma canon_closedC (α β z : ℂ) (hα : α ≠ 0) (hz : z ≠ 0) :
    ∀ x : ℕ, canonC (α * z) β x = 2 * α * z * PNc x (1 - 4 * α * β * z) / (2 * α * z) ^ x := by
  have hc : (2 * α * z) ≠ 0 := by
    have : (2 : ℂ) ≠ 0 := by norm_num
    exact mul_ne_zero (mul_ne_zero this hα) hz
  have pair : ∀ k : ℕ,
      canonC (α * z) β k = 2 * α * z * PNc k (1 - 4 * α * β * z) / (2 * α * z) ^ k ∧
      canonC (α * z) β (k + 1)
        = 2 * α * z * PNc (k + 1) (1 - 4 * α * β * z) / (2 * α * z) ^ (k + 1) := by
    intro k
    induction k with
    | zero =>
      refine ⟨by simp [canonC, PNc_zero], ?_⟩
      simp only [zero_add, pow_one, PNc_one]
      rw [show canonC (α * z) β 1 = 1 from rfl]
      field_simp
    | succ k ih =>
      obtain ⟨h1, h2⟩ := ih
      refine ⟨h2, ?_⟩
      show canonC (α * z) β (k + 2) = _
      rw [show canonC (α * z) β (k + 2)
          = (canonC (α * z) β (k + 1) - β * canonC (α * z) β k) / (α * z) from rfl,
        h1, h2, show k + 1 + 1 = k + 2 from rfl, PNc_succ_succ k]
      field_simp
      ring
  exact fun x => (pair x).1

/-! ## SI Lemma 2 for complex `z` -/

/-- First-step equations of `F x = E_x[z^H ; fixation]` for complex `z`. -/
structure GFStepsC (α β z : ℂ) (N : ℕ) (F : ℕ → ℂ) : Prop where
  zero : F 0 = 0
  top : F N = 1
  step : ∀ k, k + 2 ≤ N → F (k + 1) = α * z * F (k + 2) + β * F k

/-- SI Lemma 2, Eq. (cycle-pgf), for every complex `z`: `E_1[z^H ; fixation] = (2αz)^(N-1) / P_N(1 - 4αβz)`. -/
theorem gf_head_advances_complex (α β z : ℂ) (hα : α ≠ 0) (N : ℕ) (hN : 1 ≤ N)
    (F : ℕ → ℂ) (hF : GFStepsC α β z N F) :
    F 1 = (2 * α * z) ^ (N - 1) / PNc N (1 - 4 * α * β * z) := by
  by_cases hz : z = 0
  · subst hz
    rcases Nat.lt_or_ge N 2 with hN2 | hN2
    · have hN1 : N = 1 := by omega
      subst hN1
      simp [hF.top, PNc_one]
    · have hF1 : F 1 = 0 := by
        have := hF.step 0 (by omega)
        simpa [hF.zero] using this
      rw [hF1]
      have h0 : (2 * α * 0 : ℂ) ^ (N - 1) = 0 := by
        rw [mul_zero]
        exact zero_pow (by omega)
      rw [h0, zero_div]
  · have ha : α * z ≠ 0 := mul_ne_zero hα hz
    have hsol := eq_mul_canonC (α * z) β ha N F hF.zero hF.step N le_rfl
    rw [hF.top, canon_closedC α β z hα hz N] at hsol
    have hc : (2 * α * z) ≠ 0 := by
      have : (2 : ℂ) ≠ 0 := by norm_num
      exact mul_ne_zero (mul_ne_zero this hα) hz
    have hP : PNc N (1 - 4 * α * β * z) ≠ 0 := by
      intro h
      rw [h] at hsol
      simp at hsol
    obtain ⟨M, rfl⟩ : ∃ M, N = M + 1 := ⟨N - 1, by omega⟩
    simp only [Nat.add_sub_cancel]
    rw [eq_div_iff hP]
    rw [pow_succ, mul_comm ((2 * α * z) ^ M) (2 * α * z), mul_div_mul_left _ _ hc, mul_div_assoc',
      eq_div_iff (pow_ne_zero _ hc), one_mul] at hsol
    exact hsol.symm

/-! ## The Fourier formula (SI Lemma 2, Eq. (cycle-fourier)) -/

/-- `ω = e^(2πi/N)`. -/
def rootN (N : ℕ) : ℂ := exp (2 * π * I / N)

lemma rootN_prim (N : ℕ) (hN : N ≠ 0) : IsPrimitiveRoot (rootN N) N :=
  isPrimitiveRoot_exp N hN

/-- Orthogonality of the characters of `ℤ/N`. -/
lemma sum_rootN_pow (N : ℕ) (hN : N ≠ 0) (j : ℕ) :
    ∑ ℓ ∈ range N, rootN N ^ (ℓ * j) = if N ∣ j then (N : ℂ) else 0 := by
  have hp := rootN_prim N hN
  split_ifs with hdvd
  · have h1 : rootN N ^ j = 1 := (hp.pow_eq_one_iff_dvd j).mpr hdvd
    have hone : ∀ ℓ ∈ range N, rootN N ^ (ℓ * j) = 1 := fun ℓ _ => by
      rw [mul_comm, pow_mul, h1, one_pow]
    rw [Finset.sum_congr rfl hone]
    simp
  · have h1 : rootN N ^ j ≠ 1 := fun h => hdvd ((hp.pow_eq_one_iff_dvd j).mp h)
    have hgeom := geom_sum_eq h1 N
    have hN1 : (rootN N ^ j) ^ N = 1 := by
      rw [← pow_mul, mul_comm, pow_mul, hp.pow_eq_one, one_pow]
    rw [hN1, sub_self, zero_div] at hgeom
    rw [← hgeom]
    apply Finset.sum_congr rfl
    intro ℓ _
    rw [← pow_mul, mul_comm]

lemma dvd_iff_mod (N k h : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    N ∣ h + N + 1 - k ↔ h % N = k - 1 := by
  have e : h + N + 1 - k = (h + N) - (k - 1) := by omega
  rw [e, ← Nat.modEq_iff_dvd' (by omega : k - 1 ≤ h + N)]
  unfold Nat.ModEq
  rw [Nat.add_mod_right, Nat.mod_eq_of_lt (by omega : k - 1 < N)]
  exact eq_comm

/-- SI Lemma 2, Eq. (cycle-fourier): for every distribution `q` of `H` on `ℕ`, the fixation location
`1 + (H mod N)` is vertex `k` with the probability given by the finite Fourier sum. -/
theorem fourier_location (N : ℕ) (hN : 1 ≤ N) (q : ℕ → ℝ) (hq : ∀ h, 0 ≤ q h) (hs : Summable q)
    (k : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    ((∑' h, (if h % N = k - 1 then q h else 0) : ℝ) : ℂ) =
      (1 / N) * ∑ ℓ ∈ range N, exp (-(2 * π * I * ℓ * (k - 1) / N)) *
        ∑' h, (q h : ℂ) * exp (2 * π * I * ℓ * h / N) := by
  have hN0 : N ≠ 0 := by omega
  have hNc : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr hN0
  have phase : ∀ ℓ h : ℕ, exp (-(2 * π * I * ℓ * (k - 1) / N)) * exp (2 * π * I * ℓ * h / N)
      = rootN N ^ (ℓ * (h + N + 1 - k)) := by
    intro ℓ h
    have hcast : ((ℓ * (h + N + 1 - k) : ℕ) : ℂ) = (ℓ : ℂ) * ((h : ℂ) + N + 1 - k) := by
      push_cast [Nat.cast_sub (show k ≤ h + N + 1 by omega)]
      try ring
    have harg : (ℓ : ℂ) * ((h : ℂ) + N + 1 - k) * (2 * π * I / N)
        = (-(2 * π * I * ℓ * (k - 1) / N) + 2 * π * I * ℓ * h / N) + ℓ * (2 * π * I) := by
      field_simp
      ring
    rw [rootN, ← exp_nat_mul, hcast, harg,
      exp_add (-(2 * π * I * ℓ * (k - 1) / N) + 2 * π * I * ℓ * h / N),
      exp_nat_mul_two_pi_mul_I, mul_one, exp_add]
  have hsum : ∀ ℓ : ℕ, Summable (fun h : ℕ => (q h : ℂ) * rootN N ^ (ℓ * (h + N + 1 - k))) := by
    intro ℓ
    apply Summable.of_norm_bounded hs
    intro h
    apply le_of_eq
    rw [norm_mul, norm_pow, (rootN_prim N hN0).norm'_eq_one hN0, one_pow, mul_one, Complex.norm_real,
      Real.norm_of_nonneg (hq h)]
  have step1 : ∀ ℓ : ℕ, exp (-(2 * π * I * ℓ * (k - 1) / N)) * ∑' h, (q h : ℂ) * exp (2 * π * I * ℓ * h / N)
      = ∑' h, (q h : ℂ) * rootN N ^ (ℓ * (h + N + 1 - k)) := by
    intro ℓ
    rw [← tsum_mul_left]
    apply tsum_congr
    intro h
    rw [← phase ℓ h]
    ring
  rw [Finset.sum_congr rfl (fun ℓ _ => step1 ℓ), ← Summable.tsum_finsetSum (fun ℓ _ => hsum ℓ)]
  have step2 : ∀ h : ℕ, ∑ ℓ ∈ range N, (q h : ℂ) * rootN N ^ (ℓ * (h + N + 1 - k))
      = (N : ℂ) * ((if h % N = k - 1 then q h else 0 : ℝ) : ℂ) := by
    intro h
    rw [← Finset.mul_sum, sum_rootN_pow N hN0]
    by_cases hd : h % N = k - 1
    · rw [if_pos ((dvd_iff_mod N k h hk1 hkN).mpr hd), if_pos hd]
      ring
    · rw [if_neg (fun hdv => hd ((dvd_iff_mod N k h hk1 hkN).mp hdv)), if_neg hd]
      simp
  rw [tsum_congr step2, tsum_mul_left, Complex.ofReal_tsum, ← mul_assoc, one_div_mul_cancel hNc, one_mul]

end

end FixLoc
