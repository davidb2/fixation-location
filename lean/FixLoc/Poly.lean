import Mathlib

/-!
# The polynomial `P_N` of the directed-cycle section

`P_N(y) = ∑_m C(N, 2m+1) y^m` appears in Lemma 2 of the SI of "Fixation location in structured
populations". This file proves the three facts about `P_N` that the walk proofs use:
the three-term recurrence `P_{n+2} = 2 P_{n+1} - (1 - y) P_n`, the first two values, and positivity
for `y ≥ 0`.
-/

open Finset

namespace FixLoc

/-- `P_N(y) = ∑_m C(N, 2m+1) y^m` (SI, Lemma 2). Terms with `2m + 1 > N` vanish, so the range
`m ≤ N` contains every nonzero term. -/
noncomputable def PN (N : ℕ) (y : ℝ) : ℝ :=
  ∑ m ∈ range (N + 1), (N.choose (2 * m + 1) : ℝ) * y ^ m

/-- The sum defining `P_N` may run over any longer range. -/
lemma PN_eq_sum_range (N R : ℕ) (hR : N + 1 ≤ R) (y : ℝ) :
    PN N y = ∑ m ∈ range R, (N.choose (2 * m + 1) : ℝ) * y ^ m := by
  unfold PN
  apply Finset.sum_subset (fun m hm => Finset.mem_range.mpr (by have := Finset.mem_range.mp hm; omega))
  intro m _ hmN
  have hlt : N < 2 * m + 1 := by
    simp only [Finset.mem_range, not_lt] at hmN
    omega
  simp [Nat.choose_eq_zero_of_lt hlt]

/-- Pascal's rule applied twice, in the form used by the recurrence of `P_N`. -/
lemma choose_two_step (n m : ℕ) :
    ((n + 2).choose (2 * m + 3) : ℝ)
      = 2 * ((n + 1).choose (2 * m + 3) : ℝ) - (n.choose (2 * m + 3) : ℝ)
        + (n.choose (2 * m + 1) : ℝ) := by
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

/-- The three-term recurrence `P_{n+2}(y) = 2 P_{n+1}(y) - (1 - y) P_n(y)`. -/
lemma PN_succ_succ (n : ℕ) (y : ℝ) :
    PN (n + 2) y = 2 * PN (n + 1) y - (1 - y) * PN n y := by
  rw [PN_eq_sum_range (n + 2) ((n + 2) + 1) le_rfl, PN_eq_sum_range (n + 1) ((n + 2) + 1) (by omega),
    PN_eq_sum_range n ((n + 2) + 1) (by omega)]
  -- the coefficientwise identity, summed
  have key : ∑ m ∈ range ((n + 2) + 1),
        (((n + 2).choose (2 * m + 1) : ℝ) - 2 * ((n + 1).choose (2 * m + 1) : ℝ)
          + (n.choose (2 * m + 1) : ℝ)) * y ^ m
      = ∑ m ∈ range ((n + 2) + 1), ((n.choose (2 * m + 1) : ℝ) * y ^ m) * y := by
    conv_lhs => rw [Finset.sum_range_succ']
    conv_rhs => rw [Finset.sum_range_succ]
    have hz : (n.choose (2 * (n + 2) + 1) : ℝ) = 0 := by
      rw [Nat.choose_eq_zero_of_lt (by omega)]
      simp
    have h0 : (((n + 2).choose (2 * 0 + 1) : ℝ) - 2 * ((n + 1).choose (2 * 0 + 1) : ℝ)
          + (n.choose (2 * 0 + 1) : ℝ)) = 0 := by
      simp only [Nat.mul_zero, Nat.zero_add, Nat.choose_one_right]
      push_cast
      ring
    rw [hz, h0, zero_mul, zero_mul, zero_mul, add_zero, add_zero]
    apply Finset.sum_congr rfl
    intro m _
    have e : 2 * (m + 1) + 1 = 2 * m + 3 := by ring
    rw [e, choose_two_step n m, pow_succ]
    ring
  have expand : ∑ m ∈ range ((n + 2) + 1), ((n + 2).choose (2 * m + 1) : ℝ) * y ^ m
      = 2 * ∑ m ∈ range ((n + 2) + 1), ((n + 1).choose (2 * m + 1) : ℝ) * y ^ m
        - ∑ m ∈ range ((n + 2) + 1), (n.choose (2 * m + 1) : ℝ) * y ^ m
        + ∑ m ∈ range ((n + 2) + 1), ((n.choose (2 * m + 1) : ℝ) * y ^ m) * y := by
    rw [← key, Finset.mul_sum, ← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro m _
    ring
  rw [expand, ← Finset.sum_mul]
  ring

lemma PN_zero (y : ℝ) : PN 0 y = 0 := by
  simp [PN]

lemma PN_one (y : ℝ) : PN 1 y = 1 := by
  simp [PN, Finset.sum_range_succ, Nat.choose_eq_zero_of_lt (show 1 < 3 by norm_num)]

/-- `P_N(y) > 0` for `N ≥ 1` and `y ≥ 0`: the term `m = 0` is `N` and the other terms are
nonnegative. -/
lemma PN_pos (N : ℕ) (hN : 1 ≤ N) (y : ℝ) (hy : 0 ≤ y) : 0 < PN N y := by
  unfold PN
  rw [Finset.sum_range_succ']
  have h0 : ((N.choose (2 * 0 + 1) : ℝ) * y ^ 0) = N := by simp
  rw [h0]
  have hsum : 0 ≤ ∑ m ∈ range N, ((N.choose (2 * (m + 1) + 1) : ℝ) * y ^ (m + 1)) :=
    Finset.sum_nonneg (fun m _ => mul_nonneg (by positivity) (pow_nonneg hy _))
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  linarith

end FixLoc
