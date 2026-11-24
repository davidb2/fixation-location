import FixLoc.NeutralBound

/-!
# The constant of SI Proposition 6

The SI proof of Proposition 6 gives `|p_k - 1/N| ≤ 16 C N^(-1/2) e^(-√(N/2))` with
`C = ∑_{ℓ ≥ 1} √ℓ e^(1-√ℓ) < 10.6`. This file proves `C < 10.6` (`Cneu_lt`), the bound with
`16 C` (`neutral_location_sharp`), and the bound with `170` (`neutral_location_170`).

The bound on `C` adds upper bounds on the first 30 terms (`fC_le_1` to `fC_le_30`) and a bound on
the remaining terms. The remaining terms telescope against `G(x) = 2 e^(1-√x) (x + 2√x + 2)`, an
antiderivative of `-√x e^(1-√x)` (`fC_le_Gt_sub`, `tail_sum_le`). Each term bound uses rational
bounds on `√ℓ`, the bounds `2.71828 < e < 2.7182819`, and the Taylor polynomial of `exp` of degree 7.
-/

open Finset Complex
open scoped Real

namespace FixLoc

noncomputable section

/-! ## The summand and the telescoping bound -/

/-- `f(x) = √x e^(1-√x)`, the summand of the constant `C`. -/
def fC (x : ℝ) : ℝ := √x * Real.exp (1 - √x)

lemma fC_nonneg (x : ℝ) : 0 ≤ fC x := mul_nonneg (Real.sqrt_nonneg x) (Real.exp_pos _).le

/-- `G(x) = 2 e^(1-√x) (x + 2√x + 2)`, with `G' = -f`. -/
def Gt (x : ℝ) : ℝ := 2 * Real.exp (1 - √x) * (x + 2 * √x + 2)

lemma Gt_nonneg (x : ℝ) (hx : 0 ≤ x) : 0 ≤ Gt x := by
  unfold Gt
  have := Real.sqrt_nonneg x
  exact mul_nonneg (mul_nonneg (by norm_num) (Real.exp_pos _).le) (by linarith)

/-- One telescoping step: `f(x+1) ≤ G(x) - G(x+1)` for `x ≥ 1`. -/
lemma fC_le_Gt_sub (x : ℝ) (hx : 1 ≤ x) : fC (x + 1) ≤ Gt x - Gt (x + 1) := by
  have ha1 : 1 ≤ √x := by
    rw [show (1 : ℝ) = √1 from Real.sqrt_one.symm]
    exact Real.sqrt_le_sqrt hx
  have ha2 : √x ^ 2 = x := Real.sq_sqrt (by linarith)
  have hb2 : √(x + 1) ^ 2 = x + 1 := Real.sq_sqrt (by linarith)
  have hab : √x ≤ √(x + 1) := Real.sqrt_le_sqrt (by linarith)
  unfold fC Gt
  set a := √x with hadef
  set b := √(x + 1) with hbdef
  have ht0 : 0 ≤ b - a := by linarith
  have hQ : 1 + (b - a) + (b - a) ^ 2 / 2 + (b - a) ^ 3 / 6 ≤ Real.exp (b - a) := by
    have h := Real.sum_le_exp_of_nonneg ht0 4
    norm_num [Finset.sum_range_succ, Nat.factorial] at h
    linarith
  have hea : Real.exp (1 - a) = Real.exp (1 - b) * Real.exp (b - a) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hE : 0 < Real.exp (1 - b) := Real.exp_pos _
  have hid : 2 * (1 + (b - a) + (b - a) ^ 2 / 2 + (b - a) ^ 3 / 6) * (a ^ 2 + 2 * a + 2)
      - 2 * (b ^ 2 + 2 * b + 2) - b
      = (b - a) ^ 2 * (a ^ 2 - a) + (b - a) ^ 3 * (a ^ 2 + 2 * a - 1) / 3
        + b * (b ^ 2 - a ^ 2 - 1) := by
    ring
  have hred : 0 ≤ (b - a) ^ 2 * (a ^ 2 - a) + (b - a) ^ 3 * (a ^ 2 + 2 * a - 1) / 3 := by
    have h1 : 0 ≤ a ^ 2 - a := by nlinarith
    have h2 : 0 ≤ a ^ 2 + 2 * a - 1 := by nlinarith
    have h3 := mul_nonneg (sq_nonneg (b - a)) h1
    have h4 := mul_nonneg (pow_nonneg ht0 3) h2
    linarith
  have hcon : b ^ 2 - a ^ 2 - 1 = 0 := by
    rw [hb2, ha2]
    ring
  have hpoly : b ≤ 2 * (1 + (b - a) + (b - a) ^ 2 / 2 + (b - a) ^ 3 / 6) * (a ^ 2 + 2 * a + 2)
      - 2 * (b ^ 2 + 2 * b + 2) := by
    rw [hcon, mul_zero, add_zero] at hid
    linarith
  rw [ha2, hb2] at hpoly
  rw [hea]
  have hP : 0 ≤ x + 2 * a + 2 := by linarith
  have h1 : 2 * (Real.exp (1 - b) * (1 + (b - a) + (b - a) ^ 2 / 2 + (b - a) ^ 3 / 6))
        * (x + 2 * a + 2)
      ≤ 2 * (Real.exp (1 - b) * Real.exp (b - a)) * (x + 2 * a + 2) := by
    apply mul_le_mul_of_nonneg_right _ hP
    apply mul_le_mul_of_nonneg_left _ (by norm_num)
    exact mul_le_mul_of_nonneg_left hQ hE.le
  have h2 := mul_le_mul_of_nonneg_left hpoly hE.le
  linarith [h1, h2]

/-- `∑_{j<n} f(L + j + 1) ≤ G(L) - G(L + n)` for `L ≥ 1`. -/
lemma tail_sum_le (L : ℕ) (hL : 1 ≤ L) (n : ℕ) :
    ∑ j ∈ range n, fC ((L : ℝ) + j + 1) ≤ Gt L - Gt ((L : ℝ) + n) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ,
      show ((L : ℝ) + ((n + 1 : ℕ) : ℝ)) = (L : ℝ) + n + 1 by push_cast; ring]
    have hx : (1 : ℝ) ≤ (L : ℝ) + n := by
      have h1 : (1 : ℝ) ≤ L := by exact_mod_cast hL
      have h2 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
      linarith
    have h := fC_le_Gt_sub ((L : ℝ) + n) hx
    linarith

/-! ## Certified bounds on single terms -/

/-- `2.71828^n T₈(δ) ≤ e^(n + δ)`, with `T₈` the Taylor polynomial of `exp` of degree 7. -/
lemma exp_lower (lo δ : ℝ) (n : ℕ) (hsplit : lo = n + δ) (hδ : 0 ≤ δ) :
    (2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) ≤ Real.exp lo := by
  have hn : Real.exp (n : ℝ) = Real.exp 1 ^ n := by
    rw [← Real.exp_nat_mul, mul_one]
  rw [hsplit, Real.exp_add, hn]
  have he : (2.71828 : ℝ) ≤ Real.exp 1 := by
    have := Real.exp_one_gt_d9
    linarith
  have h1 : (2.71828 : ℝ) ^ n ≤ Real.exp 1 ^ n := pow_le_pow_left₀ (by norm_num) he n
  have h2 := Real.sum_le_exp_of_nonneg hδ 8
  have h3 : 0 ≤ ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) :=
    Finset.sum_nonneg (fun i _ => div_nonneg (pow_nonneg hδ i) (Nat.cast_nonneg _))
  exact mul_le_mul h1 h2 h3 (by positivity)

lemma taylor_pos (δ : ℝ) (n : ℕ) (hδ : 0 ≤ δ) :
    0 < (2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) := by
  have hS1 : (1 : ℝ) ≤ ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) := by
    have h := Finset.single_le_sum (f := fun i => δ ^ i / (Nat.factorial i : ℝ))
      (fun i _ => div_nonneg (pow_nonneg hδ i) (Nat.cast_nonneg _))
      (Finset.mem_range.mpr (by norm_num : 0 < 8))
    simpa using h
  exact mul_pos (by positivity) (lt_of_lt_of_le one_pos hS1)

/-- `e^(1-√x) ≤ 2.7182819 / (2.71828^n T₈(δ))` when `lo = n + δ ≤ √x`. -/
lemma exp_one_sub_sqrt_le (x lo δ : ℝ) (n : ℕ) (hlo0 : 0 ≤ lo) (hlo : lo ^ 2 ≤ x)
    (hsplit : lo = n + δ) (hδ : 0 ≤ δ) :
    Real.exp (1 - √x)
      ≤ 2.7182819 / ((2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ)) := by
  have h2 : lo ≤ √x := by
    calc lo = √(lo ^ 2) := (Real.sqrt_sq hlo0).symm
      _ ≤ √x := Real.sqrt_le_sqrt hlo
  have hEpos := taylor_pos δ n hδ
  have hE := exp_lower lo δ n hsplit hδ
  have he1 : Real.exp 1 ≤ 2.7182819 := by
    have := Real.exp_one_lt_d9
    linarith
  calc Real.exp (1 - √x) ≤ Real.exp (1 - lo) := Real.exp_le_exp.mpr (by linarith)
    _ = Real.exp 1 / Real.exp lo := by rw [Real.exp_sub]
    _ ≤ 2.7182819 / ((2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ)) := by
        rw [div_le_div_iff₀ (Real.exp_pos lo) hEpos]
        exact mul_le_mul he1 hE hEpos.le (by norm_num)

/-- A certified bound `f(x) ≤ d` from `lo ≤ √x ≤ hi`, `lo = n + δ`, and one rational inequality. -/
lemma term_cert (x lo hi δ d : ℝ) (n : ℕ) (hlo0 : 0 ≤ lo) (hhi0 : 0 ≤ hi) (hlo : lo ^ 2 ≤ x)
    (hhi : x ≤ hi ^ 2) (hsplit : lo = n + δ) (hδ : 0 ≤ δ)
    (hd : hi * 2.7182819
      ≤ d * ((2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ))) :
    fC x ≤ d := by
  have h1 : √x ≤ hi := by
    calc √x ≤ √(hi ^ 2) := Real.sqrt_le_sqrt hhi
      _ = hi := Real.sqrt_sq hhi0
  have h3 := exp_one_sub_sqrt_le x lo δ n hlo0 hlo hsplit hδ
  have hEpos := taylor_pos δ n hδ
  set E := (2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) with hEdef
  unfold fC
  calc √x * Real.exp (1 - √x) ≤ hi * (2.7182819 / E) :=
        mul_le_mul h1 h3 (Real.exp_pos _).le hhi0
    _ ≤ d := by
        rw [← mul_div_assoc, div_le_iff₀ hEpos]
        exact hd

/-- A certified bound `G(x) ≤ d`. -/
lemma Gt_cert (x lo hi δ d : ℝ) (n : ℕ) (hlo0 : 0 ≤ lo) (hhi0 : 0 ≤ hi) (hlo : lo ^ 2 ≤ x)
    (hhi : x ≤ hi ^ 2) (hsplit : lo = n + δ) (hδ : 0 ≤ δ)
    (hd : 2 * 2.7182819 * (x + 2 * hi + 2)
      ≤ d * ((2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ))) :
    Gt x ≤ d := by
  have hx0 : 0 ≤ x := le_trans (sq_nonneg lo) hlo
  have h1 : √x ≤ hi := by
    calc √x ≤ √(hi ^ 2) := Real.sqrt_le_sqrt hhi
      _ = hi := Real.sqrt_sq hhi0
  have h3 := exp_one_sub_sqrt_le x lo δ n hlo0 hlo hsplit hδ
  have hEpos := taylor_pos δ n hδ
  set E := (2.71828 : ℝ) ^ n * ∑ i ∈ range 8, δ ^ i / (Nat.factorial i : ℝ) with hEdef
  have hA : 0 ≤ x + 2 * √x + 2 := by
    have := Real.sqrt_nonneg x
    linarith
  have hB : x + 2 * √x + 2 ≤ x + 2 * hi + 2 := by linarith
  have hC : 0 ≤ 2 * (2.7182819 / E) := mul_nonneg (by norm_num) (div_nonneg (by norm_num) hEpos.le)
  unfold Gt
  calc 2 * Real.exp (1 - √x) * (x + 2 * √x + 2)
      ≤ 2 * (2.7182819 / E) * (x + 2 * hi + 2) := by
        apply mul_le_mul _ hB hA hC
        exact mul_le_mul_of_nonneg_left h3 (by norm_num)
    _ = 2 * 2.7182819 * (x + 2 * hi + 2) / E := by ring
    _ ≤ d := by
        rw [div_le_iff₀ hEpos]
        exact hd

lemma fC_le_1 : fC 1 ≤ 1.00001 :=
  term_cert 1 1 1 0 1.00001 1 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_2 : fC 2 ≤ 0.93467 :=
  term_cert 2 1.4142 1.4143 0.4142 0.93467 1 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_3 : fC 3 ≤ 0.83305 :=
  term_cert 3 1.732 1.7321 0.732 0.83305 1 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_4 : fC 4 ≤ 0.73576 :=
  term_cert 4 2 2 0 0.73576 2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_5 : fC 5 ≤ 0.64969 :=
  term_cert 5 2.236 2.2361 0.236 0.64969 2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_6 : fC 6 ≤ 0.57493 :=
  term_cert 6 2.4494 2.4495 0.4494 0.57493 2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_7 : fC 7 ≤ 0.51032 :=
  term_cert 7 2.6457 2.6458 0.6457 0.51032 2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_8 : fC 8 ≤ 0.45446 :=
  term_cert 8 2.8284 2.8285 0.8284 0.45446 2 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_9 : fC 9 ≤ 0.40601 :=
  term_cert 9 3 3 0 0.40601 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_10 : fC 10 ≤ 0.3639 :=
  term_cert 10 3.1622 3.1623 0.1622 0.3639 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_11 : fC 11 ≤ 0.32706 :=
  term_cert 11 3.3166 3.3167 0.3166 0.32706 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_12 : fC 12 ≤ 0.29476 :=
  term_cert 12 3.4641 3.4642 0.4641 0.29476 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_13 : fC 13 ≤ 0.26634 :=
  term_cert 13 3.6055 3.6056 0.6055 0.26634 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_14 : fC 14 ≤ 0.24122 :=
  term_cert 14 3.7416 3.7417 0.7416 0.24122 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_15 : fC 15 ≤ 0.21897 :=
  term_cert 15 3.8729 3.873 0.8729 0.21897 3 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_16 : fC 16 ≤ 0.19915 :=
  term_cert 16 4 4 0 0.19915 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_17 : fC 17 ≤ 0.18151 :=
  term_cert 17 4.1231 4.1232 0.1231 0.18151 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_18 : fC 18 ≤ 0.16573 :=
  term_cert 18 4.2426 4.2427 0.2426 0.16573 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_19 : fC 19 ≤ 0.15159 :=
  term_cert 19 4.3588 4.3589 0.3588 0.15159 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_20 : fC 20 ≤ 0.13888 :=
  term_cert 20 4.4721 4.4722 0.4721 0.13888 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_21 : fC 21 ≤ 0.12743 :=
  term_cert 21 4.5825 4.5826 0.5825 0.12743 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_22 : fC 22 ≤ 0.11709 :=
  term_cert 22 4.6904 4.6905 0.6904 0.11709 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_23 : fC 23 ≤ 0.10775 :=
  term_cert 23 4.7958 4.7959 0.7958 0.10775 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_24 : fC 24 ≤ 0.09928 :=
  term_cert 24 4.8989 4.899 0.8989 0.09928 4 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_25 : fC 25 ≤ 0.09158 :=
  term_cert 25 5 5 0 0.09158 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_26 : fC 26 ≤ 0.0846 :=
  term_cert 26 5.099 5.0991 0.099 0.0846 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_27 : fC 27 ≤ 0.07823 :=
  term_cert 27 5.1961 5.1962 0.1961 0.07823 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_28 : fC 28 ≤ 0.07242 :=
  term_cert 28 5.2915 5.2916 0.2915 0.07242 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_29 : fC 29 ≤ 0.06711 :=
  term_cert 29 5.3851 5.3852 0.3851 0.06711 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma fC_le_30 : fC 30 ≤ 0.06226 :=
  term_cert 30 5.4772 5.4773 0.4772 0.06226 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])

lemma Gt_le_30 : Gt 30 ≤ 0.97638 :=
  Gt_cert 30 5.4772 5.4773 0.4772 0.97638 5 (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (by norm_num [Finset.sum_range_succ, Nat.factorial])


/-! ## The constant `C` -/

lemma partial30_le : ∑ ℓ ∈ range 30, fC ((ℓ : ℝ) + 1) ≤ 9.55576 := by
  simp only [Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num
  linarith [fC_le_1, fC_le_2, fC_le_3, fC_le_4, fC_le_5, fC_le_6, fC_le_7, fC_le_8, fC_le_9,
    fC_le_10, fC_le_11, fC_le_12, fC_le_13, fC_le_14, fC_le_15, fC_le_16, fC_le_17, fC_le_18,
    fC_le_19, fC_le_20, fC_le_21, fC_le_22, fC_le_23, fC_le_24, fC_le_25, fC_le_26, fC_le_27,
    fC_le_28, fC_le_29, fC_le_30]

lemma partial_le (n : ℕ) : ∑ ℓ ∈ range n, fC ((ℓ : ℝ) + 1) ≤ 9.55576 + 0.97638 := by
  have h1 : ∑ ℓ ∈ range n, fC ((ℓ : ℝ) + 1) ≤ ∑ ℓ ∈ range (30 + n), fC ((ℓ : ℝ) + 1) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono (by omega))
      (fun i _ _ => fC_nonneg _)
  rw [Finset.sum_range_add] at h1
  have h2 := tail_sum_le 30 (by norm_num) n
  have h3 := Gt_nonneg (((30 : ℕ) : ℝ) + (n : ℝ)) (by positivity)
  push_cast at h1 h2 h3
  linarith [partial30_le, Gt_le_30]

/-- The constant `C = ∑_{ℓ ≥ 1} √ℓ e^(1-√ℓ)` of the SI proof of Proposition 6. -/
def Cneu : ℝ := ∑' ℓ : ℕ, fC ((ℓ : ℝ) + 1)

lemma summable_fC : Summable (fun ℓ : ℕ => fC ((ℓ : ℝ) + 1)) :=
  summable_of_sum_range_le (fun _ => fC_nonneg _) partial_le

/-- `C < 10.6`, the numerical value in the SI proof of Proposition 6. -/
theorem Cneu_lt : Cneu < 10.6 := by
  have h := Real.tsum_le_of_sum_range_le (fun ℓ => fC_nonneg ((ℓ : ℝ) + 1)) partial_le
  have h2 : (9.55576 : ℝ) + 0.97638 < 10.6 := by norm_num
  unfold Cneu
  linarith

/-! ## SI Proposition 6 with the constant `16 C` -/

/-- One term of the sum: `1/|P_N(1 - ω^ℓ)| ≤ (4√2/a) e^(-a) f(m)` with `a = √(N/2)` and
`m = min(ℓ, N - ℓ)`. -/
lemma term_le_fC (N ℓ : ℕ) (hN : 2 ≤ N) (hℓ1 : 1 ≤ ℓ) (hℓN : ℓ < N) :
    1 / ‖PNc N (1 - exp (2 * π * I * ℓ / N))‖
      ≤ 4 * √2 / √(N / 2) * Real.exp (-√(N / 2)) * fC ((min ℓ (N - ℓ) : ℕ) : ℝ) := by
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
  have hfC : fC (m : ℝ) = b * Real.exp (1 - b) := by
    unfold fC
    rw [← hbdef]
  rw [hfC]
  calc 1 / ‖PNc N (1 - exp ((2 * θ : ℝ) * I))‖
      ≤ 4 * √2 * (b / a) * Real.exp (-((N : ℝ) * (b / a) / 2)) := hterm
    _ ≤ 4 * √2 * (b / a) * (Real.exp (-a) * Real.exp (1 - b)) := by
        apply mul_le_mul_of_nonneg_left hexp
        positivity
    _ = 4 * √2 / a * Real.exp (-a) * (b * Real.exp (1 - b)) := by
        ring

/-- `∑_{ℓ=1}^{N-1} f(min(ℓ, N-ℓ)) ≤ 2 C`. -/
lemma sum_fC_min_le (N : ℕ) :
    ∑ i ∈ range (N - 1), fC ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) ≤ 2 * Cneu := by
  have hterm : ∀ i ∈ range (N - 1), fC ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ)
      ≤ fC ((i : ℝ) + 1) + fC (((N - 1 - 1 - i : ℕ) : ℝ) + 1) := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    have h1 := fC_nonneg ((i : ℝ) + 1)
    have h2 := fC_nonneg (((N - 1 - 1 - i : ℕ) : ℝ) + 1)
    rcases le_total (i + 1) (N - (i + 1)) with hle | hle
    · rw [min_eq_left hle]
      push_cast
      linarith
    · rw [min_eq_right hle]
      have e : ((N - (i + 1) : ℕ) : ℝ) = ((N - 1 - 1 - i : ℕ) : ℝ) + 1 := by
        rw [show N - (i + 1) = (N - 1 - 1 - i) + 1 by omega, Nat.cast_add, Nat.cast_one]
      rw [e]
      linarith
  have hpart : ∑ i ∈ range (N - 1), fC ((i : ℝ) + 1) ≤ Cneu :=
    summable_fC.sum_le_tsum (range (N - 1)) (fun i _ => fC_nonneg _)
  calc ∑ i ∈ range (N - 1), fC ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ)
      ≤ ∑ i ∈ range (N - 1), (fC ((i : ℝ) + 1) + fC (((N - 1 - 1 - i : ℕ) : ℝ) + 1)) :=
        Finset.sum_le_sum hterm
    _ = ∑ i ∈ range (N - 1), fC ((i : ℝ) + 1)
        + ∑ i ∈ range (N - 1), fC (((N - 1 - 1 - i : ℕ) : ℝ) + 1) := by
        rw [Finset.sum_add_distrib]
    _ = ∑ i ∈ range (N - 1), fC ((i : ℝ) + 1) + ∑ i ∈ range (N - 1), fC ((i : ℝ) + 1) := by
        rw [Finset.sum_range_reflect (fun j => fC ((j : ℝ) + 1)) (N - 1)]
    _ ≤ Cneu + Cneu := add_le_add hpart hpart
    _ = 2 * Cneu := by ring

/-- The deviation `|p_k - 1/N|` is at most the sum of `1/|P_N(1 - ω^ℓ)|` over `ℓ = 1, …, N-1`. -/
lemma dev_le_sum (N : ℕ) (hN : 2 ≤ N) (q : ℕ → ℝ) (hq : ∀ n, 0 ≤ q n) (hs : Summable q)
    (h : ℕ → ℝ) (hh : FixSteps 1 N h)
    (hgf : ∀ ℓ < N, ∃ F : ℕ → ℂ, GFStepsC (1 / 2) (1 / 2) (exp (2 * π * I * ℓ / N)) N F ∧
      ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N) = F 1 / (h 1 : ℂ))
    (k : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N|
      ≤ ∑ i ∈ range (N - 1), 1 / ‖PNc N (1 - exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N))‖ := by
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
  have hNM : N - 1 = M := by omega
  rw [hNM]
  exact habs

/-- SI Proposition 6 (neutral evolution) with the constant of the SI proof:
`|p_k - 1/N| ≤ 16 C N^(-1/2) e^(-√(N/2))` with `C = ∑_{ℓ ≥ 1} √ℓ e^(1-√ℓ)`. -/
theorem neutral_location_sharp (N : ℕ) (hN : 2 ≤ N) (q : ℕ → ℝ) (hq : ∀ n, 0 ≤ q n)
    (hs : Summable q) (h : ℕ → ℝ) (hh : FixSteps 1 N h)
    (hgf : ∀ ℓ < N, ∃ F : ℕ → ℂ, GFStepsC (1 / 2) (1 / 2) (exp (2 * π * I * ℓ / N)) N F ∧
      ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N) = F 1 / (h 1 : ℂ))
    (k : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N|
      ≤ 16 * Cneu / √N * Real.exp (-√(N / 2)) := by
  have habs := dev_le_sum N hN q hq hs h hh hgf k hk1 hkN
  have hterms : ∀ i ∈ range (N - 1), 1 / ‖PNc N (1 - exp (2 * π * I * ((i + 1 : ℕ) : ℂ) / N))‖
      ≤ 4 * √2 / √(N / 2) * Real.exp (-√(N / 2)) * fC ((min (i + 1) (N - (i + 1)) : ℕ) : ℝ) := by
    intro i hi
    have hi' := Finset.mem_range.mp hi
    exact term_le_fC N (i + 1) hN (by omega) (by omega)
  have hsumle := Finset.sum_le_sum hterms
  rw [← Finset.mul_sum] at hsumle
  have hmin := sum_fC_min_le N
  have hcoef : 4 * √2 / √((N : ℝ) / 2) = 8 / √N := by
    have h2 : √2 * √2 = (2 : ℝ) := Real.mul_self_sqrt (by norm_num)
    rw [Real.sqrt_div' _ (by norm_num : (0 : ℝ) ≤ 2), div_div_eq_mul_div, mul_assoc, h2]
    norm_num
  calc |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N| ≤ _ := habs
    _ ≤ _ := hsumle
    _ ≤ 4 * √2 / √((N : ℝ) / 2) * Real.exp (-√(N / 2)) * (2 * Cneu) := by
        apply mul_le_mul_of_nonneg_left hmin (by positivity)
    _ = 16 * Cneu / √N * Real.exp (-√(N / 2)) := by
        rw [hcoef]
        ring

/-- SI Proposition 6 with the numerical constant `170 > 16 · 10.6`. -/
theorem neutral_location_170 (N : ℕ) (hN : 2 ≤ N) (q : ℕ → ℝ) (hq : ∀ n, 0 ≤ q n)
    (hs : Summable q) (h : ℕ → ℝ) (hh : FixSteps 1 N h)
    (hgf : ∀ ℓ < N, ∃ F : ℕ → ℂ, GFStepsC (1 / 2) (1 / 2) (exp (2 * π * I * ℓ / N)) N F ∧
      ∑' n, (q n : ℂ) * exp (2 * π * I * ℓ * n / N) = F 1 / (h 1 : ℂ))
    (k : ℕ) (hk1 : 1 ≤ k) (hkN : k ≤ N) :
    |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N|
      ≤ 170 / √N * Real.exp (-√(N / 2)) := by
  have hmain := neutral_location_sharp N hN q hq hs h hh hgf k hk1 hkN
  have hC := Cneu_lt
  have hpos : 0 ≤ 1 / √(N : ℝ) * Real.exp (-√(N / 2)) := by positivity
  have e1 : 16 * Cneu / √N * Real.exp (-√(N / 2))
      = (16 * Cneu) * (1 / √(N : ℝ) * Real.exp (-√(N / 2))) := by ring
  have e2 : 170 / √N * Real.exp (-√(N / 2)) = 170 * (1 / √(N : ℝ) * Real.exp (-√(N / 2))) := by
    ring
  have h16 : 16 * Cneu ≤ 170 := by linarith
  calc |(∑' n, if n % N = k - 1 then q n else 0) - 1 / N|
      ≤ 16 * Cneu / √N * Real.exp (-√(N / 2)) := hmain
    _ = (16 * Cneu) * (1 / √(N : ℝ) * Real.exp (-√(N / 2))) := e1
    _ ≤ 170 * (1 / √(N : ℝ) * Real.exp (-√(N / 2))) := mul_le_mul_of_nonneg_right h16 hpos
    _ = 170 / √N * Real.exp (-√(N / 2)) := e2.symm

end

end FixLoc
