import FixLoc.Factorization

/-!
# Two islands (SI, Islands)

Two islands hold `n` individuals each. The state `(a, b)` counts the mutants on island 1 and on
island 2. The offspring of an island 1 parent is placed on island 2 with probability `m₁₂` and on
island 1 otherwise. The offspring of an island 2 parent is placed on island 1 with probability `m₂₁`
and on island 2 otherwise. The offspring replaces a uniformly chosen individual of the target
island. Under neutral evolution every individual is the parent with probability `1 / (2n)`.

* `drift1`, `drift2`: the expected changes of the mutant counts (neutral drift).
* `harmonic`: `m₁₂ a + m₂₁ b` has zero expected change at every state.
* `neutral_fixation`: the fixation probability from `(a, b)` is
  `(m₁₂ a + m₂₁ b) / (n (m₁₂ + m₂₁))`.
* `neutral_initial_island`: a single mutant on island 2 fixes `m₂₁ / m₁₂` times as often as a
  single mutant on island 1.
* `linearization`, `colonization_linearization`, `colonization_comparison`, `neutral_comparison`:
  the algebra of the rare wild type approximations.

As in the other files, the fixation probability enters through its first-step equations.
-/

open Finset Relation

namespace FixLoc

noncomputable section

namespace Islands

/-- The states `(a, b)` with `0 ≤ a, b ≤ n`. -/
def box (n : ℕ) : Finset (ℤ × ℤ) := Finset.Icc (0 : ℤ) n ×ˢ Finset.Icc (0 : ℤ) n

/-- The transient states: the states other than `(0, 0)` and `(n, n)`. -/
def T (n : ℕ) : Finset (ℤ × ℤ) := ((box n).erase (0, 0)).erase ((n : ℤ), (n : ℤ))

lemma mem_box {n : ℕ} {x : ℤ × ℤ} : x ∈ box n ↔ 0 ≤ x.1 ∧ x.1 ≤ n ∧ 0 ≤ x.2 ∧ x.2 ≤ n := by
  simp only [box, Finset.mem_product, Finset.mem_Icc, and_assoc]

lemma mem_box' {n : ℕ} {a b : ℤ} : (a, b) ∈ box n ↔ 0 ≤ a ∧ a ≤ n ∧ 0 ≤ b ∧ b ≤ n :=
  mem_box

lemma mem_T {n : ℕ} {x : ℤ × ℤ} :
    x ∈ T n ↔ x ≠ ((n : ℤ), (n : ℤ)) ∧ x ≠ (0, 0) ∧ x ∈ box n := by
  simp only [T, Finset.mem_erase]

variable (n : ℕ) (m₁₂ m₂₁ : ℝ)

/-- A mutant offspring replaces a wild type on island 1. -/
def up1 (a b : ℤ) : ℝ :=
  ((a : ℝ) * (1 - m₁₂) + (b : ℝ) * m₂₁) * ((n : ℝ) - a) / (2 * (n : ℝ) ^ 2)

/-- A wild type offspring replaces a mutant on island 1. -/
def dn1 (a b : ℤ) : ℝ :=
  (((n : ℝ) - a) * (1 - m₁₂) + ((n : ℝ) - b) * m₂₁) * (a : ℝ) / (2 * (n : ℝ) ^ 2)

/-- A mutant offspring replaces a wild type on island 2. -/
def up2 (a b : ℤ) : ℝ :=
  ((a : ℝ) * m₁₂ + (b : ℝ) * (1 - m₂₁)) * ((n : ℝ) - b) / (2 * (n : ℝ) ^ 2)

/-- A wild type offspring replaces a mutant on island 2. -/
def dn2 (a b : ℤ) : ℝ :=
  (((n : ℝ) - a) * m₁₂ + ((n : ℝ) - b) * (1 - m₂₁)) * (b : ℝ) / (2 * (n : ℝ) ^ 2)

/-- The offspring replaces an individual of its own type. -/
def stay (a b : ℤ) : ℝ :=
  (((a : ℝ) * (1 - m₁₂) + (b : ℝ) * m₂₁) * (a : ℝ)
    + (((n : ℝ) - a) * (1 - m₁₂) + ((n : ℝ) - b) * m₂₁) * ((n : ℝ) - a)
    + ((a : ℝ) * m₁₂ + (b : ℝ) * (1 - m₂₁)) * (b : ℝ)
    + (((n : ℝ) - a) * m₁₂ + ((n : ℝ) - b) * (1 - m₂₁)) * ((n : ℝ) - b)) / (2 * (n : ℝ) ^ 2)

/-- One time step of the neutral two-island chain. -/
def P (x y : ℤ × ℤ) : ℝ :=
  if x ∈ box n then
    (if y = (x.1 + 1, x.2) then up1 n m₁₂ m₂₁ x.1 x.2 else 0)
      + (if y = (x.1 - 1, x.2) then dn1 n m₁₂ m₂₁ x.1 x.2 else 0)
      + (if y = (x.1, x.2 + 1) then up2 n m₁₂ m₂₁ x.1 x.2 else 0)
      + (if y = (x.1, x.2 - 1) then dn2 n m₁₂ m₂₁ x.1 x.2 else 0)
      + (if y = x then stay n m₁₂ m₂₁ x.1 x.2 else 0)
  else 0

/-- The neutral fixation probability `(m₁₂ x₁ + m₂₁ x₂) / (m₁₂ + m₂₁)` with `xᵢ` the mutant
frequency on island `i`. -/
def h (x : ℤ × ℤ) : ℝ := (m₁₂ * x.1 + m₂₁ * x.2) / (n * (m₁₂ + m₂₁))

variable {n m₁₂ m₂₁}

lemma row_sum (hn : 0 < n) (a b : ℤ) :
    up1 n m₁₂ m₂₁ a b + dn1 n m₁₂ m₂₁ a b + up2 n m₁₂ m₂₁ a b + dn2 n m₁₂ m₂₁ a b
      + stay n m₁₂ m₂₁ a b = 1 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold up1 dn1 up2 dn2 stay
  field_simp
  ring

lemma weights_nonneg (h12 : 0 ≤ m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ ≤ 1)
    {x : ℤ × ℤ} (hx : x ∈ box n) :
    0 ≤ up1 n m₁₂ m₂₁ x.1 x.2 ∧ 0 ≤ dn1 n m₁₂ m₂₁ x.1 x.2 ∧ 0 ≤ up2 n m₁₂ m₂₁ x.1 x.2 ∧
      0 ≤ dn2 n m₁₂ m₂₁ x.1 x.2 ∧ 0 ≤ stay n m₁₂ m₂₁ x.1 x.2 := by
  obtain ⟨ha0, han, hb0, hbn⟩ := mem_box.mp hx
  have ha : (0 : ℝ) ≤ x.1 := by exact_mod_cast ha0
  have ha' : (0 : ℝ) ≤ (n : ℝ) - x.1 := by
    have : (x.1 : ℝ) ≤ ((n : ℤ) : ℝ) := by exact_mod_cast han
    push_cast at this
    linarith
  have hb : (0 : ℝ) ≤ x.2 := by exact_mod_cast hb0
  have hb' : (0 : ℝ) ≤ (n : ℝ) - x.2 := by
    have : (x.2 : ℝ) ≤ ((n : ℤ) : ℝ) := by exact_mod_cast hbn
    push_cast at this
    linarith
  have hd : (0 : ℝ) ≤ 2 * (n : ℝ) ^ 2 := by positivity
  have c12 : (0 : ℝ) ≤ 1 - m₁₂ := by linarith
  have c21 : (0 : ℝ) ≤ 1 - m₂₁ := by linarith
  have s1 : 0 ≤ (x.1 : ℝ) * (1 - m₁₂) + (x.2 : ℝ) * m₂₁ :=
    add_nonneg (mul_nonneg ha c12) (mul_nonneg hb h21)
  have s2 : 0 ≤ ((n : ℝ) - x.1) * (1 - m₁₂) + ((n : ℝ) - x.2) * m₂₁ :=
    add_nonneg (mul_nonneg ha' c12) (mul_nonneg hb' h21)
  have s3 : 0 ≤ (x.1 : ℝ) * m₁₂ + (x.2 : ℝ) * (1 - m₂₁) :=
    add_nonneg (mul_nonneg ha h12) (mul_nonneg hb c21)
  have s4 : 0 ≤ ((n : ℝ) - x.1) * m₁₂ + ((n : ℝ) - x.2) * (1 - m₂₁) :=
    add_nonneg (mul_nonneg ha' h12) (mul_nonneg hb' c21)
  unfold up1 dn1 up2 dn2 stay
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · exact div_nonneg (mul_nonneg s1 ha') hd
  · exact div_nonneg (mul_nonneg s2 ha) hd
  · exact div_nonneg (mul_nonneg s3 hb') hd
  · exact div_nonneg (mul_nonneg s4 hb) hd
  · exact div_nonneg (add_nonneg (add_nonneg (add_nonneg (mul_nonneg s1 ha) (mul_nonneg s2 ha'))
      (mul_nonneg s3 hb)) (mul_nonneg s4 hb')) hd

lemma P_nonneg (h12 : 0 ≤ m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ ≤ 1)
    (x y : ℤ × ℤ) : 0 ≤ P n m₁₂ m₂₁ x y := by
  unfold P
  by_cases hx : x ∈ box n
  · obtain ⟨w1, w2, w3, w4, w5⟩ := weights_nonneg h12 h12' h21 h21' hx
    rw [if_pos hx]
    exact add_nonneg (add_nonneg (add_nonneg (add_nonneg (ite_nonneg w1 le_rfl)
      (ite_nonneg w2 le_rfl)) (ite_nonneg w3 le_rfl)) (ite_nonneg w4 le_rfl)) (ite_nonneg w5 le_rfl)
  · simp only [if_neg hx, le_refl]

lemma sum_P_le (hn : 0 < n) (h12 : 0 ≤ m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ ≤ 1)
    {x : ℤ × ℤ} (hx : x ∈ box n) (F : Finset (ℤ × ℤ)) : ∑ y ∈ F, P n m₁₂ m₂₁ x y ≤ 1 := by
  obtain ⟨w1, w2, w3, w4, w5⟩ := weights_nonneg h12 h12' h21 h21' hx
  have hs := row_sum (m₁₂ := m₁₂) (m₂₁ := m₂₁) hn x.1 x.2
  simp only [P, if_pos hx, Finset.sum_add_distrib, Finset.sum_ite_eq']
  split_ifs <;> linarith

/-- Drift of the mutant count on island 1: `E[Δa] = m₂₁ (b - a) / (2n)`. -/
lemma drift1 (hn : 0 < n) (a b : ℤ) :
    up1 n m₁₂ m₂₁ a b - dn1 n m₁₂ m₂₁ a b = m₂₁ * ((b : ℝ) - a) / (2 * n) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold up1 dn1
  field_simp
  ring

/-- Drift of the mutant count on island 2: `E[Δb] = m₁₂ (a - b) / (2n)`. -/
lemma drift2 (hn : 0 < n) (a b : ℤ) :
    up2 n m₁₂ m₂₁ a b - dn2 n m₁₂ m₂₁ a b = m₁₂ * ((a : ℝ) - b) / (2 * n) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  unfold up2 dn2
  field_simp
  ring

/-- Drift of the neutral chain: `h` is harmonic at every state. -/
lemma harmonic (hn : 0 < n) (hm : m₁₂ + m₂₁ ≠ 0) (x : ℤ × ℤ) :
    up1 n m₁₂ m₂₁ x.1 x.2 * h n m₁₂ m₂₁ (x.1 + 1, x.2)
      + dn1 n m₁₂ m₂₁ x.1 x.2 * h n m₁₂ m₂₁ (x.1 - 1, x.2)
      + up2 n m₁₂ m₂₁ x.1 x.2 * h n m₁₂ m₂₁ (x.1, x.2 + 1)
      + dn2 n m₁₂ m₂₁ x.1 x.2 * h n m₁₂ m₂₁ (x.1, x.2 - 1)
      + stay n m₁₂ m₂₁ x.1 x.2 * h n m₁₂ m₂₁ x = h n m₁₂ m₂₁ x := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  obtain ⟨a, b⟩ := x
  simp only [up1, dn1, up2, dn2, stay, h]
  push_cast
  field_simp
  ring

lemma h_full (hn : 0 < n) (hm : m₁₂ + m₂₁ ≠ 0) : h n m₁₂ m₂₁ ((n : ℤ), (n : ℤ)) = 1 := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  simp only [h]
  push_cast
  rw [div_eq_one_iff_eq (mul_ne_zero hn' hm)]
  ring

lemma h_zero : h n m₁₂ m₂₁ (0, 0) = 0 := by
  simp [h]

lemma full_not_mem : ((n : ℤ), (n : ℤ)) ∉ T n := by
  simp [mem_T]

/-- One possible move of the first-step equation of `h`. -/
lemma term (hn : 0 < n) (hm : m₁₂ + m₂₁ ≠ 0) (y : ℤ × ℤ) (w : ℝ) (hw : y ∉ box n → w = 0) :
    (if ((n : ℤ), (n : ℤ)) = y then w else 0) + (if y ∈ T n then w * h n m₁₂ m₂₁ y else 0)
      = w * h n m₁₂ m₂₁ y := by
  by_cases hf : ((n : ℤ), (n : ℤ)) = y
  · subst hf
    rw [if_pos rfl, if_neg full_not_mem, h_full hn hm]
    ring
  · rw [if_neg hf]
    by_cases hT : y ∈ T n
    · rw [if_pos hT]
      ring
    · rw [if_neg hT]
      by_cases hb : y ∈ box n
      · have hy0 : y = (0, 0) := by
          by_contra hne
          exact hT (mem_T.mpr ⟨fun e => hf e.symm, hne, hb⟩)
        rw [hy0, h_zero]
        ring
      · rw [hw hb]
        ring

lemma hw1 {x : ℤ × ℤ} (hx : x ∈ box n) (hy : (x.1 + 1, x.2) ∉ box n) :
    up1 n m₁₂ m₂₁ x.1 x.2 = 0 := by
  simp only [mem_box] at hx hy
  have e : x.1 = n := by omega
  simp [up1, e]

lemma hw2 {x : ℤ × ℤ} (hx : x ∈ box n) (hy : (x.1 - 1, x.2) ∉ box n) :
    dn1 n m₁₂ m₂₁ x.1 x.2 = 0 := by
  simp only [mem_box] at hx hy
  have e : x.1 = 0 := by omega
  simp [dn1, e]

lemma hw3 {x : ℤ × ℤ} (hx : x ∈ box n) (hy : (x.1, x.2 + 1) ∉ box n) :
    up2 n m₁₂ m₂₁ x.1 x.2 = 0 := by
  simp only [mem_box] at hx hy
  have e : x.2 = n := by omega
  simp [up2, e]

lemma hw4 {x : ℤ × ℤ} (hx : x ∈ box n) (hy : (x.1, x.2 - 1) ∉ box n) :
    dn2 n m₁₂ m₂₁ x.1 x.2 = 0 := by
  simp only [mem_box] at hx hy
  have e : x.2 = 0 := by omega
  simp [dn2, e]

/-- First-step equation of `h` on the transient states. -/
lemma h_first_step (hn : 0 < n) (hm : m₁₂ + m₂₁ ≠ 0) {x : ℤ × ℤ} (hx : x ∈ T n) :
    h n m₁₂ m₂₁ x
      = P n m₁₂ m₂₁ x ((n : ℤ), (n : ℤ)) + ∑ y ∈ T n, P n m₁₂ m₂₁ x y * h n m₁₂ m₂₁ y := by
  have hxb : x ∈ box n := (mem_T.mp hx).2.2
  simp only [P, if_pos hxb, add_mul, ite_mul, zero_mul, Finset.sum_add_distrib,
    Finset.sum_ite_eq']
  have t1 := term hn hm (x.1 + 1, x.2) (up1 n m₁₂ m₂₁ x.1 x.2) (hw1 hxb)
  have t2 := term hn hm (x.1 - 1, x.2) (dn1 n m₁₂ m₂₁ x.1 x.2) (hw2 hxb)
  have t3 := term hn hm (x.1, x.2 + 1) (up2 n m₁₂ m₂₁ x.1 x.2) (hw3 hxb)
  have t4 := term hn hm (x.1, x.2 - 1) (dn2 n m₁₂ m₂₁ x.1 x.2) (hw4 hxb)
  have t5 := term hn hm x (stay n m₁₂ m₂₁ x.1 x.2) (fun hb => absurd hxb hb)
  have H := harmonic hn hm x
  linarith

lemma P_right {a b : ℤ} (hx : (a, b) ∈ box n) :
    P n m₁₂ m₂₁ (a, b) (a + 1, b) = up1 n m₁₂ m₂₁ a b := by
  have h1 : ((a + 1, b) : ℤ × ℤ) ≠ (a - 1, b) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h2 : ((a + 1, b) : ℤ × ℤ) ≠ (a, b + 1) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h3 : ((a + 1, b) : ℤ × ℤ) ≠ (a, b - 1) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h4 : ((a + 1, b) : ℤ × ℤ) ≠ (a, b) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  unfold P
  rw [if_pos hx]
  dsimp only
  rw [if_pos rfl, if_neg h1, if_neg h2, if_neg h3, if_neg h4]
  ring

lemma P_up {a b : ℤ} (hx : (a, b) ∈ box n) :
    P n m₁₂ m₂₁ (a, b) (a, b + 1) = up2 n m₁₂ m₂₁ a b := by
  have h1 : ((a, b + 1) : ℤ × ℤ) ≠ (a + 1, b) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h2 : ((a, b + 1) : ℤ × ℤ) ≠ (a - 1, b) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h3 : ((a, b + 1) : ℤ × ℤ) ≠ (a, b - 1) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  have h4 : ((a, b + 1) : ℤ × ℤ) ≠ (a, b) := by
    intro e
    simp only [Prod.mk.injEq] at e
    omega
  unfold P
  rw [if_pos hx]
  dsimp only
  rw [if_neg h1, if_neg h2, if_pos rfl, if_neg h3, if_neg h4]
  ring

lemma up1_pos (hn : 0 < n) (h12' : m₁₂ < 1) (h21 : 0 < m₂₁) {c d : ℤ} (hc : 0 ≤ c)
    (hcn : c < n) (hd : 0 ≤ d) (hcd : 1 ≤ c ∨ 1 ≤ d) : 0 < up1 n m₁₂ m₂₁ c d := by
  have hc' : (0 : ℝ) ≤ c := by exact_mod_cast hc
  have hd' : (0 : ℝ) ≤ d := by exact_mod_cast hd
  have hcn' : (c : ℝ) < ((n : ℤ) : ℝ) := by exact_mod_cast hcn
  push_cast at hcn'
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  unfold up1
  refine div_pos (mul_pos ?_ (by linarith)) (mul_pos two_pos (pow_pos hn' 2))
  rcases hcd with hc1 | hd1
  · have h1 : (1 : ℝ) ≤ c := by exact_mod_cast hc1
    have h2 : 0 < (c : ℝ) * (1 - m₁₂) := mul_pos (by linarith) (by linarith)
    have h3 : 0 ≤ (d : ℝ) * m₂₁ := mul_nonneg hd' h21.le
    linarith
  · have h1 : (1 : ℝ) ≤ d := by exact_mod_cast hd1
    have h2 : 0 < (d : ℝ) * m₂₁ := mul_pos (by linarith) h21
    have h3 : 0 ≤ (c : ℝ) * (1 - m₁₂) := mul_nonneg hc' (by linarith)
    linarith

lemma up2_pos (hn : 0 < n) (h12 : 0 < m₁₂) (h21' : m₂₁ < 1) {d : ℤ} (hd : 0 ≤ d)
    (hdn : d < n) : 0 < up2 n m₁₂ m₂₁ n d := by
  have hd' : (0 : ℝ) ≤ d := by exact_mod_cast hd
  have hdn' : (d : ℝ) < ((n : ℤ) : ℝ) := by exact_mod_cast hdn
  push_cast at hdn'
  have hn' : (0 : ℝ) < n := by exact_mod_cast hn
  unfold up2
  push_cast
  refine div_pos (mul_pos ?_ (by linarith)) (mul_pos two_pos (pow_pos hn' 2))
  have h1 : 0 < (n : ℝ) * m₁₂ := mul_pos hn' h12
  have h2 : 0 ≤ (d : ℝ) * (1 - m₂₁) := mul_nonneg hd' (by linarith)
  linarith

/-- The relation of one positive transition between transient states. -/
abbrev R (n : ℕ) (m₁₂ m₂₁ : ℝ) (u v : ℤ × ℤ) : Prop :=
  u ∈ T n ∧ v ∈ T n ∧ 0 < P n m₁₂ m₂₁ u v

lemma path_right (hn : 0 < n) (h12 : 0 ≤ m₁₂) (h12' : m₁₂ < 1) (h21 : 0 < m₂₁) :
    ∀ k : ℕ, ∀ c d : ℤ, (c, d) ∈ T n → c + k ≤ n → (c + k, d) ≠ ((n : ℤ), (n : ℤ)) →
      ReflTransGen (R n m₁₂ m₂₁) (c, d) (c + k, d) := by
  intro k
  induction k with
  | zero =>
    intro c d _ _ _
    simpa using (ReflTransGen.refl : ReflTransGen (R n m₁₂ m₂₁) (c, d) (c, d))
  | succ k ih =>
    intro c d hT hle hne
    obtain ⟨hTn, hT0, hTb⟩ := mem_T.mp hT
    obtain ⟨hc0, hcn, hd0, hdn⟩ := mem_box'.mp hTb
    push_cast at hle hne ⊢
    have hk : (c + k, d) ≠ ((n : ℤ), (n : ℤ)) := by
      intro e
      simp only [Prod.mk.injEq] at e
      omega
    have hpath := ih c d hT (by omega) hk
    have hb : (c + (k : ℤ), d) ∈ box n := mem_box'.mpr ⟨by omega, by omega, hd0, hdn⟩
    have hmem : (c + (k : ℤ), d) ∈ T n := by
      refine mem_T.mpr ⟨hk, ?_, hb⟩
      intro e
      simp only [Prod.mk.injEq] at e
      exact hT0 (by simp only [Prod.mk.injEq]; omega)
    have hmem' : (c + ((k : ℤ) + 1), d) ∈ T n := by
      refine mem_T.mpr ⟨hne, ?_, mem_box'.mpr ⟨by omega, by omega, hd0, hdn⟩⟩
      intro e
      simp only [Prod.mk.injEq] at e
      omega
    refine ReflTransGen.tail hpath ⟨hmem, hmem', ?_⟩
    rw [← add_assoc, P_right hb]
    apply up1_pos hn h12' h21 (by omega) (by omega) hd0
    by_cases hc : 1 ≤ c + (k : ℤ)
    · exact Or.inl hc
    · right
      by_contra hd1
      exact hT0 (by simp only [Prod.mk.injEq]; omega)

lemma path_up (hn : 0 < n) (h12 : 0 < m₁₂) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ < 1) :
    ∀ k : ℕ, ∀ d : ℤ, ((n : ℤ), d) ∈ T n → d + k < n →
      ReflTransGen (R n m₁₂ m₂₁) ((n : ℤ), d) ((n : ℤ), d + k) := by
  intro k
  induction k with
  | zero =>
    intro d _ _
    simpa using (ReflTransGen.refl : ReflTransGen (R n m₁₂ m₂₁) ((n : ℤ), d) ((n : ℤ), d))
  | succ k ih =>
    intro d hT hlt
    obtain ⟨hTn, hT0, hTb⟩ := mem_T.mp hT
    obtain ⟨_, _, hd0, hdn⟩ := mem_box'.mp hTb
    push_cast at hlt ⊢
    have hpath := ih d hT (by omega)
    have hb : ((n : ℤ), d + (k : ℤ)) ∈ box n := mem_box'.mpr ⟨by omega, le_rfl, by omega, by omega⟩
    have hmem : ((n : ℤ), d + (k : ℤ)) ∈ T n := by
      refine mem_T.mpr ⟨?_, ?_, hb⟩
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
    have hmem' : ((n : ℤ), d + ((k : ℤ) + 1)) ∈ T n := by
      refine mem_T.mpr ⟨?_, ?_, mem_box'.mpr ⟨by omega, le_rfl, by omega, by omega⟩⟩
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
    refine ReflTransGen.tail hpath ⟨hmem, hmem', ?_⟩
    rw [← add_assoc, P_up hb]
    exact up2_pos hn h12 h21' (by omega) (by omega)

lemma leak (hn : 0 < n) (h12 : 0 ≤ m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ ≤ 1)
    {z : ℤ × ℤ} (hz : z ∈ box n) (hpos : 0 < P n m₁₂ m₂₁ z ((n : ℤ), (n : ℤ))) :
    ∑ y ∈ T n, P n m₁₂ m₂₁ z y < 1 := by
  have hs := sum_P_le hn h12 h12' h21 h21' hz (insert ((n : ℤ), (n : ℤ)) (T n))
  rw [Finset.sum_insert full_not_mem] at hs
  linarith

/-- From every transient state the chain reaches `(n, n)` with positive probability. -/
lemma reach (hn : 0 < n) (h12 : 0 < m₁₂) (h12' : m₁₂ < 1) (h21 : 0 < m₂₁) (h21' : m₂₁ < 1) :
    ∀ x ∈ T n, AbsorbReach (T n) (P n m₁₂ m₂₁) x := by
  intro x hx
  unfold AbsorbReach
  obtain ⟨c, d⟩ := x
  obtain ⟨hTn, hT0, hTb⟩ := mem_T.mp hx
  obtain ⟨hc0, hcn, hd0, hdn⟩ := mem_box'.mp hTb
  have hn1 : (1 : ℤ) ≤ n := by exact_mod_cast hn
  rcases lt_or_eq_of_le hdn with hd | hd
  · obtain ⟨k, hk⟩ : ∃ k : ℕ, c + k = n := ⟨(n - c).toNat, by omega⟩
    have p1 := path_right hn h12.le h12' h21 k c d hx (by omega) (by
      intro e
      simp only [Prod.mk.injEq] at e
      omega)
    rw [hk] at p1
    have hT2 : ((n : ℤ), d) ∈ T n := by
      refine mem_T.mpr ⟨?_, ?_, mem_box'.mpr ⟨by omega, le_rfl, hd0, hdn⟩⟩
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
      · intro e
        simp only [Prod.mk.injEq] at e
        omega
    obtain ⟨k2, hk2⟩ : ∃ k2 : ℕ, d + k2 = n - 1 := ⟨(n - 1 - d).toNat, by omega⟩
    have p2 := path_up hn h12 h21.le h21' k2 d hT2 (by omega)
    rw [hk2] at p2
    refine ⟨((n : ℤ), (n : ℤ) - 1), p1.trans p2, ?_⟩
    have hzb : ((n : ℤ), (n : ℤ) - 1) ∈ box n := mem_box'.mpr ⟨by omega, le_rfl, by omega, by omega⟩
    apply leak hn h12.le h12'.le h21.le h21'.le hzb
    have e : (((n : ℤ), (n : ℤ)) : ℤ × ℤ) = ((n : ℤ), (n : ℤ) - 1 + 1) := by simp
    rw [e, P_up hzb]
    exact up2_pos hn h12 h21' (by omega) (by omega)
  · subst hd
    have hcn' : c < n := by
      by_contra hc
      have hcn2 : c = n := by omega
      exact hTn (by rw [hcn2])
    obtain ⟨k, hk⟩ : ∃ k : ℕ, c + k = n - 1 := ⟨(n - 1 - c).toNat, by omega⟩
    have p1 := path_right hn h12.le h12' h21 k c n hx (by omega) (by
      intro e
      simp only [Prod.mk.injEq] at e
      omega)
    rw [hk] at p1
    refine ⟨((n : ℤ) - 1, (n : ℤ)), p1, ?_⟩
    have hzb : ((n : ℤ) - 1, (n : ℤ)) ∈ box n := mem_box'.mpr ⟨by omega, by omega, by omega, le_rfl⟩
    apply leak hn h12.le h12'.le h21.le h21'.le hzb
    have e : (((n : ℤ), (n : ℤ)) : ℤ × ℤ) = ((n : ℤ) - 1 + 1, (n : ℤ)) := by simp
    rw [e, P_right hzb]
    exact up1_pos hn h12' h21 (by omega) (by omega) (by omega) (Or.inr hn1)

/-- **Neutral fixation probability on two islands.** If `ρ` satisfies the first-step equations
of the neutral two-island chain, then `ρ (a, b) = (m₁₂ a + m₂₁ b) / (n (m₁₂ + m₂₁))`. -/
theorem neutral_fixation (hn : 0 < n) (h12 : 0 < m₁₂) (h12' : m₁₂ < 1) (h21 : 0 < m₂₁)
    (h21' : m₂₁ < 1) (ρ : ℤ × ℤ → ℝ)
    (hρ : ∀ x ∈ T n, ρ x = P n m₁₂ m₂₁ x ((n : ℤ), (n : ℤ)) + ∑ y ∈ T n, P n m₁₂ m₂₁ x y * ρ y) :
    ∀ x ∈ T n, ρ x = (m₁₂ * x.1 + m₂₁ * x.2) / (n * (m₁₂ + m₂₁)) := by
  have hm : m₁₂ + m₂₁ ≠ 0 := by linarith
  have hD := homogeneous_zero (T n) (P n m₁₂ m₂₁) (P_nonneg h12.le h12'.le h21.le h21'.le)
    (fun x hx => sum_P_le hn h12.le h12'.le h21.le h21'.le (mem_T.mp hx).2.2 (T n))
    (reach hn h12 h12' h21 h21') (fun x => ρ x - h n m₁₂ m₂₁ x) (by
      intro x hx
      show ρ x - h n m₁₂ m₂₁ x = ∑ y ∈ T n, P n m₁₂ m₂₁ x y * (ρ y - h n m₁₂ m₂₁ y)
      rw [hρ x hx, h_first_step hn hm hx]
      simp only [mul_sub, Finset.sum_sub_distrib]
      ring)
  intro x hx
  have e := hD x hx
  simp only [h] at e
  linarith

/-- **Initial island.** A single mutant on island 2 fixes `m₂₁ / m₁₂` times as often as a single
mutant on island 1. -/
theorem neutral_initial_island (hn : 0 < n) (h12 : 0 < m₁₂) (h12' : m₁₂ < 1) (h21 : 0 < m₂₁)
    (h21' : m₂₁ < 1) (ρ : ℤ × ℤ → ℝ)
    (hρ : ∀ x ∈ T n, ρ x = P n m₁₂ m₂₁ x ((n : ℤ), (n : ℤ)) + ∑ y ∈ T n, P n m₁₂ m₂₁ x y * ρ y) :
    ρ (1, 0) = m₁₂ / (n * (m₁₂ + m₂₁)) ∧ ρ (0, 1) = m₂₁ / (n * (m₁₂ + m₂₁)) ∧
      ρ (0, 1) / ρ (1, 0) = m₂₁ / m₁₂ := by
  have hn1 : (1 : ℤ) ≤ n := by exact_mod_cast hn
  have h10 : ((1 : ℤ), (0 : ℤ)) ∈ T n := by
    refine mem_T.mpr ⟨?_, ?_, mem_box.mpr ⟨by omega, hn1, le_rfl, by omega⟩⟩
    · intro e
      simp only [Prod.mk.injEq] at e
      omega
    · intro e
      simp only [Prod.mk.injEq] at e
      omega
  have h01 : ((0 : ℤ), (1 : ℤ)) ∈ T n := by
    refine mem_T.mpr ⟨?_, ?_, mem_box.mpr ⟨le_rfl, by omega, by omega, hn1⟩⟩
    · intro e
      simp only [Prod.mk.injEq] at e
      omega
    · intro e
      simp only [Prod.mk.injEq] at e
      omega
  have e1 := neutral_fixation hn h12 h12' h21 h21' ρ hρ _ h10
  have e2 := neutral_fixation hn h12 h12' h21 h21' ρ hρ _ h01
  simp only [Int.cast_one, Int.cast_zero, mul_one, mul_zero, add_zero, zero_add] at e1 e2
  refine ⟨e1, e2, ?_⟩
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hm : m₁₂ + m₂₁ ≠ 0 := by linarith
  have h12ne : m₁₂ ≠ 0 := h12.ne'
  rw [e1, e2]
  field_simp

/-! ## The algebra of the rare wild type approximations -/

/-- **Linearization at fixation.** With `u = 1 - x₁` and `v = 1 - x₂`, the right sides of the
deterministic approximation are `M_r (u, v)` plus quadratic terms. -/
theorem linearization (r m₁₂ m₂₁ u v : ℝ) :
    -((1 - (1 - u)) * r * ((1 - m₁₂) * (1 - u) + m₂₁ * (1 - v))
        - (1 - u) * ((1 - m₁₂) * (1 - (1 - u)) + m₂₁ * (1 - (1 - v))))
      = -((r - 1) * (1 - m₁₂) + r * m₂₁) * u + m₂₁ * v
        + (r - 1) * u * ((1 - m₁₂) * u + m₂₁ * v) ∧
    -((1 - (1 - v)) * r * (m₁₂ * (1 - u) + (1 - m₂₁) * (1 - v))
        - (1 - v) * (m₁₂ * (1 - (1 - u)) + (1 - m₂₁) * (1 - (1 - v))))
      = m₁₂ * u - ((r - 1) * (1 - m₂₁) + r * m₁₂) * v
        + (r - 1) * v * (m₁₂ * u + (1 - m₂₁) * v) := by
  constructor <;> ring

/-- **Colonization near fixation.** The linear part is diagonal with rates `1 - m₁₂ + m₂₁` and
`1 - m₂₁ + m₁₂`. -/
theorem colonization_linearization (m₁₂ m₂₁ u v : ℝ) :
    -((1 - (1 - u)) * ((1 - m₁₂) * (1 - u) + m₂₁ * (1 - v)))
      = -(1 - m₁₂ + m₂₁) * u + u * ((1 - m₁₂) * u + m₂₁ * v) ∧
    -((1 - (1 - v)) * (m₁₂ * (1 - u) + (1 - m₂₁) * (1 - v)))
      = -(1 - m₂₁ + m₁₂) * v + v * (m₁₂ * u + (1 - m₂₁) * v) := by
  constructor <;> ring

/-- Island 1 wild types decay more slowly under colonization exactly when `m₁₂ > m₂₁`. -/
theorem colonization_comparison (m₁₂ m₂₁ : ℝ) :
    1 - m₁₂ + m₂₁ < 1 - m₂₁ + m₁₂ ↔ m₂₁ < m₁₂ := by
  constructor <;> intro h <;> linarith

/-- **Neutral comparison.** The effective loss rates satisfy `γ₁ < γ₂` exactly when `m₁₂` is
farther from `1/2` than `m₂₁`. -/
theorem neutral_comparison (m₁₂ m₂₁ : ℝ) (hm : 0 < m₁₂ + m₂₁) :
    2 * m₁₂ * (1 - m₁₂ + m₂₁) / (m₁₂ + m₂₁) < 2 * m₂₁ * (1 - m₂₁ + m₁₂) / (m₁₂ + m₂₁) ↔
      |m₂₁ - 1 / 2| < |m₁₂ - 1 / 2| := by
  have key : 2 * m₁₂ * (1 - m₁₂ + m₂₁) < 2 * m₂₁ * (1 - m₂₁ + m₁₂) ↔
      (m₂₁ - 1 / 2) ^ 2 < (m₁₂ - 1 / 2) ^ 2 := by
    constructor <;> intro h <;> nlinarith
  have hm' : 0 ≤ m₁₂ + m₂₁ := hm.le
  rw [← sq_lt_sq, ← key]
  constructor
  · intro h
    by_contra hc
    replace hc := not_lt.mp hc
    have : 2 * m₂₁ * (1 - m₂₁ + m₁₂) / (m₁₂ + m₂₁) ≤ 2 * m₁₂ * (1 - m₁₂ + m₂₁) / (m₁₂ + m₂₁) := by
      gcongr
    linarith
  · intro h
    gcongr

end Islands

end

end FixLoc
