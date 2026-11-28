import Mathlib

/-!
# Large islands under colonization (SI, Islands)

Two islands hold `n` individuals each. The state `(u, v)` counts the wild types on island 1 and
on island 2. In the colonization process only mutants reproduce. In each time step a uniformly
chosen mutant is the parent. The offspring of an island 1 parent is placed on island 2 with
probability `m₁₂` and on island 1 otherwise. The offspring of an island 2 parent is placed on
island 1 with probability `m₂₁` and on island 2 otherwise. The offspring replaces a uniformly
chosen individual of the target island.

`p n u v` is the probability of a fixation location on island 1 from the state `(u, v)`. As in
the other files, `p` enters through its first-step equations (`FirstStep`).

* `colonization_large_islands`: if `0 < m₂₁ < m₁₂ ≤ 1`, then from a single mutant on either
  island the probability of a fixation location on island 1 tends to `1` as `n → ∞`.
* `colonization_large_islands_symm`: if `0 < m₁₂ < m₂₁ ≤ 1`, the same probability tends to `0`.

The proof bounds `1 - p` by the potential `pot v / (u + 1)` while island 1 has at most `W`
wild types (`region`), and by the largest value of the potential on the line `u = W` before
(`entry`). The bound on the potential (`pot_pow_le`, `pot_le_exp`) uses Bernoulli's inequality.
-/

open Filter Topology

namespace FixLoc

noncomputable section

namespace LargeIslands

variable (n : ℕ) (m₁₂ m₂₁ : ℝ)

/-- Rate of mutant offspring arriving at an island 1 vertex, times `n`, per unit time in which
every mutant reproduces once on average. -/
def rate₁ (u v : ℕ) : ℝ := (1 - m₁₂) * ((n : ℝ) - u) + m₂₁ * ((n : ℝ) - v)

/-- Rate of mutant offspring arriving at an island 2 vertex, times `n`. -/
def rate₂ (u v : ℕ) : ℝ := m₁₂ * ((n : ℝ) - u) + (1 - m₂₁) * ((n : ℝ) - v)

/-- Probability that one time step replaces a wild type on island 1. -/
def pA (u v : ℕ) : ℝ := rate₁ n m₁₂ m₂₁ u v * u / ((n : ℝ) * (2 * n - u - v))

/-- Probability that one time step replaces a wild type on island 2. -/
def pB (u v : ℕ) : ℝ := rate₂ n m₁₂ m₂₁ u v * v / ((n : ℝ) * (2 * n - u - v))

/-- The first-step equations of the colonization process for the probability `p u v` of a
fixation location on island 1, from `u` wild types on island 1 and `v` on island 2. -/
structure FirstStep (p : ℕ → ℕ → ℝ) : Prop where
  /-- Island 1 has no wild type and island 2 has at least one. -/
  left : ∀ v, 1 ≤ v → v ≤ n → p 0 v = 0
  /-- Island 2 has no wild type and island 1 has at least one. -/
  right : ∀ u, 1 ≤ u → u ≤ n → p u 0 = 1
  /-- Wild types on both islands and at least one mutant. -/
  step : ∀ u v, 1 ≤ u → u ≤ n → 1 ≤ v → v ≤ n → u + v < 2 * n →
    p u v = pA n m₁₂ m₂₁ u v * p (u - 1) v + pB n m₁₂ m₂₁ u v * p u (v - 1)
      + (1 - pA n m₁₂ m₂₁ u v - pB n m₁₂ m₂₁ u v) * p u v

/-- Factor of the potential at `j` wild types on island 2. -/
def fac (q K W j : ℕ) : ℝ :=
  if j ≤ W then (q * j : ℝ) / (q * j - q + 1) else (j : ℝ) / (j - K)

/-- The potential `pot v = fac 1 * fac 2 * ... * fac v`. -/
def pot (q K W v : ℕ) : ℝ := ∏ j ∈ Finset.range v, fac q K W (j + 1)

/-- The rate bounds that hold while island 1 has at most `W` wild types. -/
structure Bounds (q K W : ℕ) : Prop where
  hq : 1 ≤ q
  hKW : K ≤ W
  hW1 : 1 ≤ W
  hWn : W < n
  r1 : ∀ u v, u ≤ W → v ≤ W →
    (q : ℝ) * rate₁ n m₁₂ m₂₁ u v ≤ ((q : ℝ) - 1) * rate₂ n m₁₂ m₂₁ u v
  r2 : ∀ u v, u ≤ W → v ≤ n → rate₁ n m₁₂ m₂₁ u v ≤ (K : ℝ) * rate₂ n m₁₂ m₂₁ u v

variable {n m₁₂ m₂₁}

/-! ### Rates and the jump chain -/

lemma rate₁_nonneg (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) {u v : ℕ} (hu : u ≤ n) (hv : v ≤ n) :
    0 ≤ rate₁ n m₁₂ m₂₁ u v := by
  have hu' : (u : ℝ) ≤ n := by exact_mod_cast hu
  have hv' : (v : ℝ) ≤ n := by exact_mod_cast hv
  unfold rate₁
  have h1 : 0 ≤ (1 - m₁₂) * ((n : ℝ) - u) := mul_nonneg (by linarith) (by linarith)
  have h2 : 0 ≤ m₂₁ * ((n : ℝ) - v) := mul_nonneg h21 (by linarith)
  linarith

lemma rate₂_pos (h12 : 0 < m₁₂) (h21' : m₂₁ < 1) {u v : ℕ} (hu : u ≤ n) (hv : v ≤ n)
    (huv : u + v < 2 * n) : 0 < rate₂ n m₁₂ m₂₁ u v := by
  have hu' : (u : ℝ) ≤ n := by exact_mod_cast hu
  have hv' : (v : ℝ) ≤ n := by exact_mod_cast hv
  unfold rate₂
  rcases Nat.lt_or_ge u n with h | h
  · have h' : (u : ℝ) < n := by exact_mod_cast h
    have h1 : 0 < m₁₂ * ((n : ℝ) - u) := mul_pos h12 (by linarith)
    have h2 : 0 ≤ (1 - m₂₁) * ((n : ℝ) - v) := mul_nonneg (by linarith) (by linarith)
    linarith
  · have hvlt : v < n := by omega
    have h' : (v : ℝ) < n := by exact_mod_cast hvlt
    have h1 : 0 ≤ m₁₂ * ((n : ℝ) - u) := mul_nonneg h12.le (by linarith)
    have h2 : 0 < (1 - m₂₁) * ((n : ℝ) - v) := mul_pos (by linarith) (by linarith)
    linarith

/-- The first-step equation without the steps that change nothing, written for `1 - p`. -/
lemma ratio {p : ℕ → ℕ → ℝ} (hp : FirstStep n m₁₂ m₂₁ p) (hn : 0 < n) {u v : ℕ}
    (hu : 1 ≤ u) (hun : u ≤ n) (hv : 1 ≤ v) (hvn : v ≤ n) (huv : u + v < 2 * n) :
    (rate₁ n m₁₂ m₂₁ u v * u + rate₂ n m₁₂ m₂₁ u v * v) * (1 - p u v)
      = rate₁ n m₁₂ m₂₁ u v * u * (1 - p (u - 1) v)
        + rate₂ n m₁₂ m₂₁ u v * v * (1 - p u (v - 1)) := by
  have h := hp.step u v hu hun hv hvn huv
  have hDpos : 0 < (n : ℝ) * (2 * n - u - v) := by
    have h1 : (u : ℝ) + v < 2 * n := by exact_mod_cast huv
    have h2 : (0 : ℝ) < n := by exact_mod_cast hn
    apply mul_pos h2
    linarith
  have hD0 : (n : ℝ) * (2 * n - u - v) ≠ 0 := hDpos.ne'
  unfold pA pB at h
  set A := rate₁ n m₁₂ m₂₁ u v * u with hA
  set B := rate₂ n m₁₂ m₂₁ u v * v with hB
  set D := (n : ℝ) * (2 * n - u - v) with hD
  have e : (A / D * p (u - 1) v + B / D * p u (v - 1) + (1 - A / D - B / D) * p u v) * D
      = A * p (u - 1) v + B * p u (v - 1) + (D - A - B) * p u v := by
    field_simp
  have h2 : p u v * D
      = (A / D * p (u - 1) v + B / D * p u (v - 1) + (1 - A / D - B / D) * p u v) * D := by
    rw [← h]
  rw [e] at h2
  linear_combination -h2

/-! ### The potential -/

lemma fac_of_le {q K W j : ℕ} (h : j ≤ W) :
    fac q K W j = (q * j : ℝ) / (q * j - q + 1) := by
  rw [fac, if_pos h]

lemma fac_of_lt {q K W j : ℕ} (h : W < j) : fac q K W j = (j : ℝ) / (j - K) := by
  rw [fac, if_neg (not_le.mpr h)]

lemma one_le_fac {q K W j : ℕ} (hq : 1 ≤ q) (hj : 1 ≤ j) (hKW : K ≤ W) :
    1 ≤ fac q K W j := by
  by_cases h : j ≤ W
  · rw [fac_of_le h]
    have hq' : (1 : ℝ) ≤ q := by exact_mod_cast hq
    have hj' : (1 : ℝ) ≤ j := by exact_mod_cast hj
    have hD : (0 : ℝ) < q * j - q + 1 := by
      nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ q) (by linarith : (0 : ℝ) ≤ j - 1)]
    rw [le_div_iff₀ hD]
    nlinarith
  · push_neg at h
    rw [fac_of_lt h]
    have hKj : (K : ℝ) < j := by exact_mod_cast (lt_of_le_of_lt hKW h)
    have hD : (0 : ℝ) < j - K := by linarith
    rw [le_div_iff₀ hD]
    have hK0 : (0 : ℝ) ≤ K := Nat.cast_nonneg K
    linarith

lemma pot_zero (q K W : ℕ) : pot q K W 0 = 1 := by
  simp [pot]

lemma pot_succ (q K W v : ℕ) : pot q K W (v + 1) = pot q K W v * fac q K W (v + 1) := by
  unfold pot
  rw [Finset.prod_range_succ]

lemma one_le_pot {q K W : ℕ} (hq : 1 ≤ q) (hKW : K ≤ W) (v : ℕ) : 1 ≤ pot q K W v := by
  induction v with
  | zero => simp [pot_zero]
  | succ v ih =>
    rw [pot_succ]
    have h := one_le_fac (q := q) (K := K) (W := W) (j := v + 1) hq (by omega) hKW
    nlinarith

lemma pot_mono {q K W : ℕ} (hq : 1 ≤ q) (hKW : K ≤ W) {v w : ℕ} (h : v ≤ w) :
    pot q K W v ≤ pot q K W w := by
  induction w, h using Nat.le_induction with
  | base => exact le_refl _
  | succ w hvw ih =>
    rw [pot_succ]
    have h1 := one_le_fac (q := q) (K := K) (W := W) (j := w + 1) hq (by omega) hKW
    have h2 := one_le_pot hq hKW w
    nlinarith

/-- The supermartingale inequality of the potential at one state. -/
lemma key_step {q K W v : ℕ} {a b : ℝ} (hq : 1 ≤ q) (hKW : K ≤ W) (hv : 1 ≤ v)
    (ha : 0 ≤ a) (hb : 0 ≤ b)
    (h1 : v ≤ W → (q : ℝ) * a ≤ ((q : ℝ) - 1) * b) (h2 : W < v → a ≤ (K : ℝ) * b) :
    a * fac q K W v ≤ b * v * (fac q K W v - 1) := by
  have hv' : (1 : ℝ) ≤ v := by exact_mod_cast hv
  by_cases h : v ≤ W
  · rw [fac_of_le h]
    have hq' : (1 : ℝ) ≤ q := by exact_mod_cast hq
    have hD : (0 : ℝ) < q * v - q + 1 := by
      nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ q) (by linarith : (0 : ℝ) ≤ v - 1)]
    have hD0 : (q * v - q + 1 : ℝ) ≠ 0 := hD.ne'
    have lhs : a * ((q * v : ℝ) / (q * v - q + 1)) = a * q * v / (q * v - q + 1) := by ring
    have rhs : b * v * ((q * v : ℝ) / (q * v - q + 1) - 1) = b * v * (q - 1) / (q * v - q + 1) := by
      field_simp
      ring
    rw [lhs, rhs]
    apply div_le_div_of_nonneg_right _ hD.le
    have h1' := h1 h
    nlinarith [mul_le_mul_of_nonneg_right h1' (by linarith : (0 : ℝ) ≤ v)]
  · push_neg at h
    rw [fac_of_lt h]
    have hKv : (K : ℝ) < v := by exact_mod_cast (lt_of_le_of_lt hKW h)
    have hE : (0 : ℝ) < v - K := by linarith
    have hE0 : ((v : ℝ) - K) ≠ 0 := hE.ne'
    have lhs : a * ((v : ℝ) / (v - K)) = a * v / (v - K) := by ring
    have rhs : b * v * ((v : ℝ) / (v - K) - 1) = b * v * K / (v - K) := by
      field_simp
      ring
    rw [lhs, rhs]
    apply div_le_div_of_nonneg_right _ hE.le
    have h2' := h2 h
    nlinarith [mul_le_mul_of_nonneg_right h2' (by linarith : (0 : ℝ) ≤ v)]

/-! ### Bounds on `1 - p` by induction over the states -/

/-- Near the end of the process `1 - p` is at most the potential. -/
theorem region {p : ℕ → ℕ → ℝ} (hp : FirstStep n m₁₂ m₂₁ p) (hn : 0 < n)
    (h12 : 0 < m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ < 1)
    {q K W : ℕ} (hb : Bounds n m₁₂ m₂₁ q K W) :
    ∀ s u v, u + v = s → u ≤ W → v ≤ n → 1 ≤ u + v →
      ((u : ℝ) + 1) * (1 - p u v) ≤ pot q K W v := by
  intro s
  induction s with
  | zero => intro u v hs _ _ h1; omega
  | succ s ih =>
    intro u v hs hu hv h1
    have hWn := hb.hWn
    rcases Nat.eq_zero_or_pos u with rfl | hu0
    · have hv1 : 1 ≤ v := by omega
      rw [hp.left v hv1 hv]
      simpa using one_le_pot hb.hq hb.hKW v
    · rcases Nat.eq_zero_or_pos v with rfl | hv0
      · rw [hp.right u hu0 (by omega)]
        simp [pot_zero]
      · obtain ⟨u', rfl⟩ : ∃ u', u = u' + 1 := ⟨u - 1, by omega⟩
        obtain ⟨v', rfl⟩ : ∃ v', v = v' + 1 := ⟨v - 1, by omega⟩
        have huv : u' + 1 + (v' + 1) < 2 * n := by omega
        have hrat := ratio hp hn (u := u' + 1) (v := v' + 1) (by omega) (by omega) (by omega) hv huv
        simp only [Nat.add_sub_cancel] at hrat
        have ih1 := ih u' (v' + 1) (by omega) (by omega) hv (by omega)
        have ih2 := ih (u' + 1) v' (by omega) hu (by omega) (by omega)
        have ha : 0 ≤ rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₁_nonneg h12' h21 (by omega) hv
        have hbpos : 0 < rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₂_pos h12 h21' (by omega) hv huv
        have hkey := key_step (q := q) (K := K) (W := W) (v := v' + 1) hb.hq hb.hKW (by omega)
          ha hbpos.le (fun h => hb.r1 (u' + 1) (v' + 1) hu h) (fun _ => hb.r2 (u' + 1) (v' + 1) hu hv)
        have hG : 0 ≤ pot q K W v' := le_trans zero_le_one (one_le_pot hb.hq hb.hKW v')
        rw [pot_succ] at ih1 ⊢
        set a := rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) with ha_def
        set b := rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) with hb_def
        set G := pot q K W v' with hG_def
        set φ := fac q K W (v' + 1) with hφ_def
        push_cast at ih2 hrat hkey ⊢
        have hx : (0 : ℝ) ≤ (u' : ℝ) + 1 := by positivity
        have hy : (0 : ℝ) < (v' : ℝ) + 1 := by positivity
        have hpos : 0 < a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1) :=
          add_pos_of_nonneg_of_pos (mul_nonneg ha hx) (mul_pos hbpos hy)
        have h1 : a * ((u' : ℝ) + 1 + 1) * (((u' : ℝ) + 1) * (1 - p u' (v' + 1)))
            ≤ a * ((u' : ℝ) + 1 + 1) * (G * φ) :=
          mul_le_mul_of_nonneg_left ih1 (mul_nonneg ha (by positivity))
        have h2 : b * ((v' : ℝ) + 1) * (((u' : ℝ) + 1 + 1) * (1 - p (u' + 1) v'))
            ≤ b * ((v' : ℝ) + 1) * G :=
          mul_le_mul_of_nonneg_left ih2 (mul_nonneg hbpos.le hy.le)
        have h3 : G * (a * φ) ≤ G * (b * ((v' : ℝ) + 1) * (φ - 1)) :=
          mul_le_mul_of_nonneg_left hkey hG
        have e1 : (a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1))
              * (((u' : ℝ) + 1 + 1) * (1 - p (u' + 1) (v' + 1)))
            = a * ((u' : ℝ) + 1 + 1) * (((u' : ℝ) + 1) * (1 - p u' (v' + 1)))
              + b * ((v' : ℝ) + 1) * (((u' : ℝ) + 1 + 1) * (1 - p (u' + 1) v')) := by
          linear_combination ((u' : ℝ) + 1 + 1) * hrat
        refine le_of_mul_le_mul_left ?_ hpos
        nlinarith [h1, h2, h3, e1]

/-- Before the end, `1 - p` is at most the largest value of the potential on the line `u = W`. -/
theorem entry {p : ℕ → ℕ → ℝ} (hp : FirstStep n m₁₂ m₂₁ p) (hn : 0 < n)
    (h12 : 0 < m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ < 1)
    {q K W : ℕ} (hb : Bounds n m₁₂ m₂₁ q K W) :
    ∀ s u v, u + v = s → W ≤ u → u ≤ n → v ≤ n → u + v < 2 * n →
      ((W : ℝ) + 1) * (1 - p u v) ≤ pot q K W n := by
  intro s
  induction s with
  | zero =>
    intro u v hs hWu _ _ _
    have := hb.hW1
    omega
  | succ s ih =>
    intro u v hs hWu hun hvn huv
    have hW1 := hb.hW1
    rcases Nat.eq_or_lt_of_le hWu with heq | hlt
    · have h1 := region hp hn h12 h12' h21 h21' hb (u + v) u v rfl (by omega) hvn (by omega)
      have hcast : (u : ℝ) = (W : ℝ) := by rw [heq]
      rw [hcast] at h1
      exact le_trans h1 (pot_mono hb.hq hb.hKW hvn)
    · rcases Nat.eq_zero_or_pos v with rfl | hv0
      · rw [hp.right u (by omega) hun]
        simp only [sub_self, mul_zero]
        exact le_trans zero_le_one (one_le_pot hb.hq hb.hKW n)
      · obtain ⟨u', rfl⟩ : ∃ u', u = u' + 1 := ⟨u - 1, by omega⟩
        obtain ⟨v', rfl⟩ : ∃ v', v = v' + 1 := ⟨v - 1, by omega⟩
        have hrat := ratio hp hn (u := u' + 1) (v := v' + 1) (by omega) hun (by omega) hvn huv
        simp only [Nat.add_sub_cancel] at hrat
        have ih1 := ih u' (v' + 1) (by omega) (by omega) (by omega) hvn (by omega)
        have ih2 := ih (u' + 1) v' (by omega) hWu hun (by omega) (by omega)
        have ha : 0 ≤ rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₁_nonneg h12' h21 hun hvn
        have hbpos : 0 < rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₂_pos h12 h21' hun hvn huv
        set a := rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) with ha_def
        set b := rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) with hb_def
        set M := pot q K W n with hM_def
        push_cast at hrat
        have hx : (0 : ℝ) ≤ (u' : ℝ) + 1 := by positivity
        have hy : (0 : ℝ) < (v' : ℝ) + 1 := by positivity
        have hpos : 0 < a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1) :=
          add_pos_of_nonneg_of_pos (mul_nonneg ha hx) (mul_pos hbpos hy)
        have h1 : a * ((u' : ℝ) + 1) * (((W : ℝ) + 1) * (1 - p u' (v' + 1)))
            ≤ a * ((u' : ℝ) + 1) * M :=
          mul_le_mul_of_nonneg_left ih1 (mul_nonneg ha hx)
        have h2 : b * ((v' : ℝ) + 1) * (((W : ℝ) + 1) * (1 - p (u' + 1) v'))
            ≤ b * ((v' : ℝ) + 1) * M :=
          mul_le_mul_of_nonneg_left ih2 (mul_nonneg hbpos.le hy.le)
        have e1 : (a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1))
              * (((W : ℝ) + 1) * (1 - p (u' + 1) (v' + 1)))
            = a * ((u' : ℝ) + 1) * (((W : ℝ) + 1) * (1 - p u' (v' + 1)))
              + b * ((v' : ℝ) + 1) * (((W : ℝ) + 1) * (1 - p (u' + 1) v')) := by
          linear_combination ((W : ℝ) + 1) * hrat
        refine le_of_mul_le_mul_left ?_ hpos
        nlinarith [h1, h2, e1]

/-- The probability `p` is at most `1` at every state with wild types and mutants. -/
theorem one_sub_nonneg {p : ℕ → ℕ → ℝ} (hp : FirstStep n m₁₂ m₂₁ p) (hn : 0 < n)
    (h12 : 0 < m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ < 1) :
    ∀ s u v, u + v = s → u ≤ n → v ≤ n → 1 ≤ u + v → u + v < 2 * n → 0 ≤ 1 - p u v := by
  intro s
  induction s with
  | zero => intro u v hs _ _ h1 _; omega
  | succ s ih =>
    intro u v hs hun hvn h1 huv
    rcases Nat.eq_zero_or_pos u with rfl | hu0
    · rw [hp.left v (by omega) hvn]
      norm_num
    · rcases Nat.eq_zero_or_pos v with rfl | hv0
      · rw [hp.right u hu0 hun]
        norm_num
      · obtain ⟨u', rfl⟩ : ∃ u', u = u' + 1 := ⟨u - 1, by omega⟩
        obtain ⟨v', rfl⟩ : ∃ v', v = v' + 1 := ⟨v - 1, by omega⟩
        have hrat := ratio hp hn (u := u' + 1) (v := v' + 1) (by omega) hun (by omega) hvn huv
        simp only [Nat.add_sub_cancel] at hrat
        have ih1 := ih u' (v' + 1) (by omega) (by omega) hvn (by omega) (by omega)
        have ih2 := ih (u' + 1) v' (by omega) hun (by omega) (by omega) (by omega)
        have ha : 0 ≤ rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₁_nonneg h12' h21 hun hvn
        have hbpos : 0 < rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) := rate₂_pos h12 h21' hun hvn huv
        set a := rate₁ n m₁₂ m₂₁ (u' + 1) (v' + 1) with ha_def
        set b := rate₂ n m₁₂ m₂₁ (u' + 1) (v' + 1) with hb_def
        push_cast at hrat
        have hx : (0 : ℝ) ≤ (u' : ℝ) + 1 := by positivity
        have hy : (0 : ℝ) < (v' : ℝ) + 1 := by positivity
        have hpos : 0 < a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1) :=
          add_pos_of_nonneg_of_pos (mul_nonneg ha hx) (mul_pos hbpos hy)
        have h1 : 0 ≤ a * ((u' : ℝ) + 1) * (1 - p u' (v' + 1)) :=
          mul_nonneg (mul_nonneg ha hx) ih1
        have h2 : 0 ≤ b * ((v' : ℝ) + 1) * (1 - p (u' + 1) v') :=
          mul_nonneg (mul_nonneg hbpos.le hy.le) ih2
        have h3 : (a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1)) * 0
            ≤ (a * ((u' : ℝ) + 1) + b * ((v' : ℝ) + 1)) * (1 - p (u' + 1) (v' + 1)) := by
          rw [mul_zero, hrat]
          linarith
        exact le_of_mul_le_mul_left h3 hpos

/-! ### Size of the potential -/

/-- Bernoulli's inequality in the form `X^(k+1) + (k+1) X^k ≤ (X+1)^(k+1)`. -/
lemma pow_succ_add_le (X : ℝ) (hX : 0 ≤ X) (k : ℕ) :
    X ^ (k + 1) + ((k : ℝ) + 1) * X ^ k ≤ (X + 1) ^ (k + 1) := by
  induction k with
  | zero => simp
  | succ k ih =>
    have e : (X ^ (k + 1) + ((k : ℝ) + 1) * X ^ k) * (X + 1)
        = X ^ (k + 1 + 1) + (((k + 1 : ℕ) : ℝ) + 1) * X ^ (k + 1) + ((k : ℝ) + 1) * X ^ k := by
      push_cast
      ring
    have h1 : (X ^ (k + 1) + ((k : ℝ) + 1) * X ^ k) * (X + 1) ≤ (X + 1) ^ (k + 1) * (X + 1) :=
      mul_le_mul_of_nonneg_right ih (by linarith)
    have h2 : 0 ≤ ((k : ℝ) + 1) * X ^ k := mul_nonneg (by positivity) (pow_nonneg hX k)
    calc X ^ (k + 1 + 1) + (((k + 1 : ℕ) : ℝ) + 1) * X ^ (k + 1)
        ≤ (X ^ (k + 1) + ((k : ℝ) + 1) * X ^ k) * (X + 1) := by rw [e]; linarith
      _ ≤ (X + 1) ^ (k + 1) * (X + 1) := h1
      _ = (X + 1) ^ (k + 1 + 1) := by ring

lemma fac_succ_of_le {q K W v : ℕ} (h : v + 1 ≤ W) :
    fac q K W (v + 1) = ((q : ℝ) * ((v : ℝ) + 1)) / ((q : ℝ) * v + 1) := by
  rw [fac_of_le h]
  push_cast
  rw [show (q : ℝ) * ((v : ℝ) + 1) - (q : ℝ) + 1 = (q : ℝ) * v + 1 by ring]

/-- The potential grows at most like `v ^ ((q - 1) / q)` on the first `W` values. -/
lemma pot_pow_le {q K W : ℕ} (hq : 1 ≤ q) (hKW : K ≤ W) :
    ∀ v, 1 ≤ v → v ≤ W → pot q K W v ^ q ≤ (q : ℝ) ^ q * (v : ℝ) ^ (q - 1) := by
  obtain ⟨k, rfl⟩ : ∃ k, q = k + 1 := ⟨q - 1, by omega⟩
  intro v hv
  induction v, hv using Nat.le_induction with
  | base =>
    intro hW
    have h1 : pot (k + 1) K W 1 = fac (k + 1) K W 1 := by simp [pot]
    rw [h1, fac_of_le hW]
    simp
  | succ v hv ih =>
    intro hW
    have ih' := ih (by omega)
    simp only [Nat.add_sub_cancel] at ih' ⊢
    rw [pot_succ, mul_pow, fac_succ_of_le hW, div_pow]
    push_cast at ih' ⊢
    have hv0 : (0 : ℝ) ≤ v := Nat.cast_nonneg v
    have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg k
    have hB : 0 < (((k : ℝ) + 1) * v + 1) ^ (k + 1) := by positivity
    have bern := pow_succ_add_le (((k : ℝ) + 1) * v) (by positivity) k
    have e : (((k : ℝ) + 1) * v) ^ (k + 1) + ((k : ℝ) + 1) * (((k : ℝ) + 1) * v) ^ k
        = ((k : ℝ) + 1) ^ (k + 1) * (v : ℝ) ^ k * ((v : ℝ) + 1) := by
      simp only [mul_pow]
      ring
    have key : (v : ℝ) ^ k * (((k : ℝ) + 1) * ((v : ℝ) + 1)) ^ (k + 1)
        ≤ ((v : ℝ) + 1) ^ k * (((k : ℝ) + 1) * v + 1) ^ (k + 1) := by
      have e2 : (v : ℝ) ^ k * (((k : ℝ) + 1) * ((v : ℝ) + 1)) ^ (k + 1)
          = ((v : ℝ) + 1) ^ k * (((k : ℝ) + 1) ^ (k + 1) * (v : ℝ) ^ k * ((v : ℝ) + 1)) := by
        simp only [mul_pow]
        ring
      rw [e2]
      apply mul_le_mul_of_nonneg_left _ (by positivity)
      linarith [bern, e]
    have hpot0 : 0 ≤ pot (k + 1) K W v := le_trans zero_le_one (one_le_pot hq hKW v)
    calc pot (k + 1) K W v ^ (k + 1)
          * ((((k : ℝ) + 1) * ((v : ℝ) + 1)) ^ (k + 1) / (((k : ℝ) + 1) * v + 1) ^ (k + 1))
        ≤ ((k : ℝ) + 1) ^ (k + 1) * (v : ℝ) ^ k
          * ((((k : ℝ) + 1) * ((v : ℝ) + 1)) ^ (k + 1) / (((k : ℝ) + 1) * v + 1) ^ (k + 1)) :=
          mul_le_mul_of_nonneg_right ih' (div_nonneg (by positivity) hB.le)
      _ = ((k : ℝ) + 1) ^ (k + 1) * ((v : ℝ) ^ k * (((k : ℝ) + 1) * ((v : ℝ) + 1)) ^ (k + 1)
          / (((k : ℝ) + 1) * v + 1) ^ (k + 1)) := by ring
      _ ≤ ((k : ℝ) + 1) ^ (k + 1) * ((v : ℝ) + 1) ^ k := by
          apply mul_le_mul_of_nonneg_left _ (by positivity)
          rw [div_le_iff₀ hB]
          exact key

/-- Each factor beyond `W` is at most `exp (K / L)`. -/
lemma fac_le_exp {q K W j : ℕ} {L : ℝ} (hL : 0 < L) (hWL : L ≤ (W : ℝ) + 1 - K) (hj : W < j) :
    fac q K W j ≤ Real.exp ((K : ℝ) / L) := by
  rw [fac_of_lt hj]
  have hj' : (W : ℝ) + 1 ≤ j := by exact_mod_cast hj
  have hpos : 0 < (j : ℝ) - K := by linarith
  have hne : (j : ℝ) - K ≠ 0 := hpos.ne'
  have e : (j : ℝ) / (j - K) = (K : ℝ) / (j - K) + 1 := by
    field_simp
    ring
  rw [e]
  calc (K : ℝ) / (j - K) + 1 ≤ Real.exp ((K : ℝ) / (j - K)) := Real.add_one_le_exp _
    _ ≤ Real.exp ((K : ℝ) / L) := by
        apply Real.exp_le_exp.mpr
        exact div_le_div_of_nonneg_left (Nat.cast_nonneg K) hL (by linarith)

/-- Beyond `W` the potential grows at most by the factor `exp c` per step. -/
lemma pot_le_exp {q K W : ℕ} (hq : 1 ≤ q) (hKW : K ≤ W) {c : ℝ}
    (hfac : ∀ j, W < j → fac q K W j ≤ Real.exp c) :
    ∀ m, W ≤ m → pot q K W m ≤ pot q K W W * Real.exp c ^ (m - W) := by
  intro m hm
  induction m, hm using Nat.le_induction with
  | base => simp
  | succ m hm ih =>
    rw [pot_succ, show m + 1 - W = (m - W) + 1 by omega, pow_succ]
    have h1 := hfac (m + 1) (by omega)
    have h3 : 0 ≤ fac q K W (m + 1) := le_trans zero_le_one (one_le_fac hq (by omega) hKW)
    have h4 : 0 ≤ pot q K W W * Real.exp c ^ (m - W) :=
      mul_nonneg (le_trans zero_le_one (one_le_pot hq hKW W)) (pow_nonneg (Real.exp_pos c).le _)
    calc pot q K W m * fac q K W (m + 1)
        ≤ (pot q K W W * Real.exp c ^ (m - W)) * Real.exp c := mul_le_mul ih h1 h3 h4
      _ = pot q K W W * (Real.exp c ^ (m - W) * Real.exp c) := by ring

/-! ### The rate bounds -/

lemma r1_bound (h12 : 0 < m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (hlt : m₂₁ < m₁₂)
    {q W : ℕ} (hq : 1 ≤ q) (hqd : 1 ≤ (q : ℝ) * (m₁₂ - m₂₁))
    (hW : (W : ℝ) ≤ (m₁₂ - m₂₁) * n / 2) {u v : ℕ} (hu : u ≤ W) (hv : v ≤ W)
    (hun : u ≤ n) (hvn : v ≤ n) :
    (q : ℝ) * rate₁ n m₁₂ m₂₁ u v ≤ ((q : ℝ) - 1) * rate₂ n m₁₂ m₂₁ u v := by
  have hu' : (u : ℝ) ≤ W := by exact_mod_cast hu
  have hv' : (v : ℝ) ≤ W := by exact_mod_cast hv
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hu0 : (0 : ℝ) ≤ u := Nat.cast_nonneg u
  have hv0 : (0 : ℝ) ≤ v := Nat.cast_nonneg v
  have hq' : (1 : ℝ) ≤ q := by exact_mod_cast hq
  have hr1 : rate₁ n m₁₂ m₂₁ u v ≤ (1 - (m₁₂ - m₂₁)) * n := by
    unfold rate₁
    nlinarith [mul_nonneg (sub_nonneg.mpr h12') hu0, mul_nonneg h21 hv0]
  have hr2 : (n : ℝ) ≤ rate₂ n m₁₂ m₂₁ u v := by
    unfold rate₂
    nlinarith [mul_le_mul_of_nonneg_left hu' h12.le,
      mul_le_mul_of_nonneg_left hv' (by linarith : (0 : ℝ) ≤ 1 - m₂₁),
      mul_le_mul_of_nonneg_left hW (by linarith : (0 : ℝ) ≤ 1 + (m₁₂ - m₂₁)),
      mul_nonneg (mul_nonneg (by linarith : (0 : ℝ) ≤ m₁₂ - m₂₁)
        (by linarith : (0 : ℝ) ≤ 1 - (m₁₂ - m₂₁))) hn0]
  nlinarith [mul_le_mul_of_nonneg_left hr1 (by linarith : (0 : ℝ) ≤ q),
    mul_nonneg (by linarith : (0 : ℝ) ≤ (q : ℝ) * (m₁₂ - m₂₁) - 1) hn0,
    mul_le_mul_of_nonneg_left hr2 (by linarith : (0 : ℝ) ≤ (q : ℝ) - 1)]

lemma r2_bound (h12 : 0 < m₁₂) (h12' : m₁₂ ≤ 1) (h21 : 0 ≤ m₂₁) (h21' : m₂₁ < 1)
    (hlt : m₂₁ < m₁₂) {K W : ℕ} (hKm : 2 ≤ (K : ℝ) * m₁₂)
    (hW : (W : ℝ) ≤ (m₁₂ - m₂₁) * n / 2) {u v : ℕ} (hu : u ≤ W) (hun : u ≤ n) (hvn : v ≤ n) :
    rate₁ n m₁₂ m₂₁ u v ≤ (K : ℝ) * rate₂ n m₁₂ m₂₁ u v := by
  have hu' : (u : ℝ) ≤ W := by exact_mod_cast hu
  have hvn' : (v : ℝ) ≤ n := by exact_mod_cast hvn
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  have hu0 : (0 : ℝ) ≤ u := Nat.cast_nonneg u
  have hv0 : (0 : ℝ) ≤ v := Nat.cast_nonneg v
  have hK0 : (0 : ℝ) ≤ K := Nat.cast_nonneg K
  have hr1 : rate₁ n m₁₂ m₂₁ u v ≤ n := by
    unfold rate₁
    nlinarith [mul_nonneg (sub_nonneg.mpr h12') hu0, mul_nonneg h21 hv0,
      mul_nonneg (by linarith : (0 : ℝ) ≤ m₁₂ - m₂₁) hn0]
  have hr2 : m₁₂ * n / 2 ≤ rate₂ n m₁₂ m₂₁ u v := by
    unfold rate₂
    nlinarith [mul_le_mul_of_nonneg_left hu' h12.le,
      mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - m₂₁) (by linarith : (0 : ℝ) ≤ n - v),
      mul_le_mul_of_nonneg_left hW h12.le,
      mul_nonneg h12.le (mul_nonneg (by linarith : (0 : ℝ) ≤ 1 - (m₁₂ - m₂₁)) hn0)]
  nlinarith [mul_le_mul_of_nonneg_left hr2 hK0,
    mul_nonneg (by linarith : (0 : ℝ) ≤ (K : ℝ) * m₁₂ - 2) hn0]

/-! ### The theorem -/

/-- **Large islands under colonization.** If `0 < m₂₁ < m₁₂ ≤ 1`, then from a single mutant on
island 1 (the state `(n - 1, n)`) or on island 2 (the state `(n, n - 1)`), the probability of a
fixation location on island 1 tends to `1` as the island size `n` tends to infinity. -/
theorem colonization_large_islands {m₁₂ m₂₁ : ℝ} (h21 : 0 < m₂₁) (hlt : m₂₁ < m₁₂)
    (h12 : m₁₂ ≤ 1) (p : ℕ → ℕ → ℕ → ℝ) (hp : ∀ n, FirstStep n m₁₂ m₂₁ (p n)) :
    Tendsto (fun n => p n (n - 1) n) atTop (𝓝 1)
      ∧ Tendsto (fun n => p n n (n - 1)) atTop (𝓝 1) := by
  have h12pos : 0 < m₁₂ := by linarith
  have h21' : m₂₁ < 1 := by linarith
  have hd : 0 < m₁₂ - m₂₁ := by linarith
  have hd1 : m₁₂ - m₂₁ < 1 := by linarith
  have hd0 : m₁₂ - m₂₁ ≠ 0 := hd.ne'
  have hm0 : m₁₂ ≠ 0 := h12pos.ne'
  -- the constants `q ≥ 1 / d` and `K ≥ 2 / m₁₂`
  obtain ⟨q, hq_def⟩ : ∃ q : ℕ, q = ⌈1 / (m₁₂ - m₂₁)⌉₊ := ⟨_, rfl⟩
  obtain ⟨K, hK_def⟩ : ∃ K : ℕ, K = ⌈2 / m₁₂⌉₊ := ⟨_, rfl⟩
  have hqd : 1 ≤ (q : ℝ) * (m₁₂ - m₂₁) := by
    have h := Nat.le_ceil (1 / (m₁₂ - m₂₁))
    rw [← hq_def] at h
    have e : 1 / (m₁₂ - m₂₁) * (m₁₂ - m₂₁) = 1 := by field_simp
    nlinarith [mul_le_mul_of_nonneg_right h hd.le]
  have hq1 : 1 ≤ q := by
    have h : (1 : ℝ) ≤ q := by
      nlinarith [mul_nonneg (Nat.cast_nonneg q : (0 : ℝ) ≤ q) (by linarith : (0 : ℝ) ≤ 1 - (m₁₂ - m₂₁))]
    exact_mod_cast h
  have hKm : 2 ≤ (K : ℝ) * m₁₂ := by
    have h := Nat.le_ceil (2 / m₁₂)
    rw [← hK_def] at h
    have e : 2 / m₁₂ * m₁₂ = 2 := by field_simp
    nlinarith [mul_le_mul_of_nonneg_right h h12pos.le]
  -- the bound for one island size `n`
  have main : ∀ n : ℕ, 4 * (K : ℝ) / (m₁₂ - m₂₁) ≤ n → 2 / (m₁₂ - m₂₁) ≤ (n : ℝ) → 2 ≤ n →
      ∀ u v, n - 1 ≤ u → u ≤ n → n - 1 ≤ v → v ≤ n → u + v < 2 * n →
      0 ≤ 1 - p n u v ∧ (1 - p n u v) ^ q
        ≤ 2 * ((q : ℝ) ^ q * Real.exp (4 * K / (m₁₂ - m₂₁)) ^ q) / ((m₁₂ - m₂₁) * n) := by
    intro n hn4 hn2 hn2' u v hu1 hun hv1 hvn huv
    have hnpos : 0 < n := by omega
    have hn' : (0 : ℝ) < n := by exact_mod_cast hnpos
    have hn0 : (n : ℝ) ≠ 0 := hn'.ne'
    have hdn : 2 ≤ (m₁₂ - m₂₁) * n := by
      have e : (m₁₂ - m₂₁) * (2 / (m₁₂ - m₂₁)) = 2 := by field_simp
      nlinarith [mul_le_mul_of_nonneg_left hn2 hd.le]
    obtain ⟨W, hW_def⟩ : ∃ W : ℕ, W = ⌊(m₁₂ - m₂₁) * n / 2⌋₊ := ⟨_, rfl⟩
    have hWle : (W : ℝ) ≤ (m₁₂ - m₂₁) * n / 2 := by
      rw [hW_def]
      exact Nat.floor_le (by have := mul_pos hd hn'; linarith)
    have hWgt : (m₁₂ - m₂₁) * n / 2 < (W : ℝ) + 1 := by
      rw [hW_def]
      exact Nat.lt_floor_add_one _
    have hW1 : 1 ≤ W := by
      rw [hW_def]
      apply Nat.le_floor
      push_cast
      linarith
    have hKle : (K : ℝ) ≤ (m₁₂ - m₂₁) * n / 4 := by
      have e : (m₁₂ - m₂₁) * (4 * (K : ℝ) / (m₁₂ - m₂₁)) = 4 * K := by field_simp
      nlinarith [mul_le_mul_of_nonneg_left hn4 hd.le]
    have hL : (m₁₂ - m₂₁) * n / 4 ≤ (W : ℝ) + 1 - K := by linarith
    have hLpos : 0 < (m₁₂ - m₂₁) * n / 4 := by have := mul_pos hd hn'; linarith
    have hKW : K ≤ W := by
      have h : (K : ℝ) < (W : ℝ) + 1 := by linarith
      have h' : K < W + 1 := by exact_mod_cast h
      omega
    have hWn : W < n := by
      have h1 : (m₁₂ - m₂₁) * n < 1 * n := mul_lt_mul_of_pos_right hd1 hn'
      have h : (W : ℝ) < n := by linarith
      exact_mod_cast h
    have hb : Bounds n m₁₂ m₂₁ q K W :=
      { hq := hq1, hKW := hKW, hW1 := hW1, hWn := hWn,
        r1 := fun u v hu hv => r1_bound h12pos h12 h21.le hlt hq1 hqd hWle hu hv (by omega) (by omega),
        r2 := fun u v hu hvn => r2_bound h12pos h12 h21.le h21' hlt hKm hWle hu (by omega) hvn }
    have hent := entry (hp n) hnpos h12pos h12 h21.le h21' hb (u + v) u v rfl (by omega) hun hvn huv
    have hnn := one_sub_nonneg (hp n) hnpos h12pos h12 h21.le h21' (u + v) u v rfl hun hvn
      (by omega) huv
    -- the potential at `n`
    have hpotn : pot q K W n ≤ pot q K W W * Real.exp (4 * K / (m₁₂ - m₂₁)) := by
      have h := pot_le_exp hq1 hKW (c := (K : ℝ) / ((m₁₂ - m₂₁) * n / 4))
        (fun j hj => fac_le_exp hLpos hL hj) n hWn.le
      have hc : 0 ≤ (K : ℝ) / ((m₁₂ - m₂₁) * n / 4) := div_nonneg (Nat.cast_nonneg K) hLpos.le
      have h2 : Real.exp ((K : ℝ) / ((m₁₂ - m₂₁) * n / 4)) ^ (n - W)
          ≤ Real.exp ((K : ℝ) / ((m₁₂ - m₂₁) * n / 4)) ^ n :=
        pow_le_pow_right₀ (Real.one_le_exp hc) (Nat.sub_le n W)
      have h3 : Real.exp ((K : ℝ) / ((m₁₂ - m₂₁) * n / 4)) ^ n
          = Real.exp (4 * K / (m₁₂ - m₂₁)) := by
        rw [← Real.exp_nat_mul]
        congr 1
        field_simp
      have hW0 : 0 ≤ pot q K W W := le_trans zero_le_one (one_le_pot hq1 hKW W)
      calc pot q K W n ≤ pot q K W W * Real.exp ((K : ℝ) / ((m₁₂ - m₂₁) * n / 4)) ^ (n - W) := h
        _ ≤ pot q K W W * Real.exp ((K : ℝ) / ((m₁₂ - m₂₁) * n / 4)) ^ n :=
          mul_le_mul_of_nonneg_left h2 hW0
        _ = pot q K W W * Real.exp (4 * K / (m₁₂ - m₂₁)) := by rw [h3]
    have hpotW := pot_pow_le hq1 hKW W hW1 le_rfl
    -- combine
    set E := Real.exp (4 * K / (m₁₂ - m₂₁)) with hE_def
    set X := 1 - p n u v with hX_def
    have hE0 : 0 ≤ E := (Real.exp_pos _).le
    have hWX : ((W : ℝ) + 1) * X ≤ pot q K W W * E := le_trans hent hpotn
    have hpow : (((W : ℝ) + 1) * X) ^ q ≤ (q : ℝ) ^ q * (W : ℝ) ^ (q - 1) * E ^ q := by
      calc (((W : ℝ) + 1) * X) ^ q ≤ (pot q K W W * E) ^ q :=
            pow_le_pow_left₀ (mul_nonneg (by positivity) hnn) hWX q
        _ = pot q K W W ^ q * E ^ q := mul_pow _ _ _
        _ ≤ ((q : ℝ) ^ q * (W : ℝ) ^ (q - 1)) * E ^ q :=
            mul_le_mul_of_nonneg_right hpotW (pow_nonneg hE0 q)
        _ = (q : ℝ) ^ q * (W : ℝ) ^ (q - 1) * E ^ q := by ring
    have hWW : (W : ℝ) ^ (q - 1) ≤ ((W : ℝ) + 1) ^ (q - 1) :=
      pow_le_pow_left₀ (Nat.cast_nonneg W) (by linarith) (q - 1)
    have hsplit : ((W : ℝ) + 1) ^ q = ((W : ℝ) + 1) ^ (q - 1) * ((W : ℝ) + 1) := by
      rw [← pow_succ]
      congr 1
      omega
    have hfinal : ((W : ℝ) + 1) * X ^ q ≤ (q : ℝ) ^ q * E ^ q := by
      have hpos : 0 < ((W : ℝ) + 1) ^ (q - 1) := by positivity
      have h1 : ((W : ℝ) + 1) ^ (q - 1) * (((W : ℝ) + 1) * X ^ q)
          ≤ ((W : ℝ) + 1) ^ (q - 1) * ((q : ℝ) ^ q * E ^ q) := by
        have e : ((W : ℝ) + 1) ^ (q - 1) * (((W : ℝ) + 1) * X ^ q) = (((W : ℝ) + 1) * X) ^ q := by
          rw [mul_pow, hsplit]
          ring
        rw [e]
        calc (((W : ℝ) + 1) * X) ^ q ≤ (q : ℝ) ^ q * (W : ℝ) ^ (q - 1) * E ^ q := hpow
          _ ≤ (q : ℝ) ^ q * ((W : ℝ) + 1) ^ (q - 1) * E ^ q :=
            mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hWW (by positivity))
              (pow_nonneg hE0 q)
          _ = ((W : ℝ) + 1) ^ (q - 1) * ((q : ℝ) ^ q * E ^ q) := by ring
      exact le_of_mul_le_mul_left h1 hpos
    refine ⟨hnn, ?_⟩
    rw [le_div_iff₀ (mul_pos hd hn')]
    have h1 : (m₁₂ - m₂₁) * n / 2 * X ^ q ≤ ((W : ℝ) + 1) * X ^ q :=
      mul_le_mul_of_nonneg_right hWgt.le (pow_nonneg hnn q)
    nlinarith [h1, hfinal]
  -- the limit
  have key : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, ∀ u v, n - 1 ≤ u → u ≤ n → n - 1 ≤ v → v ≤ n →
      u + v < 2 * n → |p n u v - 1| < ε := by
    intro ε hε
    set C := 2 * ((q : ℝ) ^ q * Real.exp (4 * K / (m₁₂ - m₂₁)) ^ q) with hC_def
    have hεq : 0 < ε ^ q := pow_pos hε q
    obtain ⟨N1, hN1⟩ := exists_nat_gt (4 * (K : ℝ) / (m₁₂ - m₂₁))
    obtain ⟨N2, hN2⟩ := exists_nat_gt (2 / (m₁₂ - m₂₁))
    obtain ⟨N3, hN3⟩ := exists_nat_gt (C / ((m₁₂ - m₂₁) * ε ^ q))
    refine ⟨N1 + N2 + N3 + 2, fun n hn u v h1 h2 h3 h4 h5 => ?_⟩
    have hnN : ((N1 + N2 + N3 + 2 : ℕ) : ℝ) ≤ n := by exact_mod_cast hn
    push_cast at hnN
    have hN1' : (0 : ℝ) ≤ N1 := Nat.cast_nonneg _
    have hN2' : (0 : ℝ) ≤ N2 := Nat.cast_nonneg _
    have hN3' : (0 : ℝ) ≤ N3 := Nat.cast_nonneg _
    obtain ⟨hX0, hXq⟩ := main n (by linarith) (by linarith) (by omega) u v h1 h2 h3 h4 h5
    have hn' : (0 : ℝ) < n := by linarith
    have hlt : (1 - p n u v) ^ q < ε ^ q := by
      calc (1 - p n u v) ^ q ≤ C / ((m₁₂ - m₂₁) * n) := hXq
        _ < ε ^ q := by
          rw [div_lt_iff₀ (mul_pos hd hn')]
          have h := hN3
          rw [div_lt_iff₀ (mul_pos hd hεq)] at h
          have hN3n : (N3 : ℝ) ≤ n := by linarith
          nlinarith [mul_le_mul_of_nonneg_right hN3n (mul_pos hd hεq).le]
    have hX : 1 - p n u v < ε := lt_of_pow_lt_pow_left₀ q hε.le hlt
    rw [abs_sub_comm, abs_of_nonneg hX0]
    exact hX
  constructor
  · rw [Metric.tendsto_atTop]
    intro ε hε
    obtain ⟨N, hN⟩ := key ε hε
    refine ⟨N + 1, fun n hn => ?_⟩
    rw [Real.dist_eq]
    exact hN n (by omega) (n - 1) n le_rfl (by omega) (by omega) le_rfl (by omega)
  · rw [Metric.tendsto_atTop]
    intro ε hε
    obtain ⟨N, hN⟩ := key ε hε
    refine ⟨N + 1, fun n hn => ?_⟩
    rw [Real.dist_eq]
    exact hN n (by omega) n (n - 1) (by omega) le_rfl le_rfl (by omega) (by omega)

/-- Exchanging the two islands turns the first-step equations into the first-step equations of
the exchanged chain for `1 - p`. -/
lemma firstStep_swap {n : ℕ} {m₁₂ m₂₁ : ℝ} {p : ℕ → ℕ → ℝ} (hp : FirstStep n m₁₂ m₂₁ p) :
    FirstStep n m₂₁ m₁₂ (fun u v => 1 - p v u) where
  left v hv hvn := by
    show 1 - p v 0 = 0
    rw [hp.right v hv hvn]
    ring
  right u hu hun := by
    show 1 - p 0 u = 1
    rw [hp.left u hu hun]
    ring
  step u v hu hun hv hvn huv := by
    show 1 - p v u = pA n m₂₁ m₁₂ u v * (1 - p v (u - 1)) + pB n m₂₁ m₁₂ u v * (1 - p (v - 1) u)
      + (1 - pA n m₂₁ m₁₂ u v - pB n m₂₁ m₁₂ u v) * (1 - p v u)
    have h := hp.step v u hv hvn hu hun (by omega)
    have eA : pA n m₂₁ m₁₂ u v = pB n m₁₂ m₂₁ v u := by
      unfold pA pB rate₁ rate₂
      ring
    have eB : pB n m₂₁ m₁₂ u v = pA n m₁₂ m₂₁ v u := by
      unfold pA pB rate₁ rate₂
      ring
    rw [eA, eB]
    linear_combination -h

/-- **Large islands under colonization, opposite case.** If `0 < m₁₂ < m₂₁ ≤ 1`, then from a
single mutant on either island the probability of a fixation location on island 1 tends to `0`
as `n → ∞`. -/
theorem colonization_large_islands_symm {m₁₂ m₂₁ : ℝ} (h12 : 0 < m₁₂) (hlt : m₁₂ < m₂₁)
    (h21 : m₂₁ ≤ 1) (p : ℕ → ℕ → ℕ → ℝ) (hp : ∀ n, FirstStep n m₁₂ m₂₁ (p n)) :
    Tendsto (fun n => p n (n - 1) n) atTop (𝓝 0)
      ∧ Tendsto (fun n => p n n (n - 1)) atTop (𝓝 0) := by
  obtain ⟨h1, h2⟩ := colonization_large_islands (m₁₂ := m₂₁) (m₂₁ := m₁₂) h12 hlt h21
    (fun n u v => 1 - p n v u) (fun n => firstStep_swap (hp n))
  constructor
  · have h := (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1)).sub h2
    simpa using h
  · have h := (tendsto_const_nhds : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (𝓝 1)).sub h1
    simpa using h

end LargeIslands

end

end FixLoc
