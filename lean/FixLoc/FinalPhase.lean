import FixLoc.Factorization

/-!
# Fixation location and the final phase of fixation (SI Proposition 2)

One time step of the Birth-death process is an active step with probability `a(X)/W(X)`, with
`W(X)` the total fitness, and leaves the mutant set unchanged otherwise (`stepP`). Let `τ_v X` be
the expected number of time steps with mutant set `V \ {v}` on the event of fixation, from the mutant
set `X`. The first-step equation of `τ_v` counts one time step in `V \ {v}` with the fixation
probability `h(V \ {v})` of the remaining process.

* `scale_chain`: solutions of first-step equations with a source at one state are multiples of the
  escape counts of that state.
* `tau_eq`: `τ_v = n(V \ {v}) h(V \ {v}) W(V \ {v}) / a(V \ {v})`.
* `final_phase` (SI Proposition 2): `T^end(u) = (r(N-1)+1)/r ∑_v L(v | u) h(V \ {v}) / θ_v`.
-/

open Finset Relation

namespace FixLoc

noncomputable section

section Chain

variable {S : Type*} [DecidableEq S]

/-- Solutions of first-step equations with source `c` at the state `z` are `c` times the expected
number of visits to `z`. -/
theorem scale_chain (T : Finset S) (P : S → S → ℝ) (hP0 : ∀ x y, 0 ≤ P x y)
    (hP1 : ∀ x ∈ T, ∑ y ∈ T, P x y ≤ 1) (hreach : ∀ x ∈ T, AbsorbReach T P x)
    (z : S) (c : ℝ) (u g : S → ℝ)
    (hu : ∀ x ∈ T, u x = (if x = z then c else 0) + ∑ y ∈ T, P x y * u y)
    (hg : ∀ x ∈ T, g x = (if x = z then 1 else 0) + ∑ y ∈ T, P x y * g y) :
    ∀ x ∈ T, u x = g x * c := by
  have hD := homogeneous_zero T P hP0 hP1 hreach (fun x => u x - g x * c) (by
    intro x hx
    show u x - g x * c = ∑ y ∈ T, P x y * (u y - g y * c)
    have e : ∑ y ∈ T, P x y * (u y - g y * c)
        = ∑ y ∈ T, P x y * u y - (∑ y ∈ T, P x y * g y) * c := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro y _
      ring
    rw [e, hu x hx, hg x hx]
    by_cases hxz : x = z
    · rw [if_pos hxz, if_pos hxz]
      ring
    · rw [if_neg hxz, if_neg hxz]
      ring)
  intro x hx
  have h2 : u x - g x * c = 0 := hD x hx
  linarith

end Chain

section BD

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Total fitness `W(X)`. -/
def totalFit (r : ℝ) (X : Finset V) : ℝ := ∑ a, fit r X a

/-- Transition probability of one time step of the Birth-death process. -/
def stepP (out : V → Finset V) (r : ℝ) (X Y : Finset V) : ℝ :=
  trW out r X Y / totalFit r X + (if X = Y then 1 - actW out r X / totalFit r X else 0)

/-- An active step changes the mutant set. -/
lemma trW_self (out : V → Finset V) (r : ℝ) (X : Finset V) : trW out r X X = 0 := by
  unfold trW
  apply Finset.sum_eq_zero
  intro a _
  apply Finset.sum_eq_zero
  intro b _
  rw [if_neg]
  rintro ⟨hc, hY⟩
  unfold after at hY
  by_cases ha : a ∈ X
  · rw [if_pos ha] at hY
    have hb : b ∉ X := hc.mp ha
    exact hb (hY ▸ Finset.mem_insert_self b X)
  · rw [if_neg ha] at hY
    have hb : b ∈ X := by
      by_contra hb
      exact ha (hc.mpr hb)
    have : b ∉ X.erase b := Finset.notMem_erase b X
    rw [hY] at this
    exact this hb

lemma totalFit_pos (r : ℝ) (hr : 0 < r) (X : Finset V) [Nonempty V] : 0 < totalFit r X := by
  unfold totalFit
  apply Finset.sum_pos
  · intro a _
    unfold fit
    split_ifs
    · exact hr
    · exact one_pos
  · exact Finset.univ_nonempty

lemma totalFit_erase (r : ℝ) (v : V) :
    totalFit r (univ.erase v) = r * ((Fintype.card V : ℝ) - 1) + 1 := by
  unfold totalFit fit
  have e : ∀ a, (if a ∈ univ.erase v then r else 1) = r + (if a = v then 1 - r else 0) := by
    intro a
    by_cases ha : a = v
    · rw [if_pos ha, ha, if_neg (Finset.notMem_erase v univ)]
      ring
    · rw [if_neg ha, if_pos (Finset.mem_erase.mpr ⟨ha, Finset.mem_univ a⟩)]
      ring
  rw [Finset.sum_congr rfl (fun a _ => e a), Finset.sum_add_distrib, Finset.sum_const,
    Finset.sum_ite_eq', if_pos (Finset.mem_univ v), Finset.card_univ, nsmul_eq_mul]
  ring

/-- The first-step equation of `τ` in one time step becomes a first-step equation of the jump
chain with source `s W(X)/a(X)`. -/
lemma tau_jump (out : V → Finset V) (r : ℝ) (hr : 0 < r) [Nonempty V] (z : Finset V) (s : ℝ)
    (τ : Finset V → ℝ)
    (hτ : ∀ X ∈ transient out r, τ X = (if X = z then s else 0)
        + ∑ Y ∈ transient out r, stepP out r X Y * τ Y) :
    ∀ X ∈ transient out r, τ X = (if X = z then s * totalFit r z / actW out r z else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * τ Y := by
  intro X hX
  have hXmem := Finset.mem_filter.mp hX
  have ha : 0 < actW out r X := hXmem.2.2.2
  have hW : 0 < totalFit r X := totalFit_pos r hr X
  have hsplit : ∑ Y ∈ transient out r, stepP out r X Y * τ Y
      = ∑ Y ∈ transient out r, trW out r X Y / totalFit r X * τ Y
        + (1 - actW out r X / totalFit r X) * τ X := by
    unfold stepP
    simp only [add_mul, Finset.sum_add_distrib]
    congr 1
    rw [Finset.sum_congr rfl (fun Y _ => by rw [ite_mul, zero_mul]), Finset.sum_ite_eq,
      if_pos hX]
  have h1 := hτ X hX
  rw [hsplit] at h1
  have hjump : ∑ Y ∈ transient out r, jumpP out r X Y * τ Y
      = (∑ Y ∈ transient out r, trW out r X Y / totalFit r X * τ Y) * (totalFit r X / actW out r X) := by
    rw [Finset.sum_mul]
    apply Finset.sum_congr rfl
    intro Y _
    unfold jumpP
    field_simp
  rw [hjump]
  by_cases hXz : X = z
  · rw [if_pos hXz] at h1 ⊢
    rw [← hXz]
    field_simp
    field_simp at h1
    linarith
  · rw [if_neg hXz] at h1 ⊢
    field_simp
    field_simp at h1
    linarith

/-- SI Proposition 2 (final phase of fixation). Let `f v X` be the probability of fixation at `v`,
`g v X` the expected number of active steps starting with `V \ {v}`, and `τ v X` the expected
number of time steps with mutant set `V \ {v}` on the event of fixation, from the mutant set `X`,
given by their first-step equations. Let `h` be the fixation probability. If every vertex has an
out-neighbor and an in-neighbor, the expected number of time steps with one wild type left,
conditional on fixation from `u`, is `(r(N-1)+1)/r ∑_v L(v | u) h(V \ {v}) / θ_v` with
`L(v | u) = f v {u} / h {u}`. -/
theorem final_phase (out : V → Finset V) (hloop : ∀ v, v ∉ out v) (r : ℝ) (hr : 0 < r)
    [Nonempty V]
    (hreach : ∀ X ∈ transient out r, AbsorbReach (transient out r) (jumpP out r) X)
    (hout : ∀ v, (out v).Nonempty) (hin : ∀ v, 0 < theta out v)
    (h : Finset V → ℝ) (f g τ : V → Finset V → ℝ) (X : Finset V) (hX : X ∈ transient out r)
    (hf : ∀ v, ∀ X ∈ transient out r, f v X = (if X = univ.erase v then jumpP out r X univ else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * f v Y)
    (hg : ∀ v, ∀ X ∈ transient out r, g v X = (if X = univ.erase v then 1 else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * g v Y)
    (hτ : ∀ v, ∀ X ∈ transient out r, τ v X = (if X = univ.erase v then h (univ.erase v) else 0)
        + ∑ Y ∈ transient out r, stepP out r X Y * τ v Y) :
    ∑ v, τ v X / h X
      = (r * ((Fintype.card V : ℝ) - 1) + 1) / r * ∑ v, f v X / h X * h (univ.erase v) / theta out v := by
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro v _
  have hP0 : ∀ X Y, 0 ≤ jumpP out r X Y := fun X Y => jumpP_nonneg out hr.le X Y
  have hP1 : ∀ X ∈ transient out r, ∑ Y ∈ transient out r, jumpP out r X Y ≤ 1 :=
    fun X _ => sum_jumpP_le out hr.le X _
  have hfv := factorization out hloop r hr hreach v (f v) (g v) (hf v) (hg v) X hX
  have hτv := scale_chain (transient out r) (jumpP out r) hP0 hP1 hreach (univ.erase v)
    (h (univ.erase v) * totalFit r (univ.erase v) / actW out r (univ.erase v)) (τ v) (g v)
    (tau_jump out r hr (univ.erase v) (h (univ.erase v)) (τ v) (hτ v)) (hg v) X hX
  rw [hτv, actW_erase out hloop r v, totalFit_erase r v, if_pos (hout v)] at *
  rw [hτv, hfv]
  have hθ := hin v
  have hden : r * theta out v + 1 ≠ 0 := by positivity
  have hθ0 : theta out v ≠ 0 := hθ.ne'
  have hr0 : r ≠ 0 := hr.ne'
  field_simp

end BD

end

end FixLoc
