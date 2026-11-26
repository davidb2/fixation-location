import FixLoc.Colonization

/-!
# Fixation location, fixation probability, and escape counts (SI Proposition 1)

The Birth-death process is described by its jump chain on mutant sets (SI, computational methods).
An active step from the mutant set `X` is an event `a → b` (offspring of `a` placed on the
out-neighbor `b`) with `a` and `b` of different types. The event has weight `f(a)/d⁺(a)`, with
fitness `f(a) = r` for a mutant and `1` for a wild type. The jump chain moves from `X` to `Y` with
probability `Q(X, Y) = (weight of active steps from X to Y) / a(X)`, with `a(X)` the total weight of
active steps from `X`.

As in the other files, probabilities and expectations enter through their first-step equations.
* `homogeneous_zero`: maximum principle. If absorption is reachable from every transient state,
  the homogeneous first-step equations have only the zero solution.
* `factorization_chain`: for every finite absorbing chain, the probability of absorption through the
  transition `z → a` equals the expected number of visits to `z` times `Q(z, a)`.
* `factorization` (SI Proposition 1): `ρ L(v | u) = n_u(V \ {v}) p_v` with
  `p_v = r θ_v / (r θ_v + 1[d⁺(v) ≥ 1])`.
* `reach_of_strongly_connected`: on a strongly connected graph, absorption is reachable from every
  transient mutant set.
-/

open Finset Relation

namespace FixLoc

noncomputable section

/-! ## Finite absorbing chains -/

section Chain

variable {S : Type*}

/-- From `x`, a path of positive transition probabilities inside the transient set `T` leads to a
state from which the chain leaves `T` with positive probability. -/
def AbsorbReach (T : Finset S) (P : S → S → ℝ) (x : S) : Prop :=
  ∃ z, ReflTransGen (fun a b => a ∈ T ∧ b ∈ T ∧ 0 < P a b) x z ∧ ∑ y ∈ T, P z y < 1

/-- At a state where `|D|` is maximal, every successor also maximizes `|D|`, and either the
maximum is `0` or the chain stays in `T`. -/
lemma max_step (T : Finset S) (P : S → S → ℝ) (hP0 : ∀ x y, 0 ≤ P x y)
    (hP1 : ∀ x ∈ T, ∑ y ∈ T, P x y ≤ 1) (D : S → ℝ)
    (hD : ∀ x ∈ T, D x = ∑ y ∈ T, P x y * D y) (M : ℝ) (hM : ∀ y ∈ T, |D y| ≤ M)
    (x : S) (hx : x ∈ T) (hxM : |D x| = M) :
    (∀ y ∈ T, 0 < P x y → |D y| = M) ∧ (M = 0 ∨ ∑ y ∈ T, P x y = 1) := by
  have hM0 : 0 ≤ M := by rw [← hxM]; exact abs_nonneg _
  have h1 : M ≤ ∑ y ∈ T, P x y * |D y| := by
    rw [← hxM, hD x hx]
    calc |∑ y ∈ T, P x y * D y| ≤ ∑ y ∈ T, |P x y * D y| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ y ∈ T, P x y * |D y| := by
        apply Finset.sum_congr rfl
        intro y _
        rw [abs_mul, abs_of_nonneg (hP0 x y)]
  have h2 : ∑ y ∈ T, P x y * |D y| ≤ ∑ y ∈ T, P x y * M :=
    Finset.sum_le_sum (fun y hy => mul_le_mul_of_nonneg_left (hM y hy) (hP0 x y))
  have h3 : ∑ y ∈ T, P x y * M = M * ∑ y ∈ T, P x y := by
    rw [← Finset.sum_mul, mul_comm]
  have hsum := hP1 x hx
  have hMs : M * ∑ y ∈ T, P x y ≤ M * 1 := mul_le_mul_of_nonneg_left hsum hM0
  constructor
  · intro y hy hpos
    have hnn : ∀ y ∈ T, 0 ≤ P x y * (M - |D y|) :=
      fun y hy => mul_nonneg (hP0 x y) (sub_nonneg.mpr (hM y hy))
    have hzero : ∑ y ∈ T, P x y * (M - |D y|) = 0 := by
      apply le_antisymm _ (Finset.sum_nonneg hnn)
      have e : ∑ y ∈ T, P x y * (M - |D y|)
          = ∑ y ∈ T, P x y * M - ∑ y ∈ T, P x y * |D y| := by
        rw [← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro y _
        ring
      rw [e, h3]
      linarith
    have := (Finset.sum_eq_zero_iff_of_nonneg hnn).mp hzero y hy
    rcases mul_eq_zero.mp this with h | h
    · exact absurd h hpos.ne'
    · linarith [hM y hy]
  · have hprod : M * (∑ y ∈ T, P x y - 1) = 0 := by
      apply le_antisymm
      · nlinarith
      · nlinarith
    rcases mul_eq_zero.mp hprod with h | h
    · exact Or.inl h
    · exact Or.inr (by linarith)

/-- Maximum principle: if absorption is reachable from every transient state, the homogeneous
first-step equations have only the zero solution on `T`. -/
theorem homogeneous_zero (T : Finset S) (P : S → S → ℝ) (hP0 : ∀ x y, 0 ≤ P x y)
    (hP1 : ∀ x ∈ T, ∑ y ∈ T, P x y ≤ 1) (hreach : ∀ x ∈ T, AbsorbReach T P x)
    (D : S → ℝ) (hD : ∀ x ∈ T, D x = ∑ y ∈ T, P x y * D y) :
    ∀ x ∈ T, D x = 0 := by
  intro x0 hx0
  obtain ⟨xm, hxm, hmax⟩ := Finset.exists_max_image T (fun x => |D x|) ⟨x0, hx0⟩
  have hM : ∀ y ∈ T, |D y| ≤ |D xm| := hmax
  have key : ∀ w, ReflTransGen (fun a b => a ∈ T ∧ b ∈ T ∧ 0 < P a b) xm w →
      w ∈ T ∧ |D w| = |D xm| := by
    intro w hw
    induction hw with
    | refl => exact ⟨hxm, rfl⟩
    | @tail b c _ hbc ih =>
      exact ⟨hbc.2.1, (max_step T P hP0 hP1 D hD _ hM b ih.1 ih.2).1 c hbc.2.1 hbc.2.2⟩
  obtain ⟨z, hpath, hleak⟩ := hreach xm hxm
  have hz := key z hpath
  have hM0 : |D xm| = 0 := by
    rcases (max_step T P hP0 hP1 D hD _ hM z hz.1 hz.2).2 with h | h
    · exact h
    · exact absurd h hleak.ne
  have := hM x0 hx0
  rw [hM0] at this
  exact abs_nonpos_iff.mp this

/-- For every finite absorbing chain: the probability `f` of absorption through the transition
`z → a` equals the expected number `g` of visits to `z` times `P z a`. -/
theorem factorization_chain (T : Finset S) (P : S → S → ℝ) (hP0 : ∀ x y, 0 ≤ P x y)
    (hP1 : ∀ x ∈ T, ∑ y ∈ T, P x y ≤ 1) (hreach : ∀ x ∈ T, AbsorbReach T P x)
    [DecidableEq S] (z a : S) (f g : S → ℝ)
    (hf : ∀ x ∈ T, f x = (if x = z then P x a else 0) + ∑ y ∈ T, P x y * f y)
    (hg : ∀ x ∈ T, g x = (if x = z then 1 else 0) + ∑ y ∈ T, P x y * g y) :
    ∀ x ∈ T, f x = g x * P z a := by
  have hD := homogeneous_zero T P hP0 hP1 hreach (fun x => f x - g x * P z a) (by
    intro x hx
    show f x - g x * P z a = ∑ y ∈ T, P x y * (f y - g y * P z a)
    have e : ∑ y ∈ T, P x y * (f y - g y * P z a)
        = ∑ y ∈ T, P x y * f y - (∑ y ∈ T, P x y * g y) * P z a := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro y _
      ring
    rw [e, hf x hx, hg x hx]
    by_cases hxz : x = z
    · rw [if_pos hxz, if_pos hxz, hxz]
      ring
    · rw [if_neg hxz, if_neg hxz]
      ring)
  intro x hx
  have h2 : f x - g x * P z a = 0 := hD x hx
  linarith

end Chain

/-! ## The jump chain of the Birth-death process -/

section BD

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- Fitness of the individual at `a` in the mutant set `X`. -/
def fit (r : ℝ) (X : Finset V) (a : V) : ℝ := if a ∈ X then r else 1

/-- Weight of the event `a → b` (offspring of `a` placed on the out-neighbor `b`), up to the
common factor `1/W(X)` with `W(X)` the total fitness. -/
def evw (out : V → Finset V) (r : ℝ) (X : Finset V) (a b : V) : ℝ :=
  if b ∈ out a then fit r X a / ((out a).card : ℝ) else 0

/-- The mutant set after the event `a → b` when `a` and `b` have different types. -/
def after (X : Finset V) (a b : V) : Finset V := if a ∈ X then insert b X else X.erase b

/-- Total weight `a(X)` of active steps from `X`. -/
def actW (out : V → Finset V) (r : ℝ) (X : Finset V) : ℝ :=
  ∑ a, ∑ b, if (a ∈ X ↔ b ∉ X) then evw out r X a b else 0

/-- Weight of the active steps from `X` to `Y`. -/
def trW (out : V → Finset V) (r : ℝ) (X Y : Finset V) : ℝ :=
  ∑ a, ∑ b, if (a ∈ X ↔ b ∉ X) ∧ after X a b = Y then evw out r X a b else 0

/-- Jump chain transition probability `Q(X, Y)`. -/
def jumpP (out : V → Finset V) (r : ℝ) (X Y : Finset V) : ℝ := trW out r X Y / actW out r X

/-- Transient mutant sets: not `∅`, not `V`, and with a possible active step. -/
def transient (out : V → Finset V) (r : ℝ) : Finset (Finset V) :=
  univ.filter (fun X => X ≠ ∅ ∧ X ≠ univ ∧ 0 < actW out r X)

/-- Temperature `θ_v = ∑_{y → v} 1 / d⁺(y)`. -/
def theta (out : V → Finset V) (v : V) : ℝ :=
  ∑ a, if v ∈ out a then 1 / ((out a).card : ℝ) else 0

lemma evw_nonneg (out : V → Finset V) {r : ℝ} (hr : 0 ≤ r) (X : Finset V) (a b : V) :
    0 ≤ evw out r X a b := by
  unfold evw fit
  split_ifs <;> positivity

lemma actW_nonneg (out : V → Finset V) {r : ℝ} (hr : 0 ≤ r) (X : Finset V) :
    0 ≤ actW out r X := by
  unfold actW
  apply Finset.sum_nonneg
  intro a _
  apply Finset.sum_nonneg
  intro b _
  split_ifs
  · exact evw_nonneg out hr X a b
  · exact le_refl 0

lemma trW_nonneg (out : V → Finset V) {r : ℝ} (hr : 0 ≤ r) (X Y : Finset V) :
    0 ≤ trW out r X Y := by
  unfold trW
  apply Finset.sum_nonneg
  intro a _
  apply Finset.sum_nonneg
  intro b _
  split_ifs
  · exact evw_nonneg out hr X a b
  · exact le_refl 0

lemma jumpP_nonneg (out : V → Finset V) {r : ℝ} (hr : 0 ≤ r) (X Y : Finset V) :
    0 ≤ jumpP out r X Y :=
  div_nonneg (trW_nonneg out hr X Y) (actW_nonneg out hr X)

/-- Every active step leads to exactly one mutant set. -/
lemma sum_trW (out : V → Finset V) (r : ℝ) (X : Finset V) :
    ∑ Y, trW out r X Y = actW out r X := by
  unfold trW actW
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro a _
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro b _
  by_cases hc : (a ∈ X ↔ b ∉ X)
  · rw [if_pos hc]
    have e : ∀ Y : Finset V, (if (a ∈ X ↔ b ∉ X) ∧ after X a b = Y then evw out r X a b else 0)
        = if after X a b = Y then evw out r X a b else 0 := by
      intro Y
      by_cases hY : after X a b = Y
      · rw [if_pos ⟨hc, hY⟩, if_pos hY]
      · rw [if_neg (fun h => hY h.2), if_neg hY]
    rw [Finset.sum_congr rfl (fun Y _ => e Y), Finset.sum_ite_eq]
    simp
  · rw [if_neg hc]
    apply Finset.sum_eq_zero
    intro Y _
    rw [if_neg (fun h => hc h.1)]

lemma sum_jumpP_le (out : V → Finset V) {r : ℝ} (hr : 0 ≤ r) (X : Finset V)
    (s : Finset (Finset V)) : ∑ Y ∈ s, jumpP out r X Y ≤ 1 := by
  have hsub : ∑ Y ∈ s, jumpP out r X Y ≤ ∑ Y, jumpP out r X Y :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ s)
      (fun Y _ _ => jumpP_nonneg out hr X Y)
  have htot : ∑ Y, jumpP out r X Y ≤ 1 := by
    unfold jumpP
    rw [← Finset.sum_div, sum_trW]
    by_cases h : actW out r X = 0
    · rw [h, div_zero]
      norm_num
    · rw [div_self h]
  linarith

lemma insert_erase_eq_univ_iff (v b : V) :
    insert b (univ.erase v) = (univ : Finset V) ↔ b = v := by
  constructor
  · intro h
    by_contra hbv
    have hv : v ∈ insert b (univ.erase v) := by
      rw [h]
      exact Finset.mem_univ v
    rcases Finset.mem_insert.mp hv with h1 | h1
    · exact hbv h1.symm
    · exact Finset.notMem_erase v univ h1
  · intro h
    rw [h, Finset.insert_erase (Finset.mem_univ v)]

lemma erase_erase_ne_univ (v b : V) : (univ.erase v).erase b ≠ (univ : Finset V) := by
  intro h
  have hv : v ∈ (univ.erase v).erase b := by
    rw [h]
    exact Finset.mem_univ v
  exact Finset.notMem_erase v univ (Finset.mem_of_mem_erase hv)

/-- From `V \ {v}`, the active steps into `V` have total weight `r θ_v`. -/
lemma trW_to_univ (out : V → Finset V) (hloop : ∀ v, v ∉ out v) (r : ℝ) (v : V) :
    trW out r (univ.erase v) univ = r * theta out v := by
  unfold trW theta
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro a _
  by_cases ha : a = v
  · rw [ha]
    have hz : ∀ b, ¬ ((v ∈ univ.erase v ↔ b ∉ univ.erase v) ∧ after (univ.erase v) v b = univ) := by
      rintro b ⟨_, h2⟩
      unfold after at h2
      rw [if_neg (Finset.notMem_erase v univ)] at h2
      exact erase_erase_ne_univ v b h2
    rw [Finset.sum_eq_zero (fun b _ => if_neg (hz b)), if_neg (hloop v), mul_zero]
  · have haX : a ∈ univ.erase v := Finset.mem_erase.mpr ⟨ha, Finset.mem_univ a⟩
    have hc : ∀ b, ((a ∈ univ.erase v ↔ b ∉ univ.erase v) ∧ after (univ.erase v) a b = univ)
        ↔ b = v := by
      intro b
      unfold after
      rw [if_pos haX, insert_erase_eq_univ_iff]
      constructor
      · exact fun h => h.2
      · intro hb
        refine ⟨?_, hb⟩
        rw [hb]
        exact ⟨fun _ => Finset.notMem_erase v univ, fun _ => haX⟩
    rw [Finset.sum_congr rfl (fun b _ => if_congr (hc b) rfl rfl), Finset.sum_ite_eq']
    simp only [Finset.mem_univ, if_true, evw, fit, haX]
    split_ifs <;> ring

/-- The total weight of active steps from `V \ {v}`. -/
lemma actW_erase (out : V → Finset V) (hloop : ∀ v, v ∉ out v) (r : ℝ) (v : V) :
    actW out r (univ.erase v) = r * theta out v + (if (out v).Nonempty then 1 else 0) := by
  have hinner : ∀ a, (∑ b, if (a ∈ univ.erase v ↔ b ∉ univ.erase v)
        then evw out r (univ.erase v) a b else 0)
      = (if a = v then (if (out v).Nonempty then 1 else 0) else 0)
        + r * (if v ∈ out a then 1 / ((out a).card : ℝ) else 0) := by
    intro a
    by_cases ha : a = v
    · rw [ha]
      have haX : v ∉ univ.erase v := Finset.notMem_erase v univ
      have hc : ∀ b, (v ∈ univ.erase v ↔ b ∉ univ.erase v) ↔ b ≠ v := by
        intro b
        simp [Finset.mem_erase]
      rw [Finset.sum_congr rfl (fun b _ => if_congr (hc b) rfl rfl), if_pos rfl, if_neg (hloop v),
        mul_zero, add_zero]
      have hev : ∀ b, (if b ≠ v then evw out r (univ.erase v) v b else 0)
          = if b ∈ out v then 1 / ((out v).card : ℝ) else 0 := by
        intro b
        unfold evw fit
        rw [if_neg haX]
        by_cases hb : b = v
        · rw [if_neg (not_not.mpr hb), hb, if_neg (hloop v)]
        · rw [if_pos hb]
      rw [Finset.sum_congr rfl (fun b _ => hev b), Finset.sum_ite_mem, Finset.univ_inter,
        Finset.sum_const, nsmul_eq_mul]
      by_cases hne : (out v).Nonempty
      · rw [if_pos hne]
        have hcard : ((out v).card : ℝ) ≠ 0 := by
          exact_mod_cast (Finset.card_pos.mpr hne).ne'
        field_simp
      · rw [if_neg hne, Finset.not_nonempty_iff_eq_empty.mp hne]
        simp
    · have haX : a ∈ univ.erase v := Finset.mem_erase.mpr ⟨ha, Finset.mem_univ a⟩
      have hc : ∀ b, (a ∈ univ.erase v ↔ b ∉ univ.erase v) ↔ b = v := by
        intro b
        simp [Finset.mem_erase, haX]
      rw [Finset.sum_congr rfl (fun b _ => if_congr (hc b) rfl rfl), Finset.sum_ite_eq', if_neg ha,
        zero_add]
      simp only [Finset.mem_univ, if_true, evw, fit, haX]
      split_ifs <;> ring
  unfold actW
  rw [Finset.sum_congr rfl (fun a _ => hinner a), Finset.sum_add_distrib, Finset.sum_ite_eq',
    if_pos (Finset.mem_univ v), ← Finset.mul_sum]
  unfold theta
  ring

/-- SI Proposition 1 (factorization of the fixation location). Let `f X` be the probability of
fixation at `v` and `g X` the expected number of active steps starting with `V \ {v}`, from the mutant
set `X`, given by their first-step equations. Then `f = g p_v` with
`p_v = r θ_v / (r θ_v + 1[d⁺(v) ≥ 1])`. For `X = {u}`, `f X = ρ L(v | u)` and `g X = n_u(V \ {v})`. -/
theorem factorization (out : V → Finset V) (hloop : ∀ v, v ∉ out v) (r : ℝ) (hr : 0 < r)
    (hreach : ∀ X ∈ transient out r, AbsorbReach (transient out r) (jumpP out r) X)
    (v : V) (f g : Finset V → ℝ)
    (hf : ∀ X ∈ transient out r, f X = (if X = univ.erase v then jumpP out r X univ else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * f Y)
    (hg : ∀ X ∈ transient out r, g X = (if X = univ.erase v then 1 else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * g Y) :
    ∀ X ∈ transient out r,
      f X = g X * (r * theta out v / (r * theta out v + if (out v).Nonempty then 1 else 0)) := by
  intro X hX
  have h := factorization_chain (transient out r) (jumpP out r)
    (fun X Y => jumpP_nonneg out hr.le X Y) (fun X _ => sum_jumpP_le out hr.le X _) hreach
    (univ.erase v) univ f g hf hg X hX
  rw [h, jumpP, trW_to_univ out hloop r v, actW_erase out hloop r v]

lemma evw_pos (out : V → Finset V) {r : ℝ} (hr : 0 < r) (X : Finset V) {c d : V}
    (hc : c ∈ X) (hd : d ∈ out c) : 0 < evw out r X c d := by
  unfold evw fit
  rw [if_pos hd, if_pos hc]
  have hcard : (0 : ℝ) < (out c).card := by exact_mod_cast Finset.card_pos.mpr ⟨d, hd⟩
  positivity

lemma actW_pos (out : V → Finset V) {r : ℝ} (hr : 0 < r) (X : Finset V) {c d : V}
    (hc : c ∈ X) (hdX : d ∉ X) (hd : d ∈ out c) : 0 < actW out r X := by
  unfold actW
  have hcond : (c ∈ X ↔ d ∉ X) := ⟨fun _ => hdX, fun _ => hc⟩
  calc (0 : ℝ) < (if (c ∈ X ↔ d ∉ X) then evw out r X c d else 0) := by
        rw [if_pos hcond]
        exact evw_pos out hr X hc hd
    _ ≤ ∑ b, (if (c ∈ X ↔ b ∉ X) then evw out r X c b else 0) :=
        Finset.single_le_sum (f := fun b => if (c ∈ X ↔ b ∉ X) then evw out r X c b else 0)
          (fun b _ => ite_nonneg (evw_nonneg out hr.le X c b) le_rfl) (Finset.mem_univ d)
    _ ≤ ∑ a, ∑ b, (if (a ∈ X ↔ b ∉ X) then evw out r X a b else 0) :=
        Finset.single_le_sum (f := fun a => ∑ b, if (a ∈ X ↔ b ∉ X) then evw out r X a b else 0)
          (fun a _ => Finset.sum_nonneg (fun b _ => ite_nonneg (evw_nonneg out hr.le X a b) le_rfl))
          (Finset.mem_univ c)

lemma trW_insert_pos (out : V → Finset V) {r : ℝ} (hr : 0 < r) (X : Finset V) {c d : V}
    (hc : c ∈ X) (hdX : d ∉ X) (hd : d ∈ out c) : 0 < trW out r X (insert d X) := by
  unfold trW
  have hcond : (c ∈ X ↔ d ∉ X) ∧ after X c d = insert d X := by
    refine ⟨⟨fun _ => hdX, fun _ => hc⟩, ?_⟩
    unfold after
    rw [if_pos hc]
  calc (0 : ℝ) < (if (c ∈ X ↔ d ∉ X) ∧ after X c d = insert d X then evw out r X c d else 0) := by
        rw [if_pos hcond]
        exact evw_pos out hr X hc hd
    _ ≤ ∑ b, (if (c ∈ X ↔ b ∉ X) ∧ after X c b = insert d X then evw out r X c b else 0) :=
        Finset.single_le_sum
          (f := fun b => if (c ∈ X ↔ b ∉ X) ∧ after X c b = insert d X then evw out r X c b else 0)
          (fun b _ => ite_nonneg (evw_nonneg out hr.le X c b) le_rfl) (Finset.mem_univ d)
    _ ≤ ∑ a, ∑ b, (if (a ∈ X ↔ b ∉ X) ∧ after X a b = insert d X then evw out r X a b else 0) :=
        Finset.single_le_sum
          (f := fun a => ∑ b, if (a ∈ X ↔ b ∉ X) ∧ after X a b = insert d X then evw out r X a b else 0)
          (fun a _ => Finset.sum_nonneg (fun b _ => ite_nonneg (evw_nonneg out hr.le X a b) le_rfl))
          (Finset.mem_univ c)

/-- On a strongly connected graph every nonempty proper mutant set has an edge leaving it. -/
lemma exists_exit_edge (out : V → Finset V) (hsc : ∀ a b : V, ReflTransGen (fun x y => y ∈ out x) a b)
    (X : Finset V) (hne : X ≠ ∅) (hnu : X ≠ univ) : ∃ c d, c ∈ X ∧ d ∉ X ∧ d ∈ out c := by
  obtain ⟨x, hx⟩ := Finset.nonempty_iff_ne_empty.mpr hne
  obtain ⟨y, hy⟩ : ∃ y, y ∉ X := by
    by_contra h
    push_neg at h
    exact hnu (Finset.eq_univ_iff_forall.mpr h)
  obtain ⟨c, d, hcd, hc, hd⟩ := exists_exit (S := fun w => w ∈ X) (hsc x y) hx hy
  exact ⟨c, d, hc, hd, hcd⟩

/-- On a strongly connected graph, absorption is reachable from every transient mutant set: the
mutants colonize the rest of the graph along edges leaving the mutant set. -/
theorem reach_of_strongly_connected (out : V → Finset V) (r : ℝ) (hr : 0 < r)
    (hsc : ∀ a b : V, ReflTransGen (fun x y => y ∈ out x) a b) :
    ∀ X ∈ transient out r, AbsorbReach (transient out r) (jumpP out r) X := by
  have memT : ∀ X, X ∈ transient out r ↔ X ≠ ∅ ∧ X ≠ univ ∧ 0 < actW out r X := by
    intro X
    simp [transient]
  have key : ∀ n : ℕ, ∀ X ∈ transient out r, (univ \ X).card = n + 1 →
      AbsorbReach (transient out r) (jumpP out r) X := by
    intro n
    induction n with
    | zero =>
      intro X hX hcard
      obtain ⟨hne, hnu, hpos⟩ := (memT X).mp hX
      obtain ⟨c, d, hc, hdX, hd⟩ := exists_exit_edge out hsc X hne hnu
      have hYu : insert d X = univ := by
        obtain ⟨w, hw⟩ := Finset.card_eq_one.mp hcard
        apply Finset.eq_univ_iff_forall.mpr
        intro z
        by_cases hz : z ∈ X
        · exact Finset.mem_insert_of_mem hz
        · have hzw : z ∈ univ \ X := Finset.mem_sdiff.mpr ⟨Finset.mem_univ z, hz⟩
          have hdw : d ∈ univ \ X := Finset.mem_sdiff.mpr ⟨Finset.mem_univ d, hdX⟩
          rw [hw, Finset.mem_singleton] at hzw hdw
          rw [hzw, ← hdw]
          exact Finset.mem_insert_self d X
      have hP : 0 < jumpP out r X univ := by
        rw [← hYu]
        exact div_pos (trW_insert_pos out hr X hc hdX hd) hpos
      refine ⟨X, ReflTransGen.refl, ?_⟩
      have huT : (univ : Finset V) ∉ transient out r := fun h => ((memT _).mp h).2.1 rfl
      have hins : ∑ y ∈ insert univ (transient out r), jumpP out r X y
          = jumpP out r X univ + ∑ y ∈ transient out r, jumpP out r X y :=
        Finset.sum_insert huT
      have hle := sum_jumpP_le out hr.le X (insert univ (transient out r))
      linarith
    | succ n ih =>
      intro X hX hcard
      obtain ⟨hne, hnu, hpos⟩ := (memT X).mp hX
      obtain ⟨c, d, hc, hdX, hd⟩ := exists_exit_edge out hsc X hne hnu
      set Y := insert d X with hYdef
      have hdmem : d ∈ univ \ X := Finset.mem_sdiff.mpr ⟨Finset.mem_univ d, hdX⟩
      have hcardY : (univ \ Y).card = n + 1 := by
        have hset : univ \ Y = (univ \ X).erase d := by
          ext w
          simp only [hYdef, Finset.mem_sdiff, Finset.mem_univ, Finset.mem_insert, Finset.mem_erase,
            true_and]
          tauto
        rw [hset, Finset.card_erase_of_mem hdmem, hcard]
        simp
      have hYne : Y ≠ ∅ := Finset.nonempty_iff_ne_empty.mp ⟨d, Finset.mem_insert_self d X⟩
      have hYnu : Y ≠ univ := by
        intro h
        rw [h, Finset.sdiff_self, Finset.card_empty] at hcardY
        omega
      obtain ⟨c', d', hc', hd'Y, hd'⟩ := exists_exit_edge out hsc Y hYne hYnu
      have hYT : Y ∈ transient out r :=
        (memT Y).mpr ⟨hYne, hYnu, actW_pos out hr Y hc' hd'Y hd'⟩
      have hP : 0 < jumpP out r X Y := div_pos (trW_insert_pos out hr X hc hdX hd) hpos
      obtain ⟨z, hpath, hleak⟩ := ih Y hYT hcardY
      exact ⟨z, ReflTransGen.head ⟨hX, hYT, hP⟩ hpath, hleak⟩
  intro X hX
  obtain ⟨hne, hnu, _⟩ := (memT X).mp hX
  have hpos : 0 < (univ \ X).card := by
    obtain ⟨y, hy⟩ : ∃ y, y ∉ X := by
      by_contra h
      push_neg at h
      exact hnu (Finset.eq_univ_iff_forall.mpr h)
    exact Finset.card_pos.mpr ⟨y, Finset.mem_sdiff.mpr ⟨Finset.mem_univ y, hy⟩⟩
  obtain ⟨n, hn⟩ : ∃ n, (univ \ X).card = n + 1 := ⟨(univ \ X).card - 1, by omega⟩
  exact key n X hX hn

/-- SI Proposition 1 on every strongly connected graph without self-loops. -/
theorem factorization_strongly_connected (out : V → Finset V) (hloop : ∀ v, v ∉ out v) (r : ℝ)
    (hr : 0 < r) (hsc : ∀ a b : V, ReflTransGen (fun x y => y ∈ out x) a b)
    (v : V) (f g : Finset V → ℝ)
    (hf : ∀ X ∈ transient out r, f X = (if X = univ.erase v then jumpP out r X univ else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * f Y)
    (hg : ∀ X ∈ transient out r, g X = (if X = univ.erase v then 1 else 0)
        + ∑ Y ∈ transient out r, jumpP out r X Y * g Y) :
    ∀ X ∈ transient out r,
      f X = g X * (r * theta out v / (r * theta out v + if (out v).Nonempty then 1 else 0)) :=
  factorization out hloop r hr (reach_of_strongly_connected out r hr hsc) v f g hf hg

end BD

end

end FixLoc
