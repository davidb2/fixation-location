import FixLoc.Colonization

/-!
# Possible fixation locations at finite selection (SI Theorem S1, Corollary S1, Theorem S2,
# Proposition S3, Lemma S1)

At finite fitness `r > 0` every Birth-death event that changes the mutant set has positive
probability. An event is a placement of an offspring of the individual at `a` on an out-neighbor `b`
of a different type. A mutant parent adds `b` to the mutant set (gain). A wild-type parent removes
`b` from the mutant set (loss). These events are the edges of the configuration graph of the SI
(`BDStep`). Fixation at `t` has positive probability exactly when some sequence of events leads from
the single mutant at `s` to a mutant set without `t` and the next event makes every vertex a mutant
(`FixableAt`). The probability space of the process is not constructed.

This file proves
* `fixable_at_iff` (SI Theorem S1) and `bdstep_erase_univ_iff` (the second condition of Theorem S1);
* `not_fixable_of_unreachable` and `not_fixable_of_no_in` (SI Corollary S1);
* `refillable_fixable` (SI Theorem S2);
* `gateway` and `gateway_initial` (SI Proposition S3);
* `single_in` (SI Lemma S1, for `t ≠ u`).
-/

open Finset Relation

namespace FixLoc

variable {V : Type*} [DecidableEq V]

/-- One Birth-death event that changes the mutant set (an edge of the configuration graph). -/
inductive BDStep (adj : V → V → Prop) : Finset V → Finset V → Prop
  | gain {X : Finset V} {a b : V} : a ∈ X → b ∉ X → adj a b → BDStep adj X (insert b X)
  | loss {X : Finset V} {a b : V} : a ∉ X → b ∈ X → adj a b → BDStep adj X (X.erase b)

lemma BDStep.cases' {adj : V → V → Prop} {X Y : Finset V} (h : BDStep adj X Y) :
    (∃ a b, a ∈ X ∧ b ∉ X ∧ adj a b ∧ Y = insert b X) ∨
    (∃ a b, a ∉ X ∧ b ∈ X ∧ adj a b ∧ Y = X.erase b) := by
  cases h with
  | gain ha hb hab => exact Or.inl ⟨_, _, ha, hb, hab, rfl⟩
  | loss ha hb hab => exact Or.inr ⟨_, _, ha, hb, hab, rfl⟩

/-- Every mutant descends from the initial mutant along a directed path. -/
lemma reach_sub {adj : V → V → Prop} {s : V} {X : Finset V}
    (h : ReflTransGen (BDStep adj) {s} X) : ∀ w ∈ X, ReflTransGen adj s w := by
  induction h with
  | refl =>
    intro w hw
    rw [Finset.mem_singleton] at hw
    subst hw
    exact ReflTransGen.refl
  | @tail Y Z _ hYZ ih =>
    intro w hw
    rcases BDStep.cases' hYZ with ⟨a, b, ha, _, hab, hZ⟩ | ⟨a, b, _, _, _, hZ⟩
    · rw [hZ] at hw
      rcases Finset.mem_insert.mp hw with hwb | hwY
      · rw [hwb]
        exact (ih a ha).tail hab
      · exact ih w hwY
    · rw [hZ] at hw
      exact ih w (Finset.mem_of_mem_erase hw)

/-- Colonization steps are Birth-death events. -/
lemma ColReach.bd {adj e : V → V → Prop} (he : ∀ a b, e a b → adj a b) {s : V} {X : Finset V}
    (h : ColReach e s X) : ReflTransGen (BDStep adj) {s} X := by
  induction h with
  | start => exact ReflTransGen.refl
  | @grow X x y _ hx hxy hy ih => exact ih.tail (BDStep.gain hx hy (he x y hxy))

/-- A path whose vertices all satisfy `P` uses only edges into vertices satisfying `P`. -/
lemma rtg_restrict {r : V → V → Prop} {P : V → Prop} {s v : V}
    (hP : ∀ w, ReflTransGen r s w → P w) (h : ReflTransGen r s v) :
    ReflTransGen (fun a b => r a b ∧ P b) s v := by
  induction h with
  | refl => exact ReflTransGen.refl
  | @tail b c hb hbc ih => exact ih.tail ⟨hbc, hP c (hb.tail hbc)⟩

/-- Colonization along the edges `e` from `s` reaches every set `R` whose vertices are reachable from
`s` by paths inside `R`. -/
lemma colReach_of_paths {e : V → V → Prop} {s : V} {R : Finset V} (hs : s ∈ R)
    (hR : ∀ v ∈ R, ReflTransGen (fun a b => e a b ∧ b ∈ R) s v) : ColReach e s R := by
  have key : ∀ n : ℕ, ∀ X : Finset V, ColReach e s X → X ⊆ R → (R \ X).card = n →
      ColReach e s R := by
    intro n
    induction n with
    | zero =>
      intro X hX hXR hcard
      have hRX : R ⊆ X := by
        intro v hv
        by_contra hvX
        have hmem : v ∈ R \ X := Finset.mem_sdiff.mpr ⟨hv, hvX⟩
        rw [Finset.card_eq_zero] at hcard
        rw [hcard] at hmem
        exact absurd hmem (Finset.notMem_empty v)
      rw [Finset.Subset.antisymm hXR hRX] at hX
      exact hX
    | succ n ih =>
      intro X hX hXR hcard
      obtain ⟨v, hv⟩ : (R \ X).Nonempty := by
        rw [← Finset.card_pos, hcard]
        exact Nat.succ_pos n
      have hv' := Finset.mem_sdiff.mp hv
      obtain ⟨c, d, hcd, hc, hd⟩ :=
        exists_exit (S := fun w => w ∈ X) (hR v hv'.1) (ColReach.mem_start hX) hv'.2
      apply ih (insert d X) (ColReach.grow hX hc hcd.1 hd)
      · exact Finset.insert_subset_iff.mpr ⟨hcd.2, hXR⟩
      · have hdmem : d ∈ R \ X := Finset.mem_sdiff.mpr ⟨hcd.2, hd⟩
        have hset : R \ insert d X = (R \ X).erase d := by
          ext w
          simp only [Finset.mem_sdiff, Finset.mem_insert, Finset.mem_erase]
          tauto
        rw [hset, Finset.card_erase_of_mem hdmem, hcard]
        simp
  exact key _ {s} ColReach.start (Finset.singleton_subset_iff.mpr hs) rfl

/-- A finite sequence of events as a trajectory indexed by time. -/
lemma traj_snoc {adj : V → V → Prop} {X : ℕ → Finset V} {m : ℕ} {C : Finset V}
    (hs : ∀ i < m, BDStep adj (X i) (X (i + 1))) (hC : BDStep adj (X m) C) :
    ∀ i < m + 1, BDStep adj ((fun i => if i ≤ m then X i else C) i)
      ((fun i => if i ≤ m then X i else C) (i + 1)) := by
  intro i hi
  show BDStep adj (if i ≤ m then X i else C) (if i + 1 ≤ m then X (i + 1) else C)
  by_cases him : i < m
  · rw [if_pos (le_of_lt him), if_pos (show i + 1 ≤ m from him)]
    exact hs i him
  · have hieq : i = m := by omega
    rw [if_pos (le_of_eq hieq), if_neg (show ¬ (i + 1 ≤ m) by omega), hieq]
    exact hC

lemma exists_traj {adj : V → V → Prop} {A B : Finset V} (h : ReflTransGen (BDStep adj) A B) :
    ∃ (X : ℕ → Finset V) (m : ℕ), X 0 = A ∧ X m = B ∧ ∀ i < m, BDStep adj (X i) (X (i + 1)) := by
  induction h with
  | refl => exact ⟨fun _ => A, 0, rfl, rfl, fun i hi => absurd hi (Nat.not_lt_zero i)⟩
  | @tail B C _ hBC ih =>
    obtain ⟨X, m, h0, hm, hs⟩ := ih
    refine ⟨fun i => if i ≤ m then X i else C, m + 1, ?_, ?_, traj_snoc hs (hm ▸ hBC)⟩
    · simp [h0]
    · simp

/-- Ancestry: while `t` has never held a mutant, every mutant descends from the initial mutant at
`u` along a directed path avoiding `t`. -/
lemma ancestry {adj : V → V → Prop} {u t : V} {X : ℕ → Finset V} {m : ℕ}
    (h0 : X 0 = {u}) (hs : ∀ i < m, BDStep adj (X i) (X (i + 1))) :
    ∀ n, n ≤ m → (∀ i ≤ n, t ∉ X i) → ∀ v ∈ X n, ReflTransGen (avoid adj t) u v := by
  intro n
  induction n with
  | zero =>
    intro _ _ v hv
    rw [h0, Finset.mem_singleton] at hv
    subst hv
    exact ReflTransGen.refl
  | succ n ih =>
    intro hn hnot v hv
    have ih' := ih (by omega) (fun i hi => hnot i (by omega))
    have htn : t ∉ X n := hnot n (by omega)
    have htn1 : t ∉ X (n + 1) := hnot (n + 1) le_rfl
    rcases BDStep.cases' (hs n (by omega)) with ⟨a, b, ha, _, hab, hY⟩ | ⟨a, b, _, _, _, hY⟩
    · rw [hY] at hv htn1
      rcases Finset.mem_insert.mp hv with hvb | hvX
      · have hat : a ≠ t := fun h => htn (h ▸ ha)
        have hbt : b ≠ t := fun h => htn1 (by rw [← h]; exact Finset.mem_insert_self b (X n))
        rw [hvb]
        exact (ih' a ha).tail ⟨hab, hat, hbt⟩
      · exact ih' v hvX
    · rw [hY] at hv
      exact ih' v (Finset.mem_of_mem_erase hv)

variable [Fintype V]

/-- Fixation at `t` has positive probability. -/
def FixableAt (adj : V → V → Prop) (s t : V) : Prop :=
  ∃ X, ReflTransGen (BDStep adj) {s} X ∧ BDStep adj X univ ∧ t ∉ X

/-- The only events into the all-mutant set start from `V \ {t}` and fill `t` from an in-neighbor. -/
lemma bdstep_univ {adj : V → V → Prop} {X Y : Finset V} (h : BDStep adj X Y) (hY : Y = univ) :
    ∃ t, X = univ.erase t ∧ ∃ x, x ≠ t ∧ adj x t := by
  cases h with
  | @gain a b ha hb hab =>
    refine ⟨b, ?_, a, fun h => hb (h ▸ ha), hab⟩
    ext v
    by_cases hv : v = b
    · rw [hv]
      simp [hb]
    · have hvX : v ∈ X := by
        have : v ∈ insert b X := by rw [hY]; exact Finset.mem_univ v
        rcases Finset.mem_insert.mp this with h | h
        · exact absurd h hv
        · exact h
      simp [hv, hvX]
  | @loss a b ha hb hab =>
    exfalso
    have : b ∈ X.erase b := by rw [hY]; exact Finset.mem_univ b
    exact Finset.notMem_erase b X this

/-- The second condition of SI Theorem S1: the configuration graph has the edge `V \ {t} → V`
exactly when `t` has an in-neighbor other than `t`. -/
theorem bdstep_erase_univ_iff (adj : V → V → Prop) (t : V) :
    BDStep adj (univ.erase t) univ ↔ ∃ x, x ≠ t ∧ adj x t := by
  constructor
  · intro h
    obtain ⟨t', hXt', x, hxt', hadj⟩ := bdstep_univ h rfl
    have htt' : t = t' := by
      by_contra hne
      have hmem : t' ∈ univ.erase t := Finset.mem_erase.mpr ⟨fun h => hne h.symm, Finset.mem_univ t'⟩
      rw [hXt'] at hmem
      exact Finset.notMem_erase t' univ hmem
    rw [htt']
    exact ⟨x, hxt', hadj⟩
  · rintro ⟨x, hxt, hadj⟩
    have h := BDStep.gain (adj := adj) (Finset.mem_erase.mpr ⟨hxt, Finset.mem_univ x⟩) (Finset.notMem_erase t univ) hadj
    rwa [Finset.insert_erase (Finset.mem_univ t)] at h

/-- SI Theorem S1 (exact characterization). -/
theorem fixable_at_iff (adj : V → V → Prop) (s t : V) :
    FixableAt adj s t ↔ ReflTransGen (BDStep adj) {s} (univ.erase t) ∧ ∃ x, x ≠ t ∧ adj x t := by
  constructor
  · rintro ⟨X, hX, hstep, htX⟩
    obtain ⟨t', hXt', x, hxt', hadj⟩ := bdstep_univ hstep rfl
    have htt' : t' = t := by
      by_contra hne
      exact htX (by rw [hXt']; exact Finset.mem_erase.mpr ⟨fun h => hne h.symm, Finset.mem_univ t⟩)
    rw [hXt', htt'] at hX
    rw [htt'] at hxt' hadj
    exact ⟨hX, x, hxt', hadj⟩
  · rintro ⟨hreach, hin⟩
    exact ⟨univ.erase t, hreach, (bdstep_erase_univ_iff adj t).mpr hin, Finset.notMem_erase t univ⟩

/-- SI Corollary S1, first check: a vertex unreachable from `s` rules out fixation. -/
theorem not_fixable_of_unreachable (adj : V → V → Prop) (s t v : V)
    (hv : ¬ ReflTransGen adj s v) : ¬ FixableAt adj s t := by
  rintro ⟨X, hX, hstep, _⟩
  obtain ⟨t', hXt', x, hxt', hadj⟩ := bdstep_univ hstep rfl
  have hsub := reach_sub hX
  by_cases hvt : v = t'
  · apply hv
    rw [hvt]
    exact (hsub x (by rw [hXt']; exact Finset.mem_erase.mpr ⟨hxt', Finset.mem_univ x⟩)).tail hadj
  · exact hv (hsub v (by rw [hXt']; exact Finset.mem_erase.mpr ⟨hvt, Finset.mem_univ v⟩))

/-- SI Corollary S1, second check: without an in-neighbor other than `t`, fixation at `t` is
impossible. -/
theorem not_fixable_of_no_in (adj : V → V → Prop) (s t : V) (ht : ∀ x, adj x t → x = t) :
    ¬ FixableAt adj s t := by
  rw [fixable_at_iff]
  rintro ⟨_, x, hxt, hadj⟩
  exact hxt (ht x hadj)

lemma insert_erase_erase {x y : V} (hxy : x ≠ y) :
    insert x ((univ.erase x).erase y) = univ.erase y := by
  ext v
  simp only [Finset.mem_insert, Finset.mem_erase, Finset.mem_univ, and_true]
  constructor
  · rintro (h | ⟨hvy, _⟩)
    · rw [h]; exact hxy
    · exact hvy
  · intro hvy
    by_cases hvx : v = x
    · exact Or.inl hvx
    · exact Or.inr ⟨hvy, hvx⟩

/-- A refillable path `u 0 → u 1 → ⋯ → u k = t` from `s` with witness `w` (SI definition of a
refillable path). The path is simple. The length `k = 0` is allowed. -/
structure Refillable (adj : V → V → Prop) (s t : V) (k : ℕ) (u : ℕ → V) (w : V) : Prop where
  path : ∀ i < k, adj (u i) (u (i + 1))
  last : u k = t
  inj : ∀ i j, i ≤ k → j ≤ k → u i = u j → i = j
  start : u 0 ≠ s
  witness : adj w (u 0) ∧ ∀ i ≤ k, w ≠ u i
  reach : ∀ v, v ≠ t → v ≠ u 0 → ReflTransGen (avoid adj (u 0)) s v

/-- Moving the single wild type from `u 0` along the path, refilling each vertex behind it. -/
lemma move_wild {adj : V → V → Prop} {k : ℕ} {u : ℕ → V} {w : V}
    (hpath : ∀ i < k, adj (u i) (u (i + 1)))
    (hinj : ∀ i j, i ≤ k → j ≤ k → u i = u j → i = j)
    (hw : adj w (u 0) ∧ ∀ i ≤ k, w ≠ u i) :
    ∀ i ≤ k, ReflTransGen (BDStep adj) (univ.erase (u 0)) (univ.erase (u i)) := by
  intro i
  induction i with
  | zero => intro _; exact ReflTransGen.refl
  | succ i ih =>
    intro hi
    have hik : i < k := by omega
    have hne : u i ≠ u (i + 1) := fun h => by
      have := hinj i (i + 1) (by omega) hi h
      omega
    have h1 : BDStep adj (univ.erase (u i)) ((univ.erase (u i)).erase (u (i + 1))) :=
      BDStep.loss (Finset.notMem_erase (u i) univ) (Finset.mem_erase.mpr ⟨hne.symm, Finset.mem_univ _⟩) (hpath i hik)
    obtain ⟨p, hp_adj, hp_i, hp_i1⟩ : ∃ p, adj p (u i) ∧ p ≠ u i ∧ p ≠ u (i + 1) := by
      cases i with
      | zero => exact ⟨w, hw.1, hw.2 0 (by omega), hw.2 1 (by omega)⟩
      | succ i' =>
        refine ⟨u i', hpath i' (by omega), ?_, ?_⟩
        · intro h
          have := hinj i' (i' + 1) (by omega) (by omega) h
          omega
        · intro h
          have := hinj i' (i' + 1 + 1) (by omega) (by omega) h
          omega
    have h2 : BDStep adj ((univ.erase (u i)).erase (u (i + 1)))
        (insert (u i) ((univ.erase (u i)).erase (u (i + 1)))) :=
      BDStep.gain (Finset.mem_erase.mpr ⟨hp_i1, Finset.mem_erase.mpr ⟨hp_i, Finset.mem_univ p⟩⟩)
        (fun h => (Finset.mem_erase.mp (Finset.mem_erase.mp h).2).1 rfl) hp_adj
    rw [insert_erase_erase hne] at h2
    exact ((ih (by omega)).tail h1).tail h2

/-- SI Theorem S2 (refillability implies fixability). The Lean statement needs no hypothesis
`s ≠ t`. -/
theorem refillable_fixable (adj : V → V → Prop) (s t : V) (k : ℕ) (u : ℕ → V) (w : V)
    (hP : Refillable adj s t k u w) : FixableAt adj s t := by
  classical
  obtain ⟨hpath, hlast, hinj, hstart, hw, hreach⟩ := hP
  have hu0 : ∀ v, ReflTransGen (avoid adj (u 0)) s v → v ≠ u 0 := by
    intro v hv
    induction hv with
    | refl => exact Ne.symm hstart
    | tail _ hbc _ => exact hbc.2.2
  -- Fill phase: colonize every vertex reachable from `s` without `u 0`.
  set R : Finset V := univ.filter (fun v => ReflTransGen (avoid adj (u 0)) s v) with hRdef
  have hRmem : ∀ v, v ∈ R ↔ ReflTransGen (avoid adj (u 0)) s v := fun v => by simp [hRdef]
  have hfill : ReflTransGen (BDStep adj) {s} R := by
    apply ColReach.bd (e := avoid adj (u 0)) (fun a b h => h.1)
    apply colReach_of_paths ((hRmem s).mpr ReflTransGen.refl)
    intro v hv
    exact rtg_restrict (P := fun w' => w' ∈ R) (fun w' hw' => (hRmem w').mpr hw') ((hRmem v).mp hv)
  rw [fixable_at_iff]
  rcases Nat.eq_zero_or_pos k with hk0 | hkpos
  · -- `k = 0`: the path is the single vertex `t = u 0`.
    have ht0 : u 0 = t := by rw [hk0] at hlast; exact hlast
    have hReq : R = univ.erase t := by
      ext v
      rw [hRmem]
      simp only [Finset.mem_erase, Finset.mem_univ, and_true]
      constructor
      · intro hv
        rw [← ht0]
        exact hu0 v hv
      · intro hvt
        exact hreach v hvt (by rw [ht0]; exact hvt)
    rw [hReq] at hfill
    refine ⟨hfill, w, ?_, ?_⟩
    · rw [← ht0]
      exact hw.2 0 (Nat.zero_le k)
    · rw [← ht0]
      exact hw.1
  · have hk : 1 ≤ k := hkpos
    have h0k : u 0 ≠ t := by
      rw [← hlast]
      intro h
      have := hinj 0 k (Nat.zero_le k) le_rfl h
      omega
    have hin : ∃ x, x ≠ t ∧ adj x t := by
      refine ⟨u (k - 1), ?_, ?_⟩
      · rw [← hlast]
        intro h
        have := hinj (k - 1) k (by omega) le_rfl h
        omega
      · have h := hpath (k - 1) (by omega)
        rw [Nat.sub_add_cancel hk, hlast] at h
        exact h
    refine ⟨?_, hin⟩
    by_cases htR : t ∈ R
    · -- `t` holds a mutant: move the wild type from `u 0` to `t` along the path.
      have hReq : R = univ.erase (u 0) := by
        ext v
        rw [hRmem]
        simp only [Finset.mem_erase, Finset.mem_univ, and_true]
        constructor
        · exact hu0 v
        · intro hv
          by_cases hvt : v = t
          · rw [hvt]
            exact (hRmem t).mp htR
          · exact hreach v hvt hv
      rw [hReq] at hfill
      have hmove := move_wild hpath hinj hw k le_rfl
      rw [hlast] at hmove
      exact hfill.trans hmove
    · -- `t` holds a wild type: refill `u 0` from the witness `w`.
      have hReq : R = (univ.erase (u 0)).erase t := by
        ext v
        rw [hRmem]
        simp only [Finset.mem_erase, Finset.mem_univ, and_true]
        constructor
        · intro hv
          refine ⟨?_, hu0 v hv⟩
          intro hvt
          rw [hvt] at hv
          exact htR ((hRmem t).mpr hv)
        · rintro ⟨hvt, hvu⟩
          exact hreach v hvt hvu
      rw [hReq] at hfill
      have hwt : w ≠ t := by rw [← hlast]; exact hw.2 k le_rfl
      have hg : BDStep adj ((univ.erase (u 0)).erase t)
          (insert (u 0) ((univ.erase (u 0)).erase t)) :=
        BDStep.gain (Finset.mem_erase.mpr ⟨hwt, Finset.mem_erase.mpr ⟨hw.2 0 (Nat.zero_le k), Finset.mem_univ w⟩⟩)
          (fun h => (Finset.mem_erase.mp (Finset.mem_erase.mp h).2).1 rfl) hw.1
      rw [insert_erase_erase h0k] at hg
      exact hfill.tail hg

/-- Condition (ii) of SI Proposition S3 for the vertex `t`: in every trajectory from the single
mutant at `u` ending in fixation, once a mutant occupies `t`, a mutant occupies `t` at every later
time. -/
def StaysMutant (adj : V → V → Prop) (u t : V) : Prop :=
  ∀ (X : ℕ → Finset V) (m : ℕ), X 0 = {u} → (∀ i < m, BDStep adj (X i) (X (i + 1))) →
    X m = univ → ∀ i j, i ≤ j → j ≤ m → t ∈ X i → t ∈ X j

/-- SI Proposition S3 (unidirectional bottlenecks). Condition (i): some vertex `w ≠ t` is reachable
from `u` only by paths through `t`. -/
theorem gateway (adj : V → V → Prop) (u t w : V) (hwt : w ≠ t)
    (hi : ¬ ReflTransGen (avoid adj t) u w) (hii : StaysMutant adj u t) : ¬ FixableAt adj u t := by
  rintro ⟨X', hX', hstep, htX'⟩
  obtain ⟨X, m, h0, hm, hs⟩ := exists_traj hX'
  have hYs := traj_snoc hs (hm ▸ hstep)
  set Y : ℕ → Finset V := fun i => if i ≤ m then X i else univ with hY
  have hY0 : Y 0 = {u} := by simp [hY, h0]
  have hYm : Y m = X' := by simp [hY, hm]
  have hYm1 : Y (m + 1) = univ := by simp [hY]
  have hexists : ∃ i ≤ m, t ∈ Y i := by
    by_contra hno
    push_neg at hno
    have hwX' : w ∈ X' := by
      obtain ⟨t', hXt', _⟩ := bdstep_univ hstep rfl
      have htt' : t' = t := by
        by_contra hne
        exact htX' (by rw [hXt']; exact Finset.mem_erase.mpr ⟨fun h => hne h.symm, Finset.mem_univ t⟩)
      rw [hXt', htt']
      exact Finset.mem_erase.mpr ⟨hwt, Finset.mem_univ w⟩
    exact hi (ancestry (m := m) hY0 (fun i hi' => hYs i (by omega)) m le_rfl hno w (hYm ▸ hwX'))
  obtain ⟨i, him, hti⟩ := hexists
  have hlate := hii Y (m + 1) hY0 hYs hYm1 i m him (by omega) hti
  rw [hYm] at hlate
  exact htX' hlate

/-- SI Proposition S3, last sentence: if condition (ii) holds for the initial vertex `u`, then `u` is
not a possible fixation location. -/
theorem gateway_initial (adj : V → V → Prop) (u : V) (hii : StaysMutant adj u u) :
    ¬ FixableAt adj u u := by
  rintro ⟨X', hX', hstep, huX'⟩
  obtain ⟨X, m, h0, hm, hs⟩ := exists_traj hX'
  have hYs := traj_snoc hs (hm ▸ hstep)
  set Y : ℕ → Finset V := fun i => if i ≤ m then X i else univ with hY
  have hY0 : Y 0 = {u} := by simp [hY, h0]
  have hYm : Y m = X' := by simp [hY, hm]
  have hYm1 : Y (m + 1) = univ := by simp [hY]
  have hlate := hii Y (m + 1) hY0 hYs hYm1 0 m (Nat.zero_le m) (by omega)
    (by rw [hY0]; exact Finset.mem_singleton_self u)
  rw [hYm] at hlate
  exact huX' hlate

/-- SI Lemma S1 (single incoming neighbor), for `t ≠ u`. -/
theorem single_in (adj : V → V → Prop) (u t x : V) (htu : t ≠ u) (hin : ∀ y, adj y t ↔ y = x)
    (hx : StaysMutant adj u x) : StaysMutant adj u t := by
  intro X m h0 hs hm i j hij hjm hti
  have hex : ∃ n, t ∈ X n := ⟨i, hti⟩
  have hc_mem : t ∈ X (Nat.find hex) := Nat.find_spec hex
  have hc_le : Nat.find hex ≤ i := Nat.find_min' hex hti
  have hc_min : ∀ n < Nat.find hex, t ∉ X n := fun n hn => Nat.find_min hex hn
  have hc_pos : Nat.find hex ≠ 0 := by
    intro hc0
    rw [hc0, h0, Finset.mem_singleton] at hc_mem
    exact htu hc_mem
  obtain ⟨c', hc'⟩ : ∃ c', Nat.find hex = c' + 1 := ⟨Nat.find hex - 1, by omega⟩
  have htc' : t ∉ X c' := hc_min c' (by omega)
  have htc : t ∈ X (c' + 1) := by rw [← hc']; exact hc_mem
  -- The first mutant at `t` is the offspring of a mutant at `x`.
  have hxc' : x ∈ X c' := by
    rcases BDStep.cases' (hs c' (by omega)) with ⟨a, b, ha, _, hab, hY⟩ | ⟨a, b, _, _, _, hY⟩
    · rw [hY] at htc
      rcases Finset.mem_insert.mp htc with htb | htX
      · rw [← htb] at hab
        rw [← (hin a).mp hab]
        exact ha
      · exact absurd htX htc'
    · rw [hY] at htc
      exact absurd (Finset.mem_of_mem_erase htc) htc'
  have hxstay : ∀ n, c' ≤ n → n ≤ m → x ∈ X n :=
    fun n h1 h2 => hx X m h0 hs hm c' n h1 h2 hxc'
  -- Afterwards `t` never receives a wild-type offspring.
  have hstay : ∀ n, c' + 1 ≤ n → n ≤ m → t ∈ X n := by
    intro n
    induction n with
    | zero => intro h1 _; exact absurd h1 (by omega)
    | succ n ih =>
      intro h1 h2
      by_cases hn : n = c'
      · rw [hn]
        exact htc
      · have htn : t ∈ X n := ih (by omega) (by omega)
        rcases BDStep.cases' (hs n (by omega)) with ⟨a, b, _, _, _, hY⟩ | ⟨a, b, ha, _, hab, hY⟩
        · rw [hY]
          exact Finset.mem_insert_of_mem htn
        · rw [hY]
          refine Finset.mem_erase.mpr ⟨?_, htn⟩
          intro htb
          rw [← htb] at hab
          have hax : a = x := (hin a).mp hab
          rw [hax] at ha
          exact ha (hxstay n (by omega) (by omega))
  exact hstay j (by omega) hjm

end FixLoc
