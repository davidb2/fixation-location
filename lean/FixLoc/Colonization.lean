import Mathlib

/-!
# Possible fixation locations under colonization (SI Propositions 4 and 5)

Under colonization (`r → ∞`) only mutants reproduce. A mutant at `x` places an offspring on an
out-neighbor `y`, and the step changes the population only if `y` holds a wild type. The mutant set
therefore grows by one vertex at each active step. Every such growth step has positive probability.
A vertex `t` is a possible fixation location exactly when some sequence of growth steps reaches the
mutant set `V \ {t}` and a mutant in-neighbor of `t` then places the last offspring on `t`.

This file models the colonization process by the growth steps (`ColReach`) and proves
* `possible_fix_iff` (SI Proposition S4): on every finite directed graph, `t` is a possible fixation
  location exactly when `t ≠ u`, every other vertex is reachable from `u` by a directed path avoiding `t`,
  and `t` has an in-neighbor other than `t`;
* `possible_fix_iff_not_cut` (SI Proposition S5): on a connected undirected graph, `t` is a possible
  fixation location exactly when `t ≠ u` and `t` is not a cut vertex.
-/

open Finset Relation

namespace FixLoc

variable {V : Type*}

/-- The edges of the graph with the vertex `t` deleted. -/
def avoid (adj : V → V → Prop) (t : V) (a b : V) : Prop :=
  adj a b ∧ a ≠ t ∧ b ≠ t

/-- `t` is a cut vertex: two other vertices have no path between them avoiding `t`. -/
def IsCutVertex (adj : V → V → Prop) (t : V) : Prop :=
  ∃ a b, a ≠ t ∧ b ≠ t ∧ ¬ ReflTransGen (avoid adj t) a b

/-- A path from inside a set to outside the set has an edge leaving the set. -/
lemma exists_exit {r : V → V → Prop} {S : V → Prop} {a b : V} (h : ReflTransGen r a b) (ha : S a) :
    ¬ S b → ∃ c d, r c d ∧ S c ∧ ¬ S d := by
  induction h with
  | refl => intro hb; exact absurd ha hb
  | @tail b c _ hbc ih =>
    intro hc
    by_cases hSb : S b
    · exact ⟨b, c, hbc, hSb, hc⟩
    · exact ih hSb

/-- Paths of a symmetric relation can be reversed. -/
lemma rtg_symm {r : V → V → Prop} (hr : ∀ a b, r a b → r b a) {a b : V} (h : ReflTransGen r a b) :
    ReflTransGen r b a := by
  induction h with
  | refl => exact ReflTransGen.refl
  | @tail b c _ hbc ih => exact ReflTransGen.head (hr b c hbc) ih

variable [DecidableEq V]

/-- Mutant sets reachable by colonization from a single mutant at `u`. A mutant at `x` places an
offspring on an out-neighbor `y` that holds a wild type. -/
inductive ColReach (adj : V → V → Prop) (u : V) : Finset V → Prop
  | start : ColReach adj u {u}
  | grow {X : Finset V} {x y : V} :
      ColReach adj u X → x ∈ X → adj x y → y ∉ X → ColReach adj u (insert y X)

/-- The initial mutant stays a mutant under colonization. -/
lemma ColReach.mem_start {adj : V → V → Prop} {u : V} {X : Finset V} (h : ColReach adj u X) :
    u ∈ X := by
  induction h with
  | start => simp
  | grow _ _ _ _ ih => exact Finset.mem_insert_of_mem ih

/-- Every vertex of a reachable mutant set without `t` is reachable from `u` avoiding `t`:
trace the parents back. -/
lemma ColReach.reach_avoid {adj : V → V → Prop} {u t : V} {X : Finset V} (h : ColReach adj u X) :
    t ∉ X → ∀ v ∈ X, ReflTransGen (avoid adj t) u v := by
  induction h with
  | start =>
    intro _ v hv
    rw [Finset.mem_singleton] at hv
    subst hv
    exact ReflTransGen.refl
  | @grow X x y _ hx hxy _ ih =>
    intro ht v hv
    have htX : t ∉ X := fun h => ht (Finset.mem_insert_of_mem h)
    have hyt : y ≠ t := fun h => ht (by rw [← h]; exact Finset.mem_insert_self y X)
    have hxt : x ≠ t := fun h => htX (h ▸ hx)
    rcases Finset.mem_insert.mp hv with hvy | hvX
    · rw [hvy]
      exact (ih htX x hx).tail ⟨hxy, hxt, hyt⟩
    · exact ih htX v hvX

variable [Fintype V]

/-- `t` is a possible fixation location: the mutant set `V \ {t}` is reachable, and a mutant
in-neighbor of `t` places the last offspring on `t`. -/
def PossibleFix (adj : V → V → Prop) (u t : V) : Prop :=
  ColReach adj u (univ.erase t) ∧ ∃ x, x ≠ t ∧ adj x t

/-- If every vertex other than `t` is reachable from `u` avoiding `t`, colonization reaches the
mutant set `V \ {t}`: repeatedly add the end of an edge leaving the current mutant set. -/
lemma colReach_erase {adj : V → V → Prop} {u t : V} (htu : t ≠ u)
    (hreach : ∀ v, v ≠ t → ReflTransGen (avoid adj t) u v) : ColReach adj u (univ.erase t) := by
  have key : ∀ n : ℕ, ∀ X : Finset V, ColReach adj u X → t ∉ X →
      ((univ.erase t) \ X).card = n → ColReach adj u (univ.erase t) := by
    intro n
    induction n with
    | zero =>
      intro X hX htX hcard
      have hsub : univ.erase t ⊆ X := by
        intro v hv
        by_contra hvX
        have hmem : v ∈ (univ.erase t) \ X := Finset.mem_sdiff.mpr ⟨hv, hvX⟩
        rw [Finset.card_eq_zero] at hcard
        rw [hcard] at hmem
        exact absurd hmem (Finset.notMem_empty v)
      have hXeq : X = univ.erase t := by
        apply Finset.Subset.antisymm _ hsub
        intro v hv
        exact Finset.mem_erase.mpr ⟨fun h => htX (h ▸ hv), Finset.mem_univ v⟩
      rw [← hXeq]
      exact hX
    | succ n ih =>
      intro X hX htX hcard
      obtain ⟨v, hv⟩ : ((univ.erase t) \ X).Nonempty := by
        rw [← Finset.card_pos, hcard]
        exact Nat.succ_pos n
      have hv' := Finset.mem_sdiff.mp hv
      have hvt : v ≠ t := (Finset.mem_erase.mp hv'.1).1
      obtain ⟨c, d, hcd, hc, hd⟩ :=
        exists_exit (S := fun w => w ∈ X) (hreach v hvt) (ColReach.mem_start hX) hv'.2
      have hdt : d ≠ t := hcd.2.2
      apply ih (insert d X) (ColReach.grow hX hc hcd.1 hd)
      · intro h
        rcases Finset.mem_insert.mp h with h | h
        · exact hdt h.symm
        · exact htX h
      · have hset : (univ.erase t) \ insert d X = ((univ.erase t) \ X).erase d := by
          ext w
          simp only [Finset.mem_sdiff, Finset.mem_insert, Finset.mem_erase, Finset.mem_univ, and_true]
          tauto
        have hdmem : d ∈ (univ.erase t) \ X :=
          Finset.mem_sdiff.mpr ⟨Finset.mem_erase.mpr ⟨hdt, Finset.mem_univ d⟩, hd⟩
        rw [hset, Finset.card_erase_of_mem hdmem, hcard]
        simp
  exact key _ {u} ColReach.start (by simpa using htu) rfl

/-- SI Proposition S4 (possible fixation locations under colonization). -/
theorem possible_fix_iff (adj : V → V → Prop) (u t : V) :
    PossibleFix adj u t ↔
      t ≠ u ∧ (∀ v, v ≠ t → ReflTransGen (avoid adj t) u v) ∧ ∃ x, x ≠ t ∧ adj x t := by
  constructor
  · rintro ⟨hX, hin⟩
    have hu : u ∈ univ.erase t := ColReach.mem_start hX
    have htu : t ≠ u := (Finset.mem_erase.mp hu).1.symm
    refine ⟨htu, ?_, hin⟩
    intro v hv
    exact ColReach.reach_avoid hX (Finset.notMem_erase t univ) v
      (Finset.mem_erase.mpr ⟨hv, Finset.mem_univ v⟩)
  · rintro ⟨htu, hreach, hin⟩
    exact ⟨colReach_erase htu hreach, hin⟩

/-- SI Proposition S5 (colonization on undirected graphs): on a connected graph with symmetric
adjacency, `t` is a possible fixation location exactly when `t ≠ u` and `t` is not a cut vertex. -/
theorem possible_fix_iff_not_cut (adj : V → V → Prop) (hsymm : ∀ a b, adj a b → adj b a)
    (hconn : ∀ a b, ReflTransGen adj a b) (u t : V) :
    PossibleFix adj u t ↔ t ≠ u ∧ ¬ IsCutVertex adj t := by
  rw [possible_fix_iff]
  have hsymm' : ∀ a b, avoid adj t a b → avoid adj t b a := fun a b h => ⟨hsymm a b h.1, h.2.2, h.2.1⟩
  constructor
  · rintro ⟨htu, hreach, -⟩
    refine ⟨htu, ?_⟩
    rintro ⟨a, b, hat, hbt, hab⟩
    exact hab ((rtg_symm hsymm' (hreach a hat)).trans (hreach b hbt))
  · rintro ⟨htu, hcut⟩
    refine ⟨htu, ?_, ?_⟩
    · intro v hv
      by_contra h
      exact hcut ⟨u, v, htu.symm, hv, h⟩
    · obtain ⟨c, d, hcd, hc, hd⟩ :=
        exists_exit (S := fun w => w ≠ t) (hconn u t) htu.symm (by simp)
      have hdt : d = t := by simpa using hd
      exact ⟨c, hc, hdt ▸ hcd⟩

end FixLoc
