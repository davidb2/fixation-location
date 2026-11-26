import FixLoc.CLT
import FixLoc.Fourier
import Mathlib.MeasureTheory.Measure.LevyConvergence
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
# Central limit theorem for the number of head advances (SI Theorem S3), part 3

`clt_charFun` gives the limit `e^{-τ²/2}` of the characteristic function of `(H - μ_N)/σ_N`.
Lévy's convergence theorem (`ProbabilityMeasure.tendsto_of_tendsto_charFun` in Mathlib) turns
this limit into convergence in distribution to the standard normal law (`clt_head_advances`).
`clt_head_advances_of_steps` takes the first-step equations of SI Lemma S2 as the hypothesis
instead of the closed form of the generating function.
-/

open Complex Filter Topology MeasureTheory ProbabilityTheory
open scoped Real

namespace FixLoc

noncomputable section

/-- The law of `(H - μ_N)/σ_N` when `p` is the law of `H` on `ℕ`. -/
def stdLaw (r : ℝ) (N : ℕ) (p : PMF ℕ) : ProbabilityMeasure ℝ :=
  ⟨p.toMeasure.map (fun n : ℕ => ((n : ℝ) - meanH r N) / √(varH r N)), inferInstance⟩

lemma charFun_stdLaw (r : ℝ) (N : ℕ) (p : PMF ℕ) (τ : ℝ) :
    charFun (stdLaw r N p : Measure ℝ) τ
      = exp (((-(τ * meanH r N / √(varH r N))) : ℝ) * I)
        * ∑' n, (((p n).toReal : ℝ) : ℂ) * exp (((τ / √(varH r N)) : ℝ) * n * I) := by
  have hφ : AEMeasurable (fun n : ℕ => ((n : ℝ) - meanH r N) / √(varH r N)) p.toMeasure :=
    measurable_from_nat.aemeasurable
  have hcont : Continuous (fun x : ℝ => exp ((τ : ℂ) * (x : ℂ) * I)) := by fun_prop
  have hint : Integrable (fun n : ℕ =>
      exp ((τ : ℂ) * ((((n : ℝ) - meanH r N) / √(varH r N) : ℝ) : ℂ) * I)) p.toMeasure := by
    refine (integrable_const (1 : ℝ)).mono' AEStronglyMeasurable.of_discrete ?_
    refine Eventually.of_forall (fun n => le_of_eq ?_)
    rw [show (τ : ℂ) * ((((n : ℝ) - meanH r N) / √(varH r N) : ℝ) : ℂ) * I
        = (((τ * (((n : ℝ) - meanH r N) / √(varH r N))) : ℝ) : ℂ) * I by push_cast; ring]
    exact Complex.norm_exp_ofReal_mul_I _
  show charFun (p.toMeasure.map (fun n : ℕ => ((n : ℝ) - meanH r N) / √(varH r N))) τ = _
  rw [charFun_apply_real, integral_map hφ hcont.aestronglyMeasurable,
    PMF.integral_eq_tsum p _ hint, ← tsum_mul_left]
  congr 1
  funext n
  rw [Complex.real_smul, mul_left_comm, ← Complex.exp_add]
  congr 2
  push_cast
  ring

/-- The standard normal law. -/
def stdNormal : ProbabilityMeasure ℝ := ⟨gaussianReal 0 1, inferInstance⟩

lemma stdNormal_coe : (stdNormal : Measure ℝ) = gaussianReal 0 1 := rfl

/-- The characteristic function of the standard normal law. -/
lemma charFun_stdNormal (τ : ℝ) :
    charFun (stdNormal : Measure ℝ) τ = exp ((-(τ ^ 2 / 2) : ℝ) : ℂ) := by
  rw [stdNormal_coe, charFun_gaussianReal]
  congr 1
  push_cast
  ring

/-- SI Theorem S3, central limit theorem. Let `p N` be the law of the number `H` of head advances
conditional on fixation. If the characteristic function of `p N` is the generating function of
SI Lemma S2 divided by the fixation probability, then the law of `(H - E[H])/√Var[H]` converges
weakly to the standard normal law. -/
theorem clt_head_advances (r : ℝ) (hr : 1 < r) (p : ℕ → PMF ℕ)
    (hgf : ∀ N, 2 ≤ N → ∀ θ : ℝ, ∑' n, (((p N n).toReal : ℝ) : ℂ) * exp ((θ : ℂ) * n * I)
        = (2 * (alpha r : ℂ) * exp ((θ : ℂ) * I)) ^ (N - 1)
          / ((rhoH r N : ℂ) * PNc N (1 - 4 * (alpha r : ℂ) * beta r * exp ((θ : ℂ) * I)))) :
    Tendsto (fun N : ℕ => stdLaw r N (p N)) atTop
      (𝓝 stdNormal) := by
  apply ProbabilityMeasure.tendsto_of_tendsto_charFun
  intro τ
  rw [charFun_stdNormal]
  exact (clt_charFun r hr (fun N n => (p N n).toReal) hgf τ).congr
    (fun N => (charFun_stdLaw r N (p N) τ).symm)

/-- SI Theorem S3, central limit theorem, from the first-step equations. `F N θ k` is
`E_k[e^{iθH} ; fixation]` for the walk of the block of mutants on the directed cycle of `N`
vertices. The law `p N` of `H` conditional on fixation has characteristic function
`F N θ 1 / ρ_N`. -/
theorem clt_head_advances_of_steps (r : ℝ) (hr : 1 < r) (p : ℕ → PMF ℕ)
    (F : ℕ → ℝ → ℕ → ℂ)
    (hF : ∀ N, 2 ≤ N → ∀ θ : ℝ,
      GFStepsC (alpha r) (beta r) (exp ((θ : ℂ) * I)) N (F N θ))
    (hp : ∀ N, 2 ≤ N → ∀ θ : ℝ,
      ∑' n, (((p N n).toReal : ℝ) : ℂ) * exp ((θ : ℂ) * n * I)
        = F N θ 1 / (rhoH r N : ℂ)) :
    Tendsto (fun N : ℕ => stdLaw r N (p N)) atTop
      (𝓝 stdNormal) := by
  apply clt_head_advances r hr p
  intro N hN θ
  have hα : (alpha r : ℂ) ≠ 0 := by exact_mod_cast (alpha_pos' r hr).ne'
  rw [hp N hN θ, gf_head_advances_complex _ _ _ hα N (by omega) _ (hF N hN θ)]
  ring

end

end FixLoc
