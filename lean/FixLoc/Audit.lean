import FixLoc.Walk
import FixLoc.Variance
import FixLoc.FinalPhase
import FixLoc.Colonization
import FixLoc.Fixability
import FixLoc.Fourier
import FixLoc.NeutralBound
import FixLoc.NeutralConstant
import FixLoc.CLTLaw
import FixLoc.Concentration
import FixLoc.Islands
import FixLoc.LargeIslands

/-! Axioms used by the headline theorems (expected: `propext`, `Classical.choice`, `Quot.sound`). -/

#print axioms FixLoc.gf_head_advances
#print axioms FixLoc.fixation_probability
#print axioms FixLoc.mean_head_advances
#print axioms FixLoc.possible_fix_iff
#print axioms FixLoc.possible_fix_iff_not_cut
#print axioms FixLoc.fixable_at_iff
#print axioms FixLoc.bdstep_erase_univ_iff
#print axioms FixLoc.not_fixable_of_unreachable
#print axioms FixLoc.not_fixable_of_no_in
#print axioms FixLoc.refillable_fixable
#print axioms FixLoc.gateway
#print axioms FixLoc.gateway_initial
#print axioms FixLoc.single_in
#print axioms FixLoc.gf_head_advances_complex
#print axioms FixLoc.fourier_location
#print axioms FixLoc.PNc_lower
#print axioms FixLoc.neutral_location
#print axioms FixLoc.variance_head_advances
#print axioms FixLoc.varRem_bounds
#print axioms FixLoc.factorization
#print axioms FixLoc.factorization_strongly_connected
#print axioms FixLoc.final_phase
#print axioms FixLoc.Bf_expansion
#print axioms FixLoc.logBf_expansion
#print axioms FixLoc.gf_factor
#print axioms FixLoc.clt_charFun
#print axioms FixLoc.clt_head_advances
#print axioms FixLoc.clt_head_advances_of_steps
#print axioms FixLoc.gauss_Ioo
#print axioms FixLoc.concentration_location
#print axioms FixLoc.Cneu_lt
#print axioms FixLoc.neutral_location_sharp
#print axioms FixLoc.neutral_location_170
#print axioms FixLoc.Islands.drift1
#print axioms FixLoc.Islands.drift2
#print axioms FixLoc.Islands.harmonic
#print axioms FixLoc.Islands.reach
#print axioms FixLoc.Islands.neutral_fixation
#print axioms FixLoc.Islands.neutral_initial_island
#print axioms FixLoc.Islands.linearization
#print axioms FixLoc.Islands.colonization_linearization
#print axioms FixLoc.Islands.colonization_comparison
#print axioms FixLoc.Islands.neutral_comparison
#print axioms FixLoc.LargeIslands.region
#print axioms FixLoc.LargeIslands.entry
#print axioms FixLoc.LargeIslands.colonization_large_islands
#print axioms FixLoc.LargeIslands.colonization_large_islands_symm
