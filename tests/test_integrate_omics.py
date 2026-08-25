"""
Regression tests for integrate_omics.py.

Locks in:
  - run_imat's epsilon parameter actually being used (was hardcoded)
  - min_growth_frac producing a non-degenerate growth floor (was: no growth constraint at all -> iMAT always returned ~0 growth)
  - The corrected _imat_fixed() big-M formulation, y_fwd/irreversible-forward branches that were broken after the first fix pass
    (only caught via the real Human1 IIS, not any toy model built before forced_negative_flux_model existed)
  - run_gimme's parsimonious=True default resolving alternate-optima flux magnitude blowup
  - GLPK-only solver resolution across all three of MEWpy's internal registries (mewpy.solvers, mewpy.simulation, and reframed via mewpy.simulation),
    plus the module-level optlang.Model monkeypatch that stops optlang silently binding to Gurobi's interface whenever gurobipy is merely importable
    (see integrate_omics.py's module docstring)
  - Gurobi and SCIP have been deliberately removed from the solve path entirely; explicit requests for either now raise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from tools.integrate_omics import run_imat, run_gimme, run_pfba_baseline, run_continuous_meeson


class TestIMATGrowthFloor:
    def test_no_floor_reproduces_vanilla_imat_near_zero_growth(self, linear_chain_model):
        """min_growth_frac=None should reproduce iMAT's literature-faithful (growth-agnostic) behaviour: 
        growth collapses near 0, since nothing in the objective rewards biomass."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        fluxes, info = run_imat(linear_chain_model, expr, min_growth_frac=None)
        assert str(info["status"]) == "OPTIMAL"
        assert info["predicted_growth"] < 1e-3

    def test_floor_produces_genuine_nonzero_growth(self, linear_chain_model):
        """This is the core fix: before it, ANY growth floor > 0 made iMAT return INFEASIBLE (empty constraints dict bug), or with constraints={} it silently returned ~0 regardless of floor."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        fluxes, info = run_imat(linear_chain_model, expr, min_growth_frac=0.1)
        assert str(info["status"]) == "OPTIMAL"
        assert info["predicted_growth"] > 0

    def test_epsilon_parameter_is_actually_used(self, linear_chain_model):
        """Regression test for the hardcoded-epsilon bug: run_imat used to always pass epsilon=0.001 to iMAT() regardless of what the caller specified.
        Verify the info dict reports the value the caller actually passed."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_imat(linear_chain_model, expr, epsilon=0.5, min_growth_frac=0.1)
        assert info["epsilon"] == 0.5


class TestIMATBigMFormulation:
    """The two-round sign-bug fix. forced_negative_flux_model is the key fixture: specifically requires a high-expression, reversible reaction to carry negative flux to satisfy a growth-floor-forced metabolite balance.
    The y_fwd branch's original bug made this structurally impossible (the reaction could only ever have flux >= 0, regardless of which binary value the solver chose).
    This was NOT caught by a simpler toy model (see linear_chain_model tests above, which passed even with the bug still present)."""

    def test_high_expression_reversible_reaction_can_go_negative(self, forced_negative_flux_model):
        expr = pd.Series({"g1": 10.0, "g2": 10.0, "g_dummy": 1.0})
        fluxes, info = run_imat(forced_negative_flux_model, expr, lower_percentile=25, upper_percentile=50, min_growth_frac=0.1,)
        assert str(info["status"]) == "OPTIMAL"
        assert fluxes["R_R1"] < 0, ("R1 must go negative to satisfy the metabolite balance under "
            "the growth floor. If this fails, the y_fwd big-M branch has "
            "regressed to only ever allowing flux >= 0.")

    def test_not_infeasible_with_growth_floor(self, forced_negative_flux_model):
        """This exact model structure returned INFEASIBLE on the real Human1 model even after the first (incomplete) fix pass."""
        expr = pd.Series({"g1": 10.0, "g2": 10.0, "g_dummy": 1.0})
        _, info = run_imat(forced_negative_flux_model, expr, lower_percentile=25, upper_percentile=50, min_growth_frac=0.1,)
        assert str(info["status"]) != "INFEASIBLE"


class TestGimmeParsimonious:
    def test_parsimonious_defaults_to_true(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr)
        assert info["parsimonious"] is True

    def test_parsimonious_reduces_flux_magnitude(self, linear_chain_model):
        """parsimonious=True should not produce larger flux magnitudes than parsimonious=False for the same problem."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        fluxes_par, _ = run_gimme(linear_chain_model, expr, parsimonious=True)
        fluxes_nonpar, _ = run_gimme(linear_chain_model, expr, parsimonious=False)
        assert fluxes_par.abs().sum() <= fluxes_nonpar.abs().sum() + 1e-6


class TestSolverSelection:
    """GLPK only now (Gurobi and SCIP deliberately removed — see tools/integrate_omics.py's module docstring).
    Locks in that GIMME and iMAT both resolve to GLPK regardless of which accepted alias ('auto', 'glpk', 'optlang') is passed, and that anything else is rejected clearly rather than silently ignored."""

    def test_auto_resolves_and_is_reported(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr, solver="auto")
        assert info["solver"] == "glpk"

    def test_gimme_works_on_glpk(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr, solver="glpk")
        assert info["solver"] == "glpk"
        assert str(info["status"]).upper() == "OPTIMAL"

    def test_gimme_works_on_optlang_alias(self, linear_chain_model):
        """'optlang' is accepted as an alias for 'glpk' — both must resolve to the same reported solver name."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr, solver="optlang")
        assert info["solver"] == "glpk"
        assert str(info["status"]).upper() == "OPTIMAL"

    def test_imat_works_on_glpk(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_imat(linear_chain_model, expr, min_growth_frac=0.1, solver="glpk")
        assert info["solver"] == "glpk"
        assert str(info["status"]) == "OPTIMAL"

    def test_imat_big_m_fix_holds_on_glpk(self, forced_negative_flux_model):
        """The sign-bug fix must not be an accidental artifact of Gurobi's specific numerics.
        re-run the exact fixture that caught the real bug on GLPK, the only backend now in the solve path."""
        expr = pd.Series({"g1": 10.0, "g2": 10.0, "g_dummy": 1.0})
        _, info = run_imat(forced_negative_flux_model, expr, lower_percentile=25, upper_percentile=50, min_growth_frac=0.1, solver="glpk",)
        assert str(info["status"]) != "INFEASIBLE"

    def test_unknown_solver_name_raises_clear_error(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        with pytest.raises(ValueError):
            run_gimme(linear_chain_model, expr, solver="not_a_real_solver")

    def test_gurobi_explicitly_rejected(self, linear_chain_model):
        """Explicitly requesting gurobi must fail clearly, not be
        silently honoured or silently downgraded to glpk."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        with pytest.raises(ValueError):
            run_gimme(linear_chain_model, expr, solver="gurobi")


class TestCobraSolverResolution:
    """Real bug: run_pfba_baseline/run_continuous_meeson use plain cobrapy (pfba()/model.optimize()) a separate solver mechanism from run_gimme/run_imat's MEWpy-based one.
    Fixing solver auto-detection for gimme/imat did nothing for these two, as they kept using whatever cobrapy's own default solver happened to be, with no check on licence usability. 
    On a restricted Gurobi licence, this surfaced as a raw GurobiError crashing pfba_baseline."""

    def test_pfba_baseline_falls_back_to_glpk(self, linear_chain_model):
        """Confirms 'auto' resolves to glpk specifically — the only backend now in the solve path (see module docstring)."""
        _, info = run_pfba_baseline(linear_chain_model, solver="auto")
        assert info["solver"] == "glpk"

    def test_pfba_baseline_reports_resolved_solver(self, linear_chain_model):
        _, info = run_pfba_baseline(linear_chain_model, solver="glpk")
        assert info["solver"] == "glpk"
        assert info["status"] == "optimal"

    def test_continuous_meeson_reports_resolved_solver(self, linear_chain_model):
        import pandas as pd
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_continuous_meeson(linear_chain_model, expr, solver="glpk")
        assert info["solver"] == "glpk"

    def test_scip_explicitly_rejected_for_cobra_methods(self, linear_chain_model):
        """scip has been removed from the solve path entirely. Must fail clearly."""
        with pytest.raises(ValueError, match="SCIP"):
            run_pfba_baseline(linear_chain_model, solver="scip")

    def test_gimme_internal_wt_growth_computation_does_not_crash(self, linear_chain_model):
        """Same propagation bug (3): run_gimme's main solve goes through _resolve_and_activate_solver (MEWpy),
        but also makes a separate internal pfba(model_copy) call —plain cobrapy again — just to compute the wt_growth reference value. 
        That internal call is a completely separate solver mechanism (_resolve_cobra_solver) and must resolve cleanly on its own."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr)
        assert str(info["status"]).upper() == "OPTIMAL"

    def test_imat_internal_wt_growth_computation_does_not_crash(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_imat(linear_chain_model, expr, min_growth_frac=0.1)
        assert str(info["status"]) == "OPTIMAL"

    def test_dispatcher_forwards_solver_to_pfba_baseline(self, linear_chain_model):
        """The master dispatcher's pfba_baseline branch didn't forward **kwargs at all — passing solver= through run_integration() would raise TypeError. Confirms the fix."""
        from tools.integrate_omics import run_integration
        _, info = run_integration(linear_chain_model, method="pfba_baseline", solver="glpk")
        assert info["solver"] == "glpk"



class TestReframedSolverLayer:
    def test_gimme_library_call_respects_resolved_solver_via_reframed(self, linear_chain_model):
        """Same propagation bug (4):
        run_gimme calls MEWpy's own shipped GIMME() library function, which internally creates a new reframed-backed Simulator whose __init__ derives reframed's solver from mewpy.simulation's registry
        (a third MEWpy-internal registry, distinct from mewpy.solvers). 
        Setting mewpy.solvers' default alone does not propagate here.
        Confirmed via crash: mewpy.solvers correctly set, but reframed still tried a stale default and crashed."""
        import mewpy.simulation as msim

        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr, solver="glpk")

        assert info["solver"] == "glpk"
        assert str(info["status"]).upper() == "OPTIMAL"
        assert msim.get_default_solver() == "glpk", (
            "mewpy.simulation's registry was not updated — the "
            "reframed-backed Simulator created inside GIMME() would "
            "still derive a stale solver choice from it.")

    def test_gimme_library_call_works_with_glpk_optlang_naming_mismatch(self, linear_chain_model):
        """mewpy.solvers calls this backend 'optlang', mewpy.simulation calls the same backend 'glpk', 
        both registries need setting with their own correct name, and both aliases passed to run_gimme must resolve to the same reported 'glpk'."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_gimme(linear_chain_model, expr, solver="optlang")
        assert info["solver"] == "glpk"
        assert str(info["status"]).upper() == "OPTIMAL"



class TestImatTimeLimitAndMipGap:
    """Covers time_limit/mip_gap feature and a regression caught while adding it: 
    restoring default_parameters after use must distinguish "key was absent before" from "key was explicitly None before".
    Always reassigning (even to None) left stray keys in the shared default_parameters dict that broke unrelated tests when run as part of the full suite."""

    def test_time_limit_and_mip_gap_accepted_without_error(self, linear_chain_model):
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        _, info = run_imat(linear_chain_model, expr, min_growth_frac=0.1, time_limit=60.0, mip_gap=0.05,)
        assert info["time_limit"] == 60.0
        assert info["mip_gap"] == 0.05
        assert str(info["status"]) == "OPTIMAL"

    def test_default_parameters_not_polluted_after_use(self, linear_chain_model):
        """Regression check: default_parameters must have exactly the same set of keys after run_imat as before."""
        from mewpy.solvers.solver import default_parameters, Parameter

        keys_before = set(default_parameters.keys())
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        run_imat(linear_chain_model, expr, min_growth_frac=0.1, time_limit=60.0, mip_gap=0.05,)
        keys_after = set(default_parameters.keys())
        assert keys_after == keys_before, (f"default_parameters keys changed after run_imat: "
                                           f"added {keys_after - keys_before}, removed {keys_before - keys_after}")

    def test_subsequent_call_without_time_limit_unaffected(self, linear_chain_model):
        """A second call with no time_limit/mip_gap must not accidentally inherit settings from a prior call that used them.
        This proves the restore is real and not key presence."""
        expr = pd.Series({"g1": 10.0, "g2": 1.0, "g3": 5.0})
        run_imat(linear_chain_model, expr, min_growth_frac=0.1, time_limit=60.0, mip_gap=0.05,)
        _, info = run_imat(linear_chain_model, expr, min_growth_frac=0.1)
        assert info["time_limit"] is None
        assert info["mip_gap"] is None