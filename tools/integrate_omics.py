"""
Transcriptomics integration algorithms for constraint-based metabolic models.

Methods:
  1. pfba_baseline      — pFBA with no expression data (benchmark)
  2. continuous_meeson  — continuous bounds scaling (Meeson & Schwartz 2024)
  3. gimme              — GIMME via MEWpy (Becker & Palsson 2008)
  4. imat               — iMAT via MEWpy (Shlomi et al. 2008)

SOLVER: GLPK only. 
Gurobi and SCIP have both been deliberately removed from the solve path.
Gurobi because the Streamlit deployment has no licence and was being silently selected by optlang regardless of any solver namepassed to MEWpy's own registries
(confirmed because gurobipy being merely importable, licensed or not, was enough for optlang to bind its generic Model class to Gurobi's interface)
SCIP because it returned UNKNOWN status on iMAT under default settings and needs further investigation for future use.

All methods return: (fluxes: pd.Series, info: dict). `info` always includes at least {"method", "solver", "predicted_growth", "status"}, with "solver" always "glpk" at present."""

import cobra
import pandas as pd
from cobra.flux_analysis import pfba

# ─────────────────────────────────────────────────────────────────────────────
# Force GLPK as the sole solver backend.
# ─────────────────────────────────────────────────────────────────────────────
#
# optlang (the library MEWpy's GIMME/iMAT solve through) does its OWN solver auto-detection, independent of mewpy.solvers/mewpy.simulation: 
# at import time it tries each interface module in its own priority order and binds its generic Model/Variable/Constraint/Objective to whichever it finds first. 
# Confirmed empirically (see check_optlang_solver.py): with gurobipy merely *importable* — regardless of whether the licence is valid or even present 
# optlang.Model bound to optlang.gurobi_interface.Model, not GLPK, no matter what solver name was passed anywhere in mewpy's own registries.
# This "worked" only by accident while a real Gurobi licence happened to be active, and crashed the moment it wasn't (a size-limited-licence GurobiError deep inside what should have been a GLPK solve).
#
# Patched here, at THIS module's import time, before any mewpy module has been imported anywhere in the process,
# mewpy's own imports of optlang are all local/lazy, inside the functions below, and this file is the only one in the codebase that imports mewpy,
# so mewpy's `from optlang import Model` (whenever it first runs) picks up GLPK's interface instead.

import optlang
import optlang.glpk_interface as _glpk_interface

optlang.Model = _glpk_interface.Model
optlang.Variable = _glpk_interface.Variable
optlang.Constraint = _glpk_interface.Constraint
optlang.Objective = _glpk_interface.Objective
# Belt-and-suspenders: some optlang-consuming code checks this dict directly rather than optlang.Model.
# Force it False so any such check also reports Gurobi as unavailable, consistent with the above.
optlang.available_solvers["GUROBI"] = False


def _glpk_scale(solver) -> None:
    """
    Enable GLPK's automatic problem scaling AND presolve immediately before a simplex solve.
    Two independent mitigations for GLPK's native "Assertion failed:
    teta >= 0.0" crash (spxchuzr.c), a numerical-degeneracy failure in its dual simplex ratio test seen on large, 
    poorly-scaled LPs (genome-scale metabolic models are notoriously badly scaled. 
    flux bounds spanning 0.0001 to 1000, tiny biomass coefficients). 
    Scaling is a mathematically-equivalent reformulation of the same problem (same optimal solution, better numerical conditioning);
    presolve is a DIFFERENT mechanism — simplifying the problem before simplex runs, which can avoid the exact degenerate pivot sequence that triggers this assertion even when scaling alone doesn't. 
    This specific assertion is a long-documented GLPK weakness that doesn't always respond to scaling on its own, hence trying both.

    Object chain verified directly (check_glpk_problem_access.py) for mewpy.solvers.optlang_solver.OptLangSolver on this GLPK backend:
      - solver.problem is an optlang.glpk_interface.Model, and solver.problem.problem is the raw swiglpk glp_prob pointer that GLPK's own glp_scale_prob() operates on.
      - solver.problem.configuration is optlang's own public, documented, cross-backend Configuration object (not an internal implementation detail)
      -.presolve is part of optlang's standard Configuration interface.
    Both silently no-op if unavailable (e.g. a non-GLPK backend rather than failing the whole solve over a best-effort numerical nicety.

    Must be called again before EVERY solve. GLPK's scale factors are matrix-shaped, and further solver.add_variable/ add_constraint calls between solves change that shape."""
    try:
        import swiglpk
        raw_problem = solver.problem.problem
        swiglpk.glp_scale_prob(raw_problem, swiglpk.GLP_SF_AUTO)
    except Exception:
        pass
    try:
        solver.problem.configuration.presolve = True
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Corrected iMAT — works around a sign bug in MEWpy 1.0.0's shipped iMAT()
# ─────────────────────────────────────────────────────────────────────────────
#
# MEWpy 1.0.0's mewpy.omics.integration.imat.iMAT() has a sign-transcription
# bug in its big-M constraints. In ALL 6 of its branches (not just some), the
# binary variable's "off" state (which is supposed to leave the constraint
# non-binding) instead forces the reaction's flux to an infeasible extreme,
# or — in the y_fwd / irreversible-forward branches — silently removes the
# reaction's ability to go negative regardless of which binary value is
# chosen. The bug is visible directly in the library's own source: its
# in-code derivation comment for the low-expression branch states the
# intended constraint is "flux + M*x <= epsilon + M", but the actual
# solver.add_constraint(...) call three lines below implements the opposite
# sign ("x_var: -M", RHS "epsilon - M + 0.001").
#
# Practical effect: any reaction in low_coeffs is unconditionally forced
# near-zero (not "optionally rewarded near-zero" as intended); reversible
# reactions in high_coeffs are forced into either a forward+reverse
# contradiction (y_rev branch) or a one-sided "can never go negative,
# regardless of y" restriction (y_fwd branch) that silently deletes half
# their natural flux range. The y_fwd-branch flaw is the more insidious of
# the two: it looks superficially plausible (the "active" case is
# correctly forced), and a small toy model can easily pass by never
# needing that reaction to go negative, hiding the bug. It only surfaced
# here once tested against the real Human1 model, via Gurobi's IIS
# (irreducible inconsistent subsystem) on a genuinely infeasible case,
# which traced the conflict to exactly this: a growth-floor-forced
# positive term in a metabolite balance that could only be cancelled by a
# high-expression reversible reaction going negative — which this bug
# prohibited outright, independent of the epsilon/tolerance/growth-floor
# values involved.
#
# Fixed here as a local fork rather than monkey-patching the installed
# package, so it's visible, auditable, and doesn't depend on an upstream
# fix landing in a version pin. Verified against: (1) a synthetic linear
# 3-reaction toy model exercising the low_coeffs/y_rev branches, and (2) a
# second synthetic toy model specifically constructed to force a
# high-expression reversible reaction negative (mirroring the exact
# metabolite-balance structure found in Human1's IIS), which the
# pre-fix y_fwd branch could not satisfy and the fixed version resolves
# correctly (status OPTIMAL, reaction flux negative as required, y_fwd=0
# and y_rev=1 chosen consistently).

def _imat_fixed(model, expr, constraints=None, cutoff=(25, 75), condition=0, epsilon=1, build_model=False, parsimonious=False,):
    """Corrected fork of mewpy.omics.integration.imat.iMAT (v1.0.0).
    Same signature and behaviour as MEWpy's iMAT, minus the big-M sign bug
    described above. See module-level comment for details.

    parsimonious : bool
        Default False — preserves exact existing behaviour/results
        unless explicitly opted into. If True, adds a second solve
        after the primary one: pin the achieved classification-
        consistency score, then minimize total flux magnitude as a
        secondary objective (the same alternate-optima concern
        run_gimme's parsimonious=True already addresses — iMAT's
        objective only rewards the classification score, so every
        reaction OUTSIDE the high/low classification is completely
        free to land on any flux value the solver happens to pick,
        with zero tiebreaker otherwise).

        Uses FRESH variable names (pflux_p_/pflux_n_ prefix), not the
        pos/neg pair already built above for the big-M reformulation
        — deliberately: those existing variables aren't referenced
        anywhere else in this function after being defined, so their
        exact role in the tested, working big-M formulation isn't
        something to risk reusing incorrectly. This adds new,
        independent, standard abs-value-linearization variables
        instead, touching nothing already validated.

        PERFORMANCE WARNING: unlike run_gimme's parsimonious pass
        (a plain LP), this secondary solve is STILL a MILP — the
        binary classification variables remain free, only pinned
        indirectly via the score constraint — so this can roughly
        double solve time or worse, not a cheap add-on. Opt in
        deliberately, and re-run check_imat_stability.py afterward:
        this changes what iMAT's output means, so any results
        already generated with parsimonious=False need regenerating
        to be comparable, not silently mixed with new results.
    """
    from copy import deepcopy
    from mewpy.solvers import solver_instance
    from mewpy.solvers.solver import VarType
    from mewpy.simulation import get_simulator
    from mewpy.omics.integration.gimme import ExpressionSet
    from mewpy.solvers.solution import to_simulation_result

    if not isinstance(cutoff, tuple) or len(cutoff) != 2:
        raise ValueError(f"cutoff must be (low, high) percentiles, got: {cutoff}")
    low_cutoff, high_cutoff = cutoff
    if not (0 <= low_cutoff < high_cutoff <= 100):
        raise ValueError(f"cutoff must satisfy 0<=low<high<=100, got: ({low_cutoff}, {high_cutoff})")

    sim = get_simulator(model) if not build_model else get_simulator(deepcopy(model))

    if isinstance(expr, ExpressionSet):
        from mewpy.omics import Preprocessing
        pp = Preprocessing(sim, expr)
        coeffs, _ = pp.percentile(condition, cutoff=cutoff)
        low_coeffs, high_coeffs = coeffs
    else:
        low_coeffs, high_coeffs = expr

    solver = solver_instance(sim)
    if not constraints:
        constraints = {}

    for r_id in sim.reactions:
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            solver.add_variable(pos, 0, float("inf"), update=False)
            solver.add_variable(neg, 0, float("inf"), update=False)
    solver.update()
    for r_id in sim.reactions:
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            solver.add_constraint("c" + pos, {r_id: -1, pos: 1}, ">", 0, update=False)
            solver.add_constraint("c" + neg, {r_id: 1, neg: 1}, ">", 0, update=False)
    solver.update()

    objective = list()

    # Highly-expressed reactions: reward |flux| >= epsilon (active).
    # ALL FOUR sub-branches here needed the sign fix, not just two as I
    # originally thought. y_fwd/irreversible-forward looked plausible on
    # first read and passed my earlier toy-model test (which never
    # exercised their y=0 case, so the bug stayed hidden), but re-deriving
    # from scratch and testing against the real Human1 IIS output showed
    # they have the same "off state isn't actually vacuous" flaw: with a
    # +M coefficient, y=0 forces flux >= M (~1100) instead of leaving the
    # reaction's full natural range (including negative flux) untouched.
    # The correct form needs -M here.
    for r_id, val in high_coeffs.items():
        lb, ub = sim.get_reaction_bounds(r_id)
        M = max(abs(lb), abs(ub)) + 100

        if lb < 0 and ub > 0:
            y_fwd = "y_" + r_id + "_fwd"
            objective.append(y_fwd)
            solver.add_variable(y_fwd, 0, 1, vartype=VarType.BINARY, update=True)
            solver.add_constraint("c" + y_fwd, {r_id: 1, y_fwd: -M}, ">", epsilon - M - 0.001, update=False)

            y_rev = "y_" + r_id + "_rev"
            objective.append(y_rev)
            solver.add_variable(y_rev, 0, 1, vartype=VarType.BINARY, update=True)
            solver.add_constraint("c" + y_rev, {r_id: 1, y_rev: M}, "<", M - epsilon + 0.001, update=False)

        elif lb >= 0:
            y = "y_" + r_id
            objective.append(y)
            solver.add_variable(y, 0, 1, vartype=VarType.BINARY, update=True)
            solver.add_constraint("c" + y, {r_id: 1, y: -M}, ">", epsilon - M - 0.001, update=False)

        else:  # ub <= 0
            y = "y_" + r_id
            objective.append(y)
            solver.add_variable(y, 0, 1, vartype=VarType.BINARY, update=True)
            solver.add_constraint("c" + y, {r_id: 1, y: M}, "<", M - epsilon + 0.001, update=False)

    solver.update()

    # Lowly-expressed reactions: reward -epsilon < flux < epsilon (inactive).
    for r_id, val in low_coeffs.items():
        lb, ub = sim.get_reaction_bounds(r_id)
        M = max(abs(lb), abs(ub)) + 100

        x_var = "x_" + r_id
        objective.append(x_var)
        solver.add_variable(x_var, 0, 1, vartype=VarType.BINARY, update=True)
        solver.add_constraint("c" + x_var + "_upper", {r_id: 1, x_var: M}, "<", epsilon + M + 0.001, update=False)
        solver.add_constraint("c" + x_var + "_lower", {r_id: 1, x_var: -M}, ">", -epsilon - M - 0.001, update=False)

    solver.update()

    objective_dict = {x: 1 for x in objective}
    solution = solver.solve(objective_dict, minimize=False, constraints=constraints)

    if parsimonious:
        pre_solution = solution
        solver.add_constraint("imat_obj", objective_dict, "=", pre_solution.fobj)

        flux_objective = dict()
        for r_id in sim.reactions:
            lb, ub = sim.get_reaction_bounds(r_id)
            bound = max(abs(lb), abs(ub))
            pflux_p, pflux_n = "pflux_p_" + r_id, "pflux_n_" + r_id
            solver.add_variable(pflux_p, 0, bound, update=False)
            solver.add_variable(pflux_n, 0, bound, update=False)
            flux_objective[pflux_p] = 1
            flux_objective[pflux_n] = 1
        solver.update()

        for r_id in sim.reactions:
            pflux_p, pflux_n = "pflux_p_" + r_id, "pflux_n_" + r_id
            # Standard abs-value linearization: pflux_p >= flux,
            # pflux_n >= -flux; minimizing their sum forces
            # pflux_p + pflux_n == |flux| at the optimum.
            solver.add_constraint("c" + pflux_p, {r_id: -1, pflux_p: 1}, ">", 0, update=False)
            solver.add_constraint("c" + pflux_n, {r_id: 1, pflux_n: 1}, ">", 0, update=False)
        solver.update()

        solution = solver.solve(flux_objective, minimize=True, constraints=constraints)
        solver.remove_constraint("imat_obj")
        solution.pre_solution = pre_solution

    if build_model:
        rx_to_delete = [r_id for r_id in sim.reactions
                        if abs(solution.values.get(r_id, 0)) < epsilon]
        sim.remove_reactions(rx_to_delete)

    res = to_simulation_result(model, None, constraints, sim, solution)
    if hasattr(solution, "pre_solution"):
        res.pre_solution = solution.pre_solution
    return (res, sim) if build_model else res


# ─────────────────────────────────────────────────────────────────────────────
# Corrected GIMME — works around reframed.solvers being unusable
# ─────────────────────────────────────────────────────────────────────────────
#
# MEWpy's shipped mewpy.omics.integration.gimme.GIMME() is algorithmically
# fine (unlike iMAT above), but has exactly one line that breaks in this
# environment: `wt_solution = sim.simulate(constraints=constraints)`. That
# call routes through reframed.solvers.solver_instance() ->
# get_default_solver(), which reads reframed's OWN internal `solvers` dict
# — populated at reframed's import time by a set of
# `try: from .X_solver import Y; except ImportError: pass` blocks in
# reframed/solvers/__init__.py. Confirmed by direct inspection
# (check_reframed_registration.py / check_reframed_pulp.py): the installed
# reframed release's __init__.py references `.optlang_solver` and
# `.pulp_solver`, but neither file actually exists in the package (only
# gurobi_solver.py/cplex_solver.py/scip_solver.py do) — so ImportError is
# raised as ModuleNotFoundError, silently caught, and `solvers` ends up
# completely empty regardless of what's installed (gurobipy, PySCIPOpt,
# optlang, pulp — none of it matters, since the glue file that would wire
# any of them into reframed's registry is simply missing from this release).
# This is a bug/inconsistency in reframed itself, not anything about this
# pipeline's solver configuration.
#
# Every OTHER line in GIMME() already uses `solver`, the object returned by
# mewpy.solvers.solver_instance(sim) — a completely different, working
# codepath (the same one _imat_fixed() above already relies on
# successfully). So the fix mirrors _imat_fixed()'s approach exactly:
# fork the function, and replace just the one sim.simulate() call with an
# equivalent solve on the already-working `solver` object. Everything else
# below is unchanged from MEWpy's own GIMME() (see its full source for
# comparison — Becker & Palsson 2008).
#
# Only build_model=False is implemented, since that's the only mode
# run_gimme() ever uses.

def _gimme_fixed(model, expr, biomass=None, condition=0, cutoff=25, growth_frac=0.9, constraints=None, parsimonious=False,):
    """Corrected fork of mewpy.omics.integration.gimme.GIMME() (v1.0.0),
    build_model=False only. Same signature/behaviour, minus the
    sim.simulate() call that fails when reframed.solvers is empty — see
    module-level comment above for the full explanation."""
    from math import inf
    from mewpy.simulation import get_simulator
    from mewpy.solvers import solver_instance
    from mewpy.solvers.solution import to_simulation_result
    from mewpy.omics.integration.gimme import ExpressionSet, Preprocessing

    sim = get_simulator(model)

    if isinstance(expr, ExpressionSet):
        pp = Preprocessing(sim, expr)
        coeffs, threshold = pp.percentile(condition, cutoff=cutoff)
    else:
        coeffs = expr
        threshold = cutoff

    solver = solver_instance(sim)

    if biomass is None:
        biomass = sim.biomass_reaction

    if not constraints:
        constraints = {}

    # Original: wt_solution = sim.simulate(constraints=constraints)
    # (routes through the broken reframed.solvers registry — see above).
    # Equivalent solve via the working `solver` object instead: maximise
    # biomass alone under the current constraints. .fobj is the maximised
    # objective value directly (avoids any risk of a key-name mismatch
    # indexing into .values).
    _glpk_scale(solver)
    wt_solution = solver.solve({biomass: 1}, minimize=False, constraints=constraints)
    wt_growth = wt_solution.fobj

    # add growth constraint (matches original: overwrites any prior
    # constraints[biomass] the caller may have set, same as MEWpy's own
    # GIMME() does)
    constraints[biomass] = (growth_frac * wt_growth, inf)

    # Make model irreversible via solver variables (build_model=False
    # strategy only — see original GIMME()'s docstring for the two
    # strategies; build_model=True is not implemented here)
    for r_id in sim.reactions:
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            solver.add_variable(pos, 0, inf, update=False)
            solver.add_variable(neg, 0, inf, update=False)
    solver.update()

    for r_id in sim.reactions:
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            solver.add_constraint("c" + pos, {r_id: -1, pos: 1}, ">", 0, update=False)
            solver.add_constraint("c" + neg, {r_id: 1, neg: 1}, ">", 0, update=False)
    solver.update()

    # define the objective
    objective = dict()
    for r_id, val in coeffs.items():
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            objective[pos] = val
            objective[neg] = val
        else:
            objective[r_id] = val

    _glpk_scale(solver)
    solution = solver.solve(objective, minimize=True, constraints=constraints)

    if parsimonious:
        pre_solution = solution

        solver.add_constraint("obj", objective, "=", pre_solution.fobj)
        objective = dict()

        for r_id in sim.reactions:
            lb, _ = sim.get_reaction_bounds(r_id)
            if lb < 0:
                pos, neg = r_id + "_p", r_id + "_n"
                objective[pos] = 1
                objective[neg] = 1
            else:
                objective[r_id] = 1

        _glpk_scale(solver)
        solution = solver.solve(objective, minimize=True, constraints=constraints)
        solver.remove_constraint("obj")
        solution.pre_solution = pre_solution

    # Reconstruct net flux for reversible reactions before deleting split
    # variables
    for r_id in sim.reactions:
        lb, _ = sim.get_reaction_bounds(r_id)
        if lb < 0:
            pos, neg = r_id + "_p", r_id + "_n"
            net_flux = solution.values.get(pos, 0) - solution.values.get(neg, 0)
            solution.values[r_id] = net_flux
            if pos in solution.values:
                del solution.values[pos]
            if neg in solution.values:
                del solution.values[neg]

    res = to_simulation_result(model, solution.fobj, constraints, sim, solution)
    if hasattr(solution, "pre_solution"):
        res.pre_solution = solution.pre_solution

    return res


# ─────────────────────────────────────────────────────────────────────────────
# GPR rule evaluation — shared by all expression-based methods
# ─────────────────────────────────────────────────────────────────────────────

def apply_gpr_expression(rxn: cobra.Reaction, expr: pd.Series) -> float | None:
    """
    Map gene expression to a reaction capacity value via its GPR rule.

    AND  (enzyme complex) → minimum across subunits
    OR   (isozymes)       → sum of isoenzyme values
    Single gene           → expression value directly
    Mixed andor           → None (skipped; only 1.6% of Human1 reactions)
    """
    rule = rxn.gene_reaction_rule.strip() if rxn.gene_reaction_rule else ""
    if not rule:
        return None

    available = {g.id: expr[g.id]
                 for g in rxn.genes
                 if g.id in expr.index}
    if not available:
        return None

    rule_lower = rule.lower()
    has_and = " and " in rule_lower
    has_or  = " or "  in rule_lower

    if has_and and has_or:
        return None
    elif has_and:
        return min(available.values())
    elif has_or:
        return sum(available.values())
    else:
        return list(available.values())[0]


def _resolve_cobra_solver(model: cobra.Model, solver: str = "auto") -> str:
    """
    Set model.solver (COBRApy's own optlang-based solver interface,
    used by run_pfba_baseline/run_continuous_meeson) to GLPK — the
    only backend this pipeline supports. See module docstring for why
    Gurobi/SCIP were removed.

    'auto' and 'glpk' both resolve to 'glpk'. Any other explicit
    request (gurobi, scip, cplex, coinor_cbc) raises a clear error
    rather than silently overriding it, so a stale solver='gurobi'
    call site is caught rather than quietly doing something different
    from what was asked.
    """
    if solver not in ("auto", "glpk"):
        raise ValueError(
            f"solver='{solver}' is not supported — this pipeline runs "
            f"GLPK only (Gurobi and SCIP were deliberately removed, "
            f"see module docstring). Use 'auto' or 'glpk'.")
    model.solver = "glpk"
    return "glpk"


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: pFBA baseline
# ─────────────────────────────────────────────────────────────────────────────

def run_pfba_baseline(model: cobra.Model, media: dict | None = None, solver: str = "auto",) -> tuple[pd.Series, dict]:
    """
    Parsimonious FBA with no transcriptomics data.
    Always run as a comparison benchmark alongside integration methods.
    Machado & Herrgard (2014) showed pFBA often equals or outperforms
    transcriptomics integration methods for flux prediction accuracy.

    solver : str
        'auto' or 'glpk' (only supported values — see module docstring).
    """
    model = model.copy()
    if media:
        _apply_media(model, media)

    resolved_solver = _resolve_cobra_solver(model, solver)

    solution = pfba(model)
    growth   = _get_growth(model, solution)

    return solution.fluxes, {
        "method":           "pfba_baseline",
        "solver":           resolved_solver,
        "predicted_growth": growth,
        "status":           solution.status,}


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Continuous (Meeson & Schwartz 2024)
# ─────────────────────────────────────────────────────────────────────────────

def run_continuous_meeson(model: cobra.Model, expr: pd.Series, growth_threshold: float | None = None, media: dict | None = None, solver: str = "auto", return_model: bool = False, media_ceiling: float | None = None,) -> tuple[pd.Series, dict]:
    """
    Continuous transcriptomics integration (Meeson & Schwartz, 2024).

    1. Normalise expression to [0, 1000].
    2. Set reaction bounds proportional to normalised expression capacity.
    3. Optional growth threshold correction: reopen most-limiting reactions
       iteratively until predicted growth >= growth_threshold.
    4. Run pFBA on constrained model.

    solver : str
        'auto' or 'glpk' (only supported values — see module docstring).
    return_model : bool
        If True, include the final constrained cobra.Model in the
        returned info dict under "model" — e.g. for downstream
        single-gene-deletion essentiality analysis on the exact same
        cell-line-specific constrained model this method produced,
        rather than duplicating the constraint-building logic
        elsewhere. Default False to keep existing callers' info dicts
        unchanged.
    media_ceiling : float, optional
        The media-constrained growth ceiling with ZERO expression
        constraint (i.e. what pfba_baseline achieves under the same
        media) — pass this in if you already have it (e.g.
        run_integration.py computes it once per model, before the
        cell-line loop) to get an explicit, self-documenting
        "threshold_unreachable" flag in the returned info dict,
        instead of the caller having to separately infer it from
        "were 100% of reactions reopened" pattern-matching. If
        growth_threshold exceeds this ceiling, the correction below
        is mathematically unable to succeed regardless of how many
        reactions get reopened — it will still run (falling back
        toward the fully unconstrained model, the same as before),
        but now the return value says so explicitly rather than
        leaving it implicit. Omit (default None) to skip this check
        — existing callers that don't pass it get identical behaviour
        to before, just without the new "threshold_unreachable" key.
    """
    model = model.copy()
    if media:
        _apply_media(model, media)

    resolved_solver = _resolve_cobra_solver(model, solver)

    max_expr = expr.max()
    if max_expr <= 0:
        raise ValueError(
            "All expression values are zero or negative. "
            "Check expression data loaded correctly."
        )
    expr_norm = (expr / max_expr) * 1000.0

    constrained: list[tuple[cobra.Reaction, float, float, float]] = []

    for rxn in model.reactions:
        capacity = apply_gpr_expression(rxn, expr_norm)
        if capacity is None:
            continue
        capacity = max(capacity, 0.0)
        orig_lb, orig_ub = rxn.lower_bound, rxn.upper_bound

        if rxn.lower_bound < 0:
            rxn.lower_bound = -capacity
        rxn.upper_bound = capacity
        constrained.append((rxn, capacity, orig_lb, orig_ub))

    n_constrained   = len(constrained)
    reopened_ids: list[str] = []

    if growth_threshold is not None and growth_threshold > 0:
        current_growth = model.slim_optimize()
        if current_growth is None or current_growth < growth_threshold:
            for rxn, capacity, orig_lb, orig_ub in sorted(
                constrained, key=lambda x: x[1]
            ):
                if current_growth is not None and \
                        current_growth >= growth_threshold:
                    break
                rxn.lower_bound = orig_lb
                rxn.upper_bound  = orig_ub
                reopened_ids.append(rxn.id)
                current_growth = model.slim_optimize()

    solution = pfba(model)
    growth   = _get_growth(model, solution)

    info = {
        "method":                "continuous_meeson",
        "solver":                resolved_solver,
        "genes_matched_to_model": len(expr),
        "reactions_constrained": n_constrained,
        "reactions_reopened":    len(reopened_ids),
        "reopened_reaction_ids": reopened_ids,
        "predicted_growth":      growth,
        "status":                solution.status,
    }
    if media_ceiling is not None and growth_threshold is not None:
        info["threshold_unreachable"] = growth_threshold > media_ceiling
    if return_model:
        info["model"] = model
    return solution.fluxes, info


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: GIMME via MEWpy
# ─────────────────────────────────────────────────────────────────────────────

def run_gimme(model: cobra.Model, expr: pd.Series, cutoff: float = 25.0, growth_frac: float = 0.9, parsimonious: bool = True, solver: str = "auto", media: dict | None = None) -> tuple[pd.Series, dict]:
    """
    GIMME: Gene Inactivity Moderated by Metabolism and Expression.
    (Becker & Palsson, 2008) via a corrected local fork (_gimme_fixed)
    of MEWpy 1.0.0's shipped GIMME().

    Minimises usage of lowly expressed reactions while maintaining
    a minimum growth requirement.

    NOTE ON THE FORK: MEWpy's own GIMME() calls sim.simulate() once,
    internally, to compute its own wild-type growth reference — that
    call routes through reframed's own solver registry
    (reframed.solvers), which is completely empty in this environment
    regardless of what's installed: the installed reframed release's
    __init__.py references .optlang_solver and .pulp_solver, but
    neither file actually exists in the package (confirmed by direct
    inspection), so every registration attempt silently fails via
    ImportError. _gimme_fixed replaces just that one call with an
    equivalent solve on the mewpy.solvers-based `solver` object —
    the same working codepath _imat_fixed already relies on — leaving
    the rest of GIMME's algorithm unchanged from MEWpy's own version.

    NOTE ON SOLVER: GLPK only (Gurobi and SCIP were deliberately
    removed from the solve path — see module docstring). GIMME is a
    pure LP (no binary variables), so GLPK is entirely adequate here
    even at genome scale.

    NOTE ON PARSIMONIOUS: GIMME's objective only penalises flux through
    *low*-expression reactions — it says nothing about magnitude for any
    other reaction, so the LP is degenerate over that unconstrained
    subset: many wildly different flux vectors can share the exact same
    objective value. Confirmed on Human1 at genome scale: with
    parsimonious=False, cross-cell-line "differential" reaction counts
    were inflated by alternate-optima noise (1941/2103 non-trivial
    reactions flagged, several with flux magnitudes >1000 in a network
    with +/-1000 bounds and no media constraints — a futile-cycling
    signature, not biology). Switching to parsimonious=True (added as a
    secondary objective term minimising total flux among GIMME-optimal
    solutions) dropped this to 269/372 reactions, sane magnitudes.
    Default True; set False only to reproduce vanilla GIMME exactly.

    Parameters
    ----------
    model : cobra.Model
    expr : pd.Series
        Output of normalise_to_model_ids().
    cutoff : float
        Percentile below which reactions are considered lowly expressed.
        Default 25.
    growth_frac : float
        Minimum growth as fraction of wild-type. Default 0.9.
    parsimonious : bool
        Break ties among GIMME-optimal solutions by minimum total flux.
        Default True — see note above.
    solver : str
        'auto' or 'glpk' (only supported values — see module docstring).
    media : dict, optional
    """
    try:
        from mewpy.omics.integration.gimme import ExpressionSet
        from mewpy.io import load_sbml_simulator
        import tempfile
        import os
    except ImportError:
        raise ImportError("MEWpy is required for GIMME. Run: pip install mewpy")

    resolved_solver = _resolve_and_activate_solver(solver)

    model_copy = model.copy()
    if media:
        _apply_media(model_copy, media)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cobra.io.write_sbml_model(model_copy, tmp_path)
        sim = load_sbml_simulator(tmp_path)
        # Reframed has its own separate solver registry from both
        # mewpy.solvers and the module-level optlang.Model patch above --
        # set explicitly rather than assuming it picked up either.
        sim.solver = "glpk"
    finally:
        os.unlink(tmp_path)

    biomass_id = _get_biomass_id(model)
    mewpy_biomass_id = "R_" + biomass_id

    sim.set_objective(mewpy_biomass_id)

    # Apply exchange bounds and cap biomass at COBRApy WT value
    env_cond = {}
    for rxn in model_copy.reactions:
        if len(rxn.metabolites) == 1:
            r_id = "R_" + rxn.id
            if r_id in sim.reactions:
                # Cap at 1000 -- see the infinite-bounds pass below for
                # why this matters for GLPK specifically.
                lb = max(rxn.lower_bound, -1000)
                ub = min(rxn.upper_bound,  1000)
                env_cond[r_id] = (lb, ub)

    # Internal reference computation via plain cobrapy — a separate
    # solver mechanism from the main GIMME solve (see
    # _resolve_cobra_solver's docstring). Both always resolve to glpk
    # now, so this is really just consistency with that mechanism.
    _resolve_cobra_solver(model_copy, "auto")
    wt_growth = float(pfba(model_copy).fluxes[biomass_id])
    env_cond[mewpy_biomass_id] = (0, wt_growth)
    sim.set_environmental_conditions(env_cond)

    # Cap any reaction with an infinite bound to prevent a native GLPK
    # crash. Mirrors the identical pass in run_imat (added there for
    # the big-M formulation's NaN risk) — confirmed necessary here too
    # after a real crash: glp_simplex segfaulted the whole Python
    # process (Fatal Python error: Aborted, not a catchable exception)
    # on a tiny toy model, inside MEWpy's own shipped GIMME() library
    # call specifically. GLPK's C library is far less defensive than
    # Gurobi's about literal inf bounds surviving the SBML round-trip
    # (write model_copy out, reload via load_sbml_simulator above) --
    # where Gurobi silently tolerates it, GLPK can abort outright
    # rather than error cleanly.
    import math
    for r_id in sim.reactions:
        lb, ub = sim.get_reaction_bounds(r_id)
        new_lb = lb
        new_ub = ub
        if math.isinf(lb):
            new_lb = -1000.0
        if math.isinf(ub):
            new_ub = 1000.0
        if new_lb != lb or new_ub != ub:
            sim.set_reaction_bounds(r_id, new_lb, new_ub)

    # Build ExpressionSet — MEWpy uses G_ prefix for gene IDs
    expr_set = ExpressionSet(
        identifiers=["G_" + g for g in expr.index.tolist()],
        conditions=["condition_1"],
        expression=expr.values.reshape(-1, 1).astype(float),)

    gimme_constraints = {mewpy_biomass_id: (growth_frac * wt_growth, wt_growth)}

    result = _gimme_fixed(
        model=sim,
        expr=expr_set,
        biomass=mewpy_biomass_id,
        condition=0,
        cutoff=cutoff,
        growth_frac=growth_frac,
        constraints=gimme_constraints,
        parsimonious=parsimonious,)

    if result is None:
        raise RuntimeError("GIMME returned None — solver may have failed.")

    fluxes = pd.Series(result.fluxes)
    growth = fluxes.get(mewpy_biomass_id)

    return fluxes, {
        "method":           "gimme",
        "solver":           resolved_solver,
        "cutoff":           cutoff,
        "growth_frac":      growth_frac,
        "parsimonious":     parsimonious,
        "predicted_growth": growth,
        "status":           getattr(result, "status", "optimal"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: iMAT via MEWpy
# ─────────────────────────────────────────────────────────────────────────────

def run_imat(model: cobra.Model, expr: pd.Series, lower_percentile: float = 25.0, upper_percentile: float = 75.0, epsilon: float = 0.001, min_growth_frac: float | None = 0.1, solver: str = "auto", time_limit: float | None = None, mip_gap: float | None = None, media: dict | None = None, parsimonious: bool = False,) -> tuple[pd.Series, dict]:
    """
    iMAT: Integrative Metabolic Analysis Tool.
    (Shlomi et al. 2008, Zur et al. 2010) via MEWpy 1.0.0

    Uses MILP with binary variables to maximise consistency between
    fluxes and gene expression data.

    NOTE ON SOLVER: GLPK only (Gurobi and SCIP were deliberately
    removed from the solve path — see module docstring). This is a
    genuine MILP (thousands of binary variables at genome scale) and
    GLPK's MIP solver is markedly weaker than Gurobi's or SCIP's on
    problems this size — expect noticeably longer solve times than
    the Gurobi runs this pipeline was originally developed against.
    `time_limit`/`mip_gap` below are the main lever for keeping that
    practical.

    NOTE ON SOLVE TIME: GLPK's branch-and-bound can take a long time
    to reach PROVEN optimality on a large MILP like this. Since
    iMAT's objective is a consistency-count heuristic (not something
    that needs an exact-optimality guarantee to be biologically
    meaningful), setting `time_limit` and/or `mip_gap` to accept a
    near-optimal solution early can turn an impractically long solve
    into a fast one with little to no meaningful loss in result
    quality.

    NOTE ON GROWTH: iMAT's objective only maximises agreement between
    fluxes and expression classification — it has no term for biomass.
    Machado & Herrgard (2014) found this exact behaviour causes iMAT (and
    other growth-agnostic methods) to predict zero growth under many
    conditions, and that adding a minimum-growth constraint improved
    predictions. `min_growth_frac` reproduces that fix: if set, biomass
    flux is constrained to >= min_growth_frac * (unconstrained pFBA
    growth), leaving the MILP free to otherwise choose the most
    expression-consistent flux distribution. Set to None to reproduce
    the literature-faithful, growth-agnostic formulation (may return
    ~0 growth).

    Parameters
    ----------
    model : cobra.Model
    expr : pd.Series
        Output of normalise_to_model_ids().
    lower_percentile : float
        Genes below this percentile are lowly expressed. Default 25.
    upper_percentile : float
        Genes above this percentile are highly expressed. Default 75.
    epsilon : float
        Minimum flux for a reaction to be considered active. Default 0.001.
    min_growth_frac : float or None
        Minimum growth as a fraction of unconstrained pFBA growth.
        Default 0.1. Set to None to disable (matches vanilla iMAT).
    time_limit : float, optional
        Stop after this many seconds and return the best solution
        found so far, even if not proven optimal. This pair was
        previously verified on Gurobi and SCIP; it has NOT yet been
        specifically verified on GLPK — if GLPK's MEWpy interface
        doesn't honour it the same way, iMAT may simply run to full
        (possibly very long) completion regardless of this setting.
        Worth confirming empirically before relying on it.
    mip_gap : float, optional
        Stop once the solution is within this fraction of the proven
        optimal bound (e.g. 0.01 = accept anything within 1%), rather
        than requiring an exact 0% gap. Same GLPK-verification caveat
        as time_limit above.
    solver : str
        'auto' or 'glpk' (only supported values — see module docstring).
    media : dict, optional
    parsimonious : bool
        Default False. See _imat_fixed()'s docstring for full detail
        — adds a secondary "minimize total flux" solve to reduce
        alternate-optima noise, same concern run_gimme's
        parsimonious=True already addresses, but note this one is
        still a MILP (not a cheap LP pass) and changes what iMAT's
        output means, so it's opt-in rather than a new default.
    """
    try:
        from mewpy.omics.integration.gimme import ExpressionSet
        from mewpy.io import load_sbml_simulator
        import tempfile
        import os
    except ImportError:
        raise ImportError("MEWpy is required for iMAT. Run: pip install mewpy")

    resolved_solver = _resolve_and_activate_solver(solver)

    model_copy = model.copy()
    if media:
        _apply_media(model_copy, media)

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cobra.io.write_sbml_model(model_copy, tmp_path)
        sim = load_sbml_simulator(tmp_path)
        # Reframed has its own separate solver registry from both
        # mewpy.solvers and the module-level optlang.Model patch above --
        # set explicitly rather than assuming it picked up either.
        sim.solver = "glpk"
    finally:
        os.unlink(tmp_path)

    biomass_id       = _get_biomass_id(model)
    mewpy_biomass_id = "R_" + biomass_id

    sim.set_objective(mewpy_biomass_id)

    # Unconstrained pFBA growth, used only as a scale reference for the
    # optional growth floor below (mirrors run_gimme's wt_growth).
    # Same independent solver resolution as run_gimme — see comment
    # there for why this doesn't need to match the caller's main
    # solver choice for the MILP solve itself.
    _resolve_cobra_solver(model_copy, "auto")
    wt_growth = float(pfba(model_copy).fluxes[biomass_id])

    # Apply exchange bounds and cap biomass at COBRApy WT value
    env_cond = {}
    for rxn in model_copy.reactions:
        if len(rxn.metabolites) == 1:
            r_id = "R_" + rxn.id
            if r_id in sim.reactions:
                # Cap at 1000 to prevent inf bounds causing NaN in iMAT big-M
                lb = max(rxn.lower_bound, -1000)
                ub = min(rxn.upper_bound,  1000)
                env_cond[r_id] = (lb, ub)

    sim.set_environmental_conditions(env_cond)

    # Cap only reactions with infinite bounds to prevent NaN in iMAT big-M
    # Only 11 reactions in Human1 have inf bounds — cap these specifically
    import math
    for r_id in sim.reactions:
        lb, ub = sim.get_reaction_bounds(r_id)
        new_lb = lb
        new_ub = ub
        if math.isinf(lb):
            new_lb = -1000.0
        if math.isinf(ub):
            new_ub = 1000.0
        if new_lb != lb or new_ub != ub:
            sim.set_reaction_bounds(r_id, new_lb, new_ub)

    # Compute reaction-level expression using GPR rules
    # then pass directly to iMAT as (low_coeffs, high_coeffs) tuple
    # This bypasses MEWpy's internal gene->reaction mapping which may fail
    # due to G_ prefix differences
    
    lower_thresh = float(expr.quantile(lower_percentile / 100.0))
    upper_thresh = float(expr.quantile(upper_percentile / 100.0))

    low_coeffs  = {}
    high_coeffs = {}

    for rxn in model_copy.reactions:
        r_id = "R_" + rxn.id
        if r_id not in sim.reactions:
            continue
        val = apply_gpr_expression(rxn, expr)
        if val is None:
            continue
        if val <= lower_thresh:
            low_coeffs[r_id]  = 1.0
        elif val >= upper_thresh:
            high_coeffs[r_id] = 1.0

    imat_constraints = {}
    if min_growth_frac is not None:
        imat_constraints[mewpy_biomass_id] = (min_growth_frac * wt_growth, wt_growth)

    # MEWpy's shipped iMAT() has a big-M sign bug that makes the "off"
    # state of the low-expression / reversible-high-expression binaries
    # non-vacuous — see the _imat_fixed() docstring at the top of this
    # file for the full derivation. Use the corrected local version
    # instead.
    from mewpy.solvers.solver import default_parameters, Parameter
    _params_touched = [Parameter.TIME_LIMIT, Parameter.MIP_REL_GAP,]
    # Track whether each key was PRESENT at all before this call, not
    # just its value — default_parameters.get(p) returning None is
    # ambiguous between "key absent" and "key explicitly set to None".
    # Restoring by always reassigning (even to None) would leave keys
    # in the dict that weren't there originally, silently changing
    # default_parameters' shape for every solver call afterwards in
    # the same process — confirmed via a real regression: this exact
    # mistake made 9 unrelated tests fail when run as part of the full
    # suite (passing individually), because TIME_LIMIT/MIP_REL_GAP
    # weren't in the dict before this feature was added.
    _was_present = {p: (p in default_parameters) for p in _params_touched}
    _orig_values = {p: default_parameters.get(p) for p in _params_touched}

    # time_limit/mip_gap — see docstring above re: not yet independently
    # verified on GLPK specifically (only confirmed on gurobi/scip,
    # which are no longer in the solve path — see module docstring).
    if time_limit is not None:
        default_parameters[Parameter.TIME_LIMIT] = time_limit
    if mip_gap is not None:
        default_parameters[Parameter.MIP_REL_GAP] = mip_gap
    try:
        result = _imat_fixed(
            model=sim,
            expr=(low_coeffs, high_coeffs),
            cutoff=(lower_percentile, upper_percentile),
            condition=0,
            epsilon=epsilon,
            build_model=False,
            constraints=imat_constraints,
            parsimonious=parsimonious,
        )
    finally:
        for _param in _params_touched:
            if _was_present[_param]:
                default_parameters[_param] = _orig_values[_param]
            else:
                default_parameters.pop(_param, None)

    if result is None:
        raise RuntimeError("iMAT returned None — solver may have failed.")

    fluxes = pd.Series(result.fluxes)
    growth = fluxes.get(mewpy_biomass_id)

    return fluxes, {
        "method":            "imat",
        "solver":            resolved_solver,
        "lower_percentile":  lower_percentile,
        "upper_percentile":  upper_percentile,
        "epsilon":           epsilon,
        "min_growth_frac":   min_growth_frac,
        "time_limit":        time_limit,
        "mip_gap":           mip_gap,
        "parsimonious":      parsimonious,
        "predicted_growth":  growth,
        "status":            getattr(result, "status", "optimal"),}


# ─────────────────────────────────────────────────────────────────────────────
# Master dispatcher — single entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_integration(model: cobra.Model, method: str = "continuous_meeson", expr: pd.Series | None = None, growth_threshold: float | None = None, media: dict | None = None, **kwargs) -> tuple[pd.Series, dict]:
    """
    Single entry point for all integration methods.

    Parameters
    ----------
    model : cobra.Model
    method : str
        "pfba_baseline", "continuous_meeson", "gimme", or "imat"
    expr : pd.Series or None
        Not required for "pfba_baseline".
    growth_threshold : float, optional
        For "continuous_meeson" only.
    media : dict, optional
    """
    method = method.lower().strip()

    if method == "pfba_baseline":
        return run_pfba_baseline(model, media=media, **kwargs)

    if expr is None:
        raise ValueError(
            f"expr is required for method '{method}'. "
            "Only 'pfba_baseline' runs without expression data."
        )

    if method == "continuous_meeson":
        return run_continuous_meeson(
            model, expr,
            growth_threshold=growth_threshold,
            media=media, **kwargs
        )

    if method == "gimme":
        return run_gimme(model, expr, media=media, **kwargs)

    if method == "imat":
        return run_imat(model, expr, media=media, **kwargs)

    raise ValueError(
        f"Unknown method: '{method}'. "
        "Choose from: 'pfba_baseline', 'continuous_meeson', 'gimme', 'imat'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _apply_media(model: cobra.Model, media: dict, strict: bool = True) -> None:
    """
    Apply media constraints to a model's boundary reactions.

    strict : bool
        If True (default), ALL boundary reactions are closed
        (lower_bound = 0, no uptake permitted) before applying `media`.
        This is what actually "defining a medium" means — without it,
        any boundary reaction not explicitly listed in `media` keeps
        its original (usually wide-open, +/-1000) bound, so the model
        can bypass every nutrient limit you thought you'd set via an
        unlisted alternative route. Set False only for incremental
        adjustments on top of bounds you've already constrained some
        other way.

        Uses model.boundary (cobrapy's own official classifier —
        exchanges AND demands AND sinks together), not model.exchanges
        (a narrower, compartment-naming-based subset). Confirmed
        necessary via a real leak on Recon3D: 95 SK_/DM_ reactions
        (e.g. SK_nad_c, SK_pmtcoa_c, SK_btn_c — all sitting in the
        cytosol, not extracellular) sat open at (-1000, 1000) even
        with a correct, fully-closed model.exchanges pass, letting the
        model manufacture key cofactors/precursors internally with no
        external nutrient required at all — inflating growth to ~370
        under a media dict that correctly restricted every real amino
        acid/vitamin exchange. model.exchanges simply doesn't include
        demand/sink reactions; model.boundary does.
    """
    if strict:
        for rxn in model.boundary:
            rxn.lower_bound = 0.0

    for rxn_id, rate in media.items():
        if rxn_id in model.reactions:
            model.reactions.get_by_id(rxn_id).lower_bound = -abs(float(rate))
        else:
            print(f"  Warning: '{rxn_id}' not in model — skipped.")


def _get_biomass_id(model: cobra.Model) -> str:
    for rxn in model.reactions:
        if rxn.objective_coefficient != 0:
            return rxn.id
    raise ValueError("No objective reaction found in model.")


def _get_growth(model: cobra.Model, solution: cobra.Solution) -> float | None:
    try:
        return float(solution.fluxes[_get_biomass_id(model)])
    except Exception:
        return None


def _check_glpk_available() -> None:
    """
    Check GLPK/optlang is actually usable. Raises a clear, actionable
    RuntimeError if not. GLPK is the only backend this pipeline uses
    now (see module docstring for why Gurobi/SCIP were removed).
    """
    try:
        import optlang  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "GLPK/optlang is not installed. Install: pip install optlang"
        )


def _resolve_and_activate_solver(solver: str) -> str:
    """
    Verify GLPK is usable, set it as MEWpy's active solver, and return
    'glpk' (kept as a function, and the `solver` parameter kept on the
    call sites below, so the call sites don't need touching if a
    different backend is ever reintroduced later — see module
    docstring). Any explicit request for something other than
    'auto'/'glpk'/'optlang' raises rather than silently overriding it.

    TWO independently-configured internal solver registries need
    setting to the GLPK-equivalent name:
      1. mewpy.solvers — used directly by both our own _imat_fixed()
         and _gimme_fixed() (their sole solve mechanism — see either
         function's docstring). GLPK's name here: 'optlang'.
      2. mewpy.simulation — read by some paths through the
         reframed-backed Simulator. GLPK's name here: 'glpk' (not
         'optlang'). Neither _imat_fixed() nor _gimme_fixed() actually
         exercises those paths (both avoid sim.simulate() entirely,
         specifically because reframed's OWN separate solver registry
         — reframed.solvers — is permanently empty in this environment:
         the installed reframed release's __init__.py references
         .optlang_solver and .pulp_solver, but neither file exists in
         the package at all, confirmed by direct inspection). Setting
         this registry is harmless defensive consistency, not load-
         bearing for anything this module currently does.
    Neither registry actually controls which interface *optlang
    itself* picks, though — that's the separate, more fundamental
    issue fixed by the module-level optlang.Model patch at the top of
    this file.
    """
    if solver not in ("auto", "glpk", "optlang"):
        raise ValueError(
            f"solver='{solver}' is not supported — this pipeline runs "
            f"GLPK only (Gurobi and SCIP were deliberately removed, "
            f"see module docstring). Use 'auto', 'glpk', or 'optlang'.")
    _check_glpk_available()

    import mewpy.solvers as _msolvers
    import mewpy.simulation as _msimulation

    _msolvers.set_default_solver("optlang")
    try:
        _msimulation.set_default_solver("glpk")
    except RuntimeError:
        pass  # e.g. name not recognised by this registry; registry 1 above is still correctly set

    return "glpk"