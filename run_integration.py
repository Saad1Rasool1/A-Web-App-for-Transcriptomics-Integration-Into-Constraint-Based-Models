"""
End-to-end transcriptomics integration pipeline.

BEFORE RUNNING:
  1. Place Human1.xml, Human2.xml, Recon3D.xml in DATA/Models/
  2. Place OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv in DATA/Expression/
  3. Place Model.csv in DATA/Expression/
  4. Place Gene.csv in DATA/Expression/ (only required if running Human1/Human2)
  5. Activate venv311 and run: python run_integration.py

PATHS: all data/results paths are relative to this file by default, so
this script works unmodified on both machines (previously hardcoded to
one Windows username or the other). Override the whole DATA/ root with
the PROJECT2_DATA_DIR environment variable if you keep data somewhere
else on a given machine, e.g.:
    setx PROJECT2_DATA_DIR "D:\\some\\other\\place"
"""
import os
import sys
import math
import pandas as pd
from pathlib import Path

from tools.model_loader import load_model, get_biomass_reaction_id
from tools.gene_mapping import normalise_to_model_ids
from tools.data_loaders import load_depmap
from tools.integrate_omics import run_integration
from tools.default_media import get_default_media

# ── Subtype labels, for the cross-cell-line diagnostic below ─────────────────
# (Same grouping as the CELL_LINES comments: 59M/HEYA8 = LGSOC,
# CAOV3/COV318/OAW28 = HGSOC.)
CELL_LINE_SUBTYPE = {"59M": "LGSOC", "HEYA8": "LGSOC", "CAOV3": "HGSOC", "COV318": "HGSOC", "OAW28": "HGSOC",}

# =============================================================================
# USER INPUT SECTION — only edit values in this block
# =============================================================================

# ── Run mode ──────────────────────────────────────────────────────────────────
# "single"   → runs only the model named in ACTIVE_MODEL
# "multiple" → runs all models in MODELS that are found on disk
RUN_MODE     = "single"     # ← USER INPUT
ACTIVE_MODEL = "Human1"     # ← USER INPUT: only used when RUN_MODE = "single"

# ── Solver ────────────────────────────────────────────────────────────────────
# GLPK only. Gurobi and SCIP have been deliberately removed from the solve
# path entirely (see tools/integrate_omics.py's module docstring) — Gurobi
# because optlang was silently selecting it regardless of any solver name
# requested, whenever gurobipy was merely importable (licensed or not), and
# Streamlit deployment has no licence anyway; SCIP because it
# returned UNKNOWN status on iMAT and needs further investigation before
# being trusted again. 'auto' and 'glpk' both resolve to 'glpk' — kept as a
# variable (rather than removed outright) so this is one line to change if
# a different backend is ever reintroduced later.
SOLVER = "glpk"             # ← USER INPUT (only 'auto'/'glpk' supported)

# ── Media ─────────────────────────────────────────────────────────────────────
# None (default) = auto-lookup the correct media for whichever model is
# active, per model, via tools/default_media.py (DMEM_HIGH_GLUCOSE for
# Human1/Human2, RECON3D_DMEM_HIGH_GLUCOSE for Recon3D). This matters —
# unlike a single hardcoded dict, which was a real bug: it silently ran
# every model against Human1's MAR-prefixed media regardless of which
# model was actually active, closing every boundary reaction on Recon3D
# (none of Human1's IDs exist there) with nothing to reopen, producing
# an all-zero-flux "optimal" solution for every method/cell line.
# False = force NO media constraints for every model (fully open).
# A dict = force this exact media for every model, regardless of which
# one is active — only sensible if you specifically want to test one
# model's biology against another model's media.
MEDIA = None                # ← USER INPUT

# ── Data root ─────────────────────────────────────────────────────────────────
# Relative to this file by default; override via PROJECT2_DATA_DIR env var.
PROJECT_ROOT = Path(os.environ.get("PROJECT2_DATA_DIR", Path(__file__).resolve().parent))
DATA_DIR     = PROJECT_ROOT / "DATA"
RESULTS_DIR  = PROJECT_ROOT / "results"   # ← USER INPUT (change if you want results elsewhere)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {"Human1":  DATA_DIR / "Models" / "Human1.xml", "Human2":  DATA_DIR / "Models" / "Human2.xml", "Recon3D": DATA_DIR / "Models" / "Recon3D.xml",}
# Models whose internal gene IDs are Ensembl-format and therefore need
# GENE_CSV_PATH (the Entrez->Ensembl bridge) for DepMap-format expression.
_ENSEMBL_MODELS = {"Human1", "Human2"}

# ── DepMap data files ─────────────────────────────────────────────────────────
EXPRESSION_PATH = DATA_DIR / "Expression" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MODEL_CSV_PATH  = DATA_DIR / "Expression" / "Model.csv"
GENE_CSV_PATH   = DATA_DIR / "Expression" / "Gene.csv"

# ── Cell lines ────────────────────────────────────────────────────────────────
# Names must match exactly as they appear in Model.csv
CELL_LINES = [
    "59M",     # Low-grade serous OC (LGSOC)
    "HEYA8",   # Low-grade serous OC (LGSOC)
    "CAOV3",   # High-grade serous OC (HGSOC)
    "COV318",  # High-grade serous OC (HGSOC) 
    "OAW28",   # High-grade serous OC (HGSOC)
]

# ── Experimental doubling times (hours) ───────────────────────────────────────
# From Meeson & Schwartz (2024) Figure 3.
# Used to compute growth threshold for continuous_meeson correction.
# Set to None for any cell line without experimental data.
DOUBLING_TIMES_HOURS = {"59M":    37.66, "HEYA8":  4.68, "CAOV3":  4.68, "COV318": 38.70, "OAW28":  35.47,}

# ── Methods ───────────────────────────────────────────────────────────────────
# Run any subset — one, two, three, or all four, in any order. Just
# remove the ones you don't want; nothing else in this script assumes
# all four are present (the differential-flux diagnostic below already
# skips gracefully over any method whose result files don't exist). List so you remember which methods exist ("pfba_baseline","continuous_meeson","gimme","imat")
METHODS = ["pfba_baseline", "continuous_meeson", "gimme", "imat",]

# ── iMAT-specific overrides ─────────────────────────────────────────────────
# iMAT is a genuine MILP and GLPK's MIP solver is markedly weaker than
# Gurobi's or SCIP's at this problem size; these accept a near-optimal
# solution instead of solving to proven optimality. Ignored for the other
# three methods.
IMAT_TIME_LIMIT = 120.0   # seconds — raise if runs are hitting the limit unsolved
IMAT_MIP_GAP    = 0.05    # accept within 5% of optimal
# Note (see tools/integrate_omics.py's run_imat docstring): this pair was
# only previously verified on Gurobi/SCIP, not yet specifically confirmed
# to be honoured by GLPK's MEWpy interface. If iMAT runs are taking far
# longer than IMAT_TIME_LIMIT would suggest, that's worth checking first.

# =============================================================================
# End of user input — do not edit below this line
# =============================================================================


def summarise_subtype_differential_flux(model_results_dir, method, cell_lines):
    """
    Diagnostic, not a replacement for proper statistical validation: are
    the per-cell-line flux vectors for this method actually different
    between subtypes, independent of whether predicted growth differs?
    Growth alone can be identical across cell lines (see the module
    docstring in tools/integrate_omics.py and the note in the README
    about GIMME/iMAT deriving their growth bound from an
    expression-independent reference) while the underlying flux
    distribution still varies per cell line.

    Mirrors the differential-flux criterion used in Meeson & Schwartz
    (2024): a reaction counts as differentially regulated between
    LGSOC and HGSOC groups if the group means differ by >=10% (relative
    to the larger of the two group means) AND at least one group's mean
    |flux| is >= 0.5 mmol/gDW/h, OR the flux changes sign between groups.

    Returns (n_differential, output_path), or None if the per-cell-line
    flux CSVs for this method aren't all present yet.
    """
    frames = {}
    for cl in cell_lines:
        path = model_results_dir / f"{cl}_{method}_fluxes.csv"
        if not path.exists():
            return None
        frames[cl] = pd.read_csv(path, index_col=0)["flux_mmol_gDW_h"]
    flux_df = pd.DataFrame(frames)

    lg_cols = [c for c in flux_df.columns if CELL_LINE_SUBTYPE.get(c) == "LGSOC"]
    hg_cols = [c for c in flux_df.columns if CELL_LINE_SUBTYPE.get(c) == "HGSOC"]
    if not lg_cols or not hg_cols:
        return None

    lg_mean = flux_df[lg_cols].mean(axis=1)
    hg_mean = flux_df[hg_cols].mean(axis=1)

    denom = lg_mean.abs().combine(hg_mean.abs(), max)
    # denom.replace(0, nan) already turns "both means are zero" into NaN
    # directly (not inf), so no inf-handling is actually needed here --
    # the old use_inf_as_na context manager was vestigial, and deprecated
    # in current pandas.
    pct_change = ((lg_mean - hg_mean).abs() / denom.replace(0, float("nan"))).fillna(0.0)
    sign_change = (lg_mean * hg_mean) < 0
    magnitude_ok = (lg_mean.abs() >= 0.5) | (hg_mean.abs() >= 0.5)

    mask = (pct_change >= 0.10) & magnitude_ok | sign_change
    differential = flux_df[mask].copy()
    differential["LGSOC_mean"] = lg_mean[mask]
    differential["HGSOC_mean"] = hg_mean[mask]

    out_path = model_results_dir / f"{method}_subtype_differential_flux.csv"
    differential.to_csv(out_path)
    return len(differential), out_path


def run_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resolve which models to run ───────────────────────────────────────────
    if RUN_MODE == "single":
        if ACTIVE_MODEL not in MODELS:
            print(f"ERROR: '{ACTIVE_MODEL}' not in MODELS.")
            print(f"Options: {list(MODELS.keys())}")
            sys.exit(1)
        models_to_run = {ACTIVE_MODEL: MODELS[ACTIVE_MODEL]}
    elif RUN_MODE == "multiple":
        models_to_run = MODELS
    else:
        print(f"ERROR: RUN_MODE must be 'single' or 'multiple'.")
        sys.exit(1)

    print("=" * 60)
    print("Transcriptomics Integration Pipeline")
    print(f"Mode:    {RUN_MODE}")
    print(f"Models:  {list(models_to_run.keys())}")
    print(f"Methods: {METHODS}")
    print(f"Solver:  {SOLVER}")
    print(f"Data root: {DATA_DIR}")
    print("=" * 60)

    # ── Check required data files exist ───────────────────────────────────────
    required_files = [(EXPRESSION_PATH, "Expression.csv"), (MODEL_CSV_PATH, "Model.csv")]
    needs_gene_csv = any(m in _ENSEMBL_MODELS for m in models_to_run)
    if needs_gene_csv:
        required_files.append((GENE_CSV_PATH, "Gene.csv"))

    for path, label in required_files:
        if not path.exists():
            print(f"\nERROR: {label} not found at: {path}")
            sys.exit(1)

    # ── Compute growth thresholds from doubling times ─────────────────────────
    growth_thresholds = {
        name: math.log(2) / dt
        for name, dt in DOUBLING_TIMES_HOURS.items()
        if dt is not None}

    all_summaries = {}

    # ── Loop over models ──────────────────────────────────────────────────────
    for model_name, model_path in models_to_run.items():

        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        if not model_path.exists():
            print(f"  SKIPPED: file not found at {model_path}")
            print(f"  Download and place in DATA/Models/ to include.")
            continue

        model      = load_model(model_path)
        biomass_id = get_biomass_reaction_id(model)
        print(f"  Biomass reaction: {biomass_id}")

        # Resolve media for THIS model specifically (see MEDIA's own
        # comment above — None means auto-lookup per model, not "no
        # constraints"; that's media_for_model = False below).
        if MEDIA is None:
            media_for_model = get_default_media(model_path.name)
            print(f"  Media: auto-selected for {model_name} "
                  f"({'found' if media_for_model else 'NONE REGISTERED'})")
        elif MEDIA is False:
            media_for_model = None
            print(f"  Media: forced off (MEDIA=False)")
        else:
            media_for_model = MEDIA
            print(f"  Media: forced to the same dict for every model "
                  f"(MEDIA set to an explicit dict)")

        # Results subfolder per model
        model_results_dir = RESULTS_DIR / model_name
        model_results_dir.mkdir(exist_ok=True)

        # ── Media-constrained ceiling check ─────────────────────────────────
        # Only relevant to continuous_meeson's correction (see below) --
        # skipped entirely if it's not in METHODS, so removing it from
        # METHODS doesn't waste a solve or print a warning about a
        # correction that isn't even running.
        media_ceiling = None
        if "continuous_meeson" in METHODS:
            # continuous_meeson's growth-threshold correction can only
            # ever reopen reactions up to what this model+media combo
            # can achieve with ZERO expression constraint (i.e. the
            # pfba_baseline growth). If a cell line's growth_threshold
            # exceeds that ceiling, the correction is mathematically
            # unable to succeed -- it will silently reopen every
            # constrained reaction and fall back to the fully
            # unconstrained model, which is indistinguishable from
            # pfba_baseline in the growth column (but see the flux-level
            # diagnostic below for whether anything else differs).
            _, _ceiling_info = run_integration(
                model=model, method="pfba_baseline", media=media_for_model, solver=SOLVER,
            )
            media_ceiling = _ceiling_info.get("predicted_growth")
            print(f"  Media-constrained ceiling (no expression): "
                  f"{media_ceiling:.6f} g/gDW/h" if media_ceiling is not None
                  else "  Media-constrained ceiling: N/A")
            if media_ceiling is not None:
                for cl_name, dt in DOUBLING_TIMES_HOURS.items():
                    if dt is None:
                        continue
                    thr = math.log(2) / dt
                    if thr > media_ceiling:
                        print(f"  WARNING: growth threshold for {cl_name} "
                              f"({thr:.6f}) exceeds the media ceiling "
                              f"({media_ceiling:.6f}) for {model_name} -- "
                              f"continuous_meeson's correction cannot succeed "
                              f"for this cell line and will fully unconstrain "
                              f"the model instead.")

        model_summary_rows = []

        # ── Loop over cell lines ──────────────────────────────────────────────
        for cell_name in CELL_LINES:
            print(f"\n{'─' * 50}")
            print(f"  Cell line: {cell_name}  |  Model: {model_name}")
            print(f"{'─' * 50}")

            # Load raw expression
            try:
                raw_expr = load_depmap(expression_path=EXPRESSION_PATH, model_csv_path=MODEL_CSV_PATH, cell_line_name=cell_name,)
            except ValueError as e:
                print(f"  SKIPPED: {e}")
                continue

            # Normalise gene IDs to match model format
            print("  Normalising gene IDs...")
            expr = normalise_to_model_ids(raw_expr, model, id_format="depmap", gene_csv_path=GENE_CSV_PATH if GENE_CSV_PATH.exists() else None,)

            threshold = growth_thresholds.get(cell_name)
            if threshold:
                print(f"  Growth threshold: {threshold:.6f} g/gDW/h "
                      f"(doubling time: {DOUBLING_TIMES_HOURS[cell_name]}h)")
            else:
                print("  No growth threshold — correction skipped.")

            row = {"cell_line": cell_name}

            # ── Loop over methods ─────────────────────────────────────────────
            for method in METHODS:
                print(f"\n  [{method}]")
                try:
                    extra_kwargs = {"solver": SOLVER}
                    if method == "imat":
                        extra_kwargs["time_limit"] = IMAT_TIME_LIMIT
                        extra_kwargs["mip_gap"]    = IMAT_MIP_GAP
                    if method == "continuous_meeson":
                        extra_kwargs["media_ceiling"] = media_ceiling

                    fluxes, info = run_integration(
                        model=model,
                        method=method,
                        expr=expr if method != "pfba_baseline" else None,
                        growth_threshold=(
                            threshold if method == "continuous_meeson"
                            else None
                        ),
                        media=media_for_model,
                        **extra_kwargs,)

                    growth = info.get("predicted_growth")
                    print(f"    Solver:           {info.get('solver')}")
                    print(f"    Status:           {info.get('status')}")
                    if growth is not None:
                        print(f"    Predicted growth: {growth:.6f} g/gDW/h")
                    else:
                        print(f"    Predicted growth: N/A")

                    if method == "continuous_meeson":
                        print(f"    Reactions constrained: "
                              f"{info.get('reactions_constrained')}")
                        print(f"    Reactions reopened:    "
                              f"{info.get('reactions_reopened')}")
                        if info.get("threshold_unreachable"):
                            print(f"    NOTE: growth threshold was "
                                  f"unreachable under this media -- "
                                  f"correction fell back toward the "
                                  f"fully unconstrained model (see the "
                                  f"per-model warning above).")
                        reopened_ids = info.get("reopened_reaction_ids")
                        if reopened_ids:
                            reopened_path = (model_results_dir /
                                f"{cell_name}_continuous_meeson_reopened.txt")
                            reopened_path.write_text("\n".join(reopened_ids))
                            print(f"    Reopened reaction IDs saved: "
                                  f"{reopened_path}")

                    if method == "gimme":
                        print(f"    Cutoff percentile: {info.get('cutoff')}")
                        print(f"    Growth fraction:   {info.get('growth_frac')}")

                    if method == "imat":
                        print(f"    Lower percentile: {info.get('lower_percentile')}")
                        print(f"    Upper percentile: {info.get('upper_percentile')}")
                        print(f"    Epsilon:          {info.get('epsilon')}")

                    row[method] = growth

                    out = model_results_dir / f"{cell_name}_{method}_fluxes.csv"
                    fluxes.to_csv(out, header=["flux_mmol_gDW_h"])
                    print(f"    Saved: {out}")

                except Exception as e:
                    import traceback
                    print(f"    ERROR: {e}")
                    traceback.print_exc()
                    row[method] = None

            model_summary_rows.append(row)

        # ── Per-model summary ─────────────────────────────────────────────────
        if model_summary_rows:
            summary = pd.DataFrame(model_summary_rows).set_index("cell_line")
            print(f"\n{model_name} — predicted biomass flux (g/gDW/h)")
            print(summary.to_string())
            summary_path = model_results_dir / "growth_summary.csv"
            summary.to_csv(summary_path)
            print(f"Summary saved: {summary_path}")
            all_summaries[model_name] = summary

            # ── Flux-level cross-cell-line diagnostic ─────────────────────
            # See summarise_subtype_differential_flux()'s docstring: this
            # is what actually shows whether transcriptomics integration
            # changed anything, independent of whether the growth column
            # (above) happened to converge to the same ceiling.
            print(f"\n{model_name} — subtype differential flux "
                  f"(LGSOC vs HGSOC, per method)")
            for method in METHODS:
                if method == "pfba_baseline":
                    continue  # no expression data used; nothing to diff
                result = summarise_subtype_differential_flux(model_results_dir, method, CELL_LINES)
                if result is None:
                    print(f"  [{method}] SKIPPED (flux files missing)")
                    continue
                n_diff, out_path = result
                print(f"  [{method}] {n_diff} reactions differ >=10% "
                      f"between subtypes -- {out_path}")

    # ── Cross-model summary ───────────────────────────────────────────────────
    if len(all_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("Cross-model comparison — continuous_meeson (g/gDW/h)")
        print("=" * 60)
        cross = pd.DataFrame({
            name: df.get("continuous_meeson")
            for name, df in all_summaries.items()})
        print(cross.to_string())
        cross_path = RESULTS_DIR / "cross_model_growth_summary.csv"
        cross.to_csv(cross_path)
        print(f"Cross-model summary saved: {cross_path}")

    print("\nDone.")


if __name__ == "__main__":
    run_pipeline()