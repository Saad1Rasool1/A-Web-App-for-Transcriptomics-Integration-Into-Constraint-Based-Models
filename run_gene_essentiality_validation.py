"""
Gene-essentiality validation: does the pipeline's predicted gene essentiality (in silico single-gene knockout, on each cell line's continuous_meeson-constrained model) agree with real experimental CRISPR-Cas9 gene dependency data?

This mirrors Meeson & Schwartz (2024)'s own validation approach for these exact five cell lines: knock out each gene one at a time, record the ratio of post-knockout growth to the constrained model's own optimal growth, and correlate that ratio against DepMap's CRISPR dependency score for the same gene in the same cell line. 
A gene with dependency score near 1.0 is one CCLE cells genuinely can't grow without (real experimental essentiality); if the model's knockout growth ratio is low for the same gene, that is independent, orthogonal evidence the constrained model captures real biology.

Runs on continuous_meeson's constrained model specifically (via run_continuous_meeson's return_model=True), since that's the method whose growth-threshold-corrected bounds are the closest analogue to Meeson & Schwartz's own approach.

BEFORE RUNNING:
  1. Everything run_integration.py already needs (Human1.xml, DepMap expression/Model/Gene CSVs).
  2. CRISPR_gene_dependency.csv from the same DepMap release (https://depmap.org/portal/download/all/, DepMap Public 22Q2 to match Meeson & Schwartz — a different release will still work, just won't be a like-for-like comparison against their published numbers).

Usage: python run_gene_essentiality_validation.py"""
import os
import re
from pathlib import Path

import pandas as pd
from cobra.flux_analysis import single_gene_deletion

from tools.model_loader import load_model
from tools.gene_mapping import normalise_to_model_ids
from tools.data_loaders import load_depmap
from tools.integrate_omics import run_continuous_meeson
from tools.dmem_media import DMEM_HIGH_GLUCOSE

# ── USER INPUT ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.environ.get("PROJECT2_DATA_DIR", Path(__file__).resolve().parent))
DATA_DIR = PROJECT_ROOT / "DATA"

MODEL_PATH        = DATA_DIR / "Models" / "Human1.xml"
EXPRESSION_PATH   = DATA_DIR / "Expression" / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MODEL_CSV_PATH     = DATA_DIR / "Expression" / "Model.csv"
GENE_CSV_PATH      = DATA_DIR / "Expression" / "Gene.csv"
DEPENDENCY_PATH    = DATA_DIR / "Expression" / "CRISPRGeneDependency.csv"  # ← set if elsewhere
MEDIA              = DMEM_HIGH_GLUCOSE

CELL_LINE_SUBTYPE = { "59M": "LGSOC", "HEYA8": "LGSOC", "CAOV3": "HGSOC", "COV318": "HGSOC", "OAW28": "HGSOC",}
DOUBLING_TIMES_HOURS = {"59M": 37.66, "HEYA8": 4.68, "CAOV3": 4.68, "COV318": 38.70, "OAW28": 35.47,}

RESULTS_DIR = PROJECT_ROOT / "results" / "gene_essentiality_validation"
# ─────────────────────────────────────────────────────────────────────────


def _entrez_from_column(col: str) -> str | None:
    """DepMap columns are named 'GENE_SYMBOL (ENTREZ_ID)' — extract the
    Entrez ID. Returns None for columns that don't match (e.g. a stray
    non-gene column)."""
    m = re.search(r"\((\d+)\)\s*$", col)
    return m.group(1) if m else None


def build_ensembl_to_entrez(gene_csv_path: Path) -> dict[str, str]:
    """Human1's genes are Ensembl IDs; DepMap's dependency file is
    keyed by Entrez. Build the Ensembl -> Entrez bridge from the same
    Gene.csv already used elsewhere in this pipeline for the reverse
    direction, so both sides of the correlation are self-consistent."""
    gene_df = pd.read_csv(gene_csv_path, low_memory=False)
    gene_df = gene_df.dropna(subset=["entrez_id", "ensembl_gene_id"])
    return dict(zip(gene_df["ensembl_gene_id"].astype(str), gene_df["entrez_id"].astype(int).astype(str),))


def find_model_id_for_dependency_row(model_csv_path: Path, cell_line_name: str) -> str:
    """Same lookup load_depmap uses internally, exposed here since the
    dependency file is keyed by ModelID too, and we need it directly
    (not via load_depmap, which is for expression, not dependency)."""
    model_df = pd.read_csv(model_csv_path, low_memory=False)
    mask = (model_df["CellLineName"].str.upper().str.strip() == cell_line_name.upper().strip()) | (model_df["StrippedCellLineName"].str.upper().str.strip() == cell_line_name.upper().strip())
    matches = model_df[mask]
    if len(matches) == 0:
        raise ValueError(f"Cell line '{cell_line_name}' not found in Model.csv.")
    return matches.iloc[0]["ModelID"]


def main():
    try:
        from scipy.stats import pearsonr
    except ImportError:
        raise ImportError(
            "scipy is required for the correlation step. "
            "Run: pip install scipy --break-system-packages")

    for path, label in [
        (MODEL_PATH, "Human1.xml"), (EXPRESSION_PATH, "Expression CSV"),
        (MODEL_CSV_PATH, "Model.csv"), (GENE_CSV_PATH, "Gene.csv"),
        (DEPENDENCY_PATH, "CRISPRGeneDependency.csv"),]:
        if not path.exists():
            print(f"ERROR: {label} not found at: {path}")
            return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dependency data (this file can be large)...")
    dep_df = pd.read_csv(DEPENDENCY_PATH, index_col=0, low_memory=False)
    entrez_by_column = {col: _entrez_from_column(col) for col in dep_df.columns}

    ensembl_to_entrez = build_ensembl_to_entrez(GENE_CSV_PATH)
    # Reverse lookup: entrez -> the dependency column name that has it
    entrez_to_dep_column = {v: k for k, v in entrez_by_column.items() if v is not None}

    model = load_model(MODEL_PATH)

    per_cell_line_results = {}   # cell_line -> DataFrame(gene, ratio, dependency, subtype)
    growth_thresholds = {name: __import__("math").log(2) / dt
                         for name, dt in DOUBLING_TIMES_HOURS.items()}

    for cell_name in CELL_LINE_SUBTYPE:
        print(f"\n{'─' * 50}")
        print(f"Cell line: {cell_name}")
        print(f"{'─' * 50}")

        raw_expr = load_depmap(expression_path=EXPRESSION_PATH, model_csv_path=MODEL_CSV_PATH, cell_line_name=cell_name,)
        expr = normalise_to_model_ids(raw_expr, model, id_format="depmap", gene_csv_path=GENE_CSV_PATH,)

        _, info = run_continuous_meeson(model, expr, growth_threshold=growth_thresholds[cell_name], media=MEDIA, solver="auto", return_model=True,)
        constrained_model = info["model"]
        baseline_growth = constrained_model.slim_optimize()
        print(f"  Constrained baseline growth: {baseline_growth:.6f} g/gDW/h")

        if baseline_growth is None or baseline_growth <= 0:
            print(f"  SKIPPED: baseline growth is zero/None for {cell_name} — "
                  f"knockout ratios would be undefined.")
            continue

        print("  Running single-gene-deletion (this can take a while — "
              f"{len(constrained_model.genes)} genes)...")
        deletion_results = single_gene_deletion(constrained_model)
        # deletion_results index: frozenset of gene IDs (one per row for
        # single deletions); .ids column has the same as a set/tuple
        deletion_results = deletion_results.reset_index(drop=True)

        model_id = find_model_id_for_dependency_row(MODEL_CSV_PATH, cell_name)
        if model_id not in dep_df.index:
            print(f"  SKIPPED: {model_id} ({cell_name}) not present in "
                  f"CRISPRGeneDependency.csv.")
            continue
        dep_row = dep_df.loc[model_id]

        rows = []
        for _, row in deletion_results.iterrows():
            gene_ids = row["ids"]
            if not gene_ids or len(gene_ids) != 1:
                continue  # skip anything not a clean single-gene row
            ensembl_id = next(iter(gene_ids))
            entrez_id = ensembl_to_entrez.get(ensembl_id)
            if entrez_id is None:
                continue
            dep_col = entrez_to_dep_column.get(entrez_id)
            if dep_col is None or dep_col not in dep_row.index:
                continue
            dependency_score = dep_row[dep_col]
            if pd.isna(dependency_score):
                continue

            ratio = row["growth"] / baseline_growth if baseline_growth else None
            rows.append({
                "ensembl_id": ensembl_id,
                "entrez_id": entrez_id,
                "growth_ratio": ratio,
                "dependency_score": dependency_score,})

        result_df = pd.DataFrame(rows)
        if len(result_df) < 3:
            print(f"  SKIPPED: only {len(result_df)} genes matched between "
                  f"model knockouts and dependency data — too few to "
                  f"correlate meaningfully.")
            continue

        result_df.to_csv(RESULTS_DIR / f"{cell_name}_gene_essentiality.csv", index=False)

        r_all, p_all = pearsonr(result_df["growth_ratio"], result_df["dependency_score"])
        print(f"  All {len(result_df)} matched genes: "
              f"Pearson r={r_all:.4f}, p={p_all:.4g}")

        # A strict < 1.0 cutoff is too naive: "no effect" knockouts can
        # land at e.g. 0.999999999997 rather than exactly 1.0 (solver
        # floating-point precision, not a real effect), which would
        # wrongly count as "essential" under a bare < 1.0 comparison.
        # Confirmed via a real case: COV318's growth_ratio distribution
        # is a clean bimodal split (~104 genes near 0, ~2676 genes at
        # essentially exactly 1.0) once actually inspected — but a
        # naive < 1.0 filter put ALL 2780 in the "essential" bucket,
        # because the ~2676 "no effect" genes were a tiny epsilon below
        # 1.0 rather than exactly at it. 0.99 is a reasonable, standard
        # essentiality cutoff that comfortably clears this precision
        # noise without excluding any real partial-effect genes near
        # the boundary.
        essential_subset = result_df[result_df["growth_ratio"] < 0.99]
        if len(essential_subset) >= 3:
            r_ess, p_ess = pearsonr(essential_subset["growth_ratio"], essential_subset["dependency_score"])
            print(f"  Predicted-essential subset (growth_ratio<1.0, "
                  f"n={len(essential_subset)}): Pearson r={r_ess:.4f}, p={p_ess:.4g}")
        else:
            r_ess, p_ess = None, None
            print("  Predicted-essential subset too small to correlate.")

        per_cell_line_results[cell_name] = {
            "n_genes": len(result_df),
            "r_all": r_all, "p_all": p_all,
            "r_essential_subset": r_ess, "p_essential_subset": p_ess,}

    # ── Summary ────────────────────────────────────────────────────────────
    if per_cell_line_results:
        summary = pd.DataFrame(per_cell_line_results).T
        print(f"\n{'=' * 60}")
        print("Gene essentiality validation summary")
        print("=" * 60)
        print(summary.to_string())
        summary_path = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSummary saved: {summary_path}")
    else:
        print("\nNo cell lines produced a usable correlation.")


if __name__ == "__main__":
    main()