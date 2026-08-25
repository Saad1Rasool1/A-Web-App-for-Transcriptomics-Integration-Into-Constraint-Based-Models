"""
Single-function API for integration into Streamlit interface.

Only file needed to import for Streamlit. It wraps the entire pipeline
into one callable function with a simple, documented interface.

Usage for Streamlit app:
    from tools.pipeline import run_transcriptomics_integration

    # media=None (the default) automatically uses Human1's registered
    # default medium (DMEM) — no need to pass anything for standard use
    fluxes, info = run_transcriptomics_integration(
        model_path="path/to/Human1.xml",
        expression_path="path/to/OmicsExpression.csv",
        model_csv_path="path/to/Model.csv",
        gene_csv_path="path/to/Gene.csv",
        cell_line_name="CAOV3",
        method="continuous_meeson",
        growth_threshold=0.148,)"""
import pandas as pd
from pathlib import Path

from tools.model_loader import load_model, get_biomass_reaction_id
from tools.gene_mapping import normalise_to_model_ids
from tools.data_loaders import load_depmap, load_generic_csv, load_tcga
from tools.integrate_omics import run_integration
from tools.default_media import get_default_media


def run_transcriptomics_integration(
    model_path: str | Path,
    expression_path: str | Path,
    cell_line_name: str | None = None,
    model_csv_path: str | Path | None = None,
    gene_csv_path: str | Path | None = None,
    ensembl_mapping_path: str | Path | None = None,
    tcga_sample_barcode: str | None = None,
    tcga_sample_type: str = "01",
    method: str = "continuous_meeson",
    growth_threshold: float | None = None,
    media: dict | bool | None = None,
    id_format: str = "auto",
    parsimonious: bool | None = None,
    time_limit: float | None = None,
    mip_gap: float | None = None,
) -> tuple[pd.Series, dict]:
    """
    Run transcriptomics integration on a GEM model.

    This is the main entry point for the Streamlit interface.
    Handles model loading, gene ID normalisation, and FBA in one call.

    Parameters
    ----------
    model_path : str or Path
        Path to GEM file (.xml or .sbml).
        Supports Human1, Human2, Recon3D, or any SBML model.

    expression_path : str or Path
        Path to expression data file.
        Supported formats:
          - DepMap OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
          - TCGA/Xena Toil-recompute expression matrix (Ensembl-indexed)
          - Any CSV/TSV/Excel with genes as rows, samples as columns

    cell_line_name : str, optional
        Cell line name for DepMap data e.g. "CAOV3", "HEYA8".
        Required when using DepMap expression files.
        Not needed for generic CSV or TCGA files.

    model_csv_path : str or Path, optional
        Path to DepMap Model.csv.
        Required when using DepMap expression files.

    gene_csv_path : str or Path, optional
        Path to DepMap Gene.csv.
        For Human1/Human2 (Ensembl-format) models, only needed when the
        expression data is DepMap-format (Entrez-based); not needed for
        TCGA data, which is already Ensembl-indexed and matches directly.
        For Recon3D (Entrez-format), needed as the Ensembl<->Entrez
        bridge when using TCGA data — falls back to ensembl_mapping_path
        if not provided.

    ensembl_mapping_path : str or Path, optional
        BioMart TSV (columns: ensembl_gene_id, entrez_id and/or
        hgnc_symbol). Fallback bridge for TCGA data against Recon3D when
        gene_csv_path isn't available.

    tcga_sample_barcode : str, optional
        TCGA sample/patient barcode (or a prefix of one), e.g.
        "TCGA-25-1319". Required when using TCGA expression files.
        Not needed for DepMap or generic CSV files.

    tcga_sample_type : str
        TCGA sample type code, used to disambiguate when a barcode
        prefix matches multiple columns (tumour/normal/metastatic).
        Default "01" = primary solid tumour.

    method : str
        Integration method. One of:
          "pfba_baseline"     — no expression data, use as benchmark
          "continuous_meeson" — continuous bounds scaling
        Default: "continuous_meeson"

    growth_threshold : float, optional
        Minimum acceptable biomass flux (g/gDW/h).
        Convert from doubling time T (hours): threshold = ln(2) / T
        Only used by "continuous_meeson".
        If None, growth threshold correction is skipped.

    media : dict, False, or None
        Media conditions as {exchange_reaction_id: uptake_rate}.
        - None (default): use the registered default medium for this
          model if one exists (see tools/default_media.py) — e.g.
          Human1 defaults to DMEM. Falls back to no constraints if no
          default is registered for this model.
        - A dict: use exactly these media constraints, ignoring any
          registered default.
        - False: force no media constraints at all, even if a default
          is registered for this model.

    id_format : str
        Gene ID format of expression data.
        "auto" detects automatically (recommended).
        Options: "auto", "depmap", "ensembl", "entrez", "hgnc_symbol"

    parsimonious : bool or None
        Only relevant for method="gimme" or "imat" — ignored (and not
        forwarded) otherwise. None (default) leaves each method's own
        built-in default alone (GIMME defaults to True, iMAT to
        False — see run_gimme/run_imat's own docstrings for why they
        differ). Pass True/False explicitly to override either one.
        For iMAT specifically: without this, many reactions can land
        on the raw box bound (-1000/1000) rather than a real computed
        value, since iMAT's objective only rewards satisfying the
        expression classification and has no preference between an
        epsilon-sized flux and a 1000-sized one.
        WARNING, confirmed via a real genome-scale run: iMAT's
        parsimonious pass roughly DOUBLES the problem size (a new
        variable pair per reaction) and has been observed returning a
        false "INFEASIBLE" on GLPK at full genome scale after several
        minutes, even though the pre-parsimony solution is
        mathematically still a valid feasible point for the expanded
        problem (GLPK numerical fragility at scale, not a real
        contradiction — see the third bullet under time_limit/mip_gap
        below). ALWAYS pair parsimonious=True with a time_limit for
        iMAT until this is better understood; don't run it unbounded.

    time_limit, mip_gap : float or None
        Only relevant for method="imat" — ignored (and not forwarded)
        otherwise. See run_imat's docstring. Strongly recommended for
        any iMAT call on a genome-scale model without Gurobi/SCIP:
        GLPK's branch-and-bound has shown real numerical fragility at
        this scale (a confirmed segfault on malformed bounds, a
        confirmed false-"infeasible" result on the parsimonious pass
        specifically) — an unbounded search is a real risk, not just
        a slow one. time_limit=120, mip_gap=0.05 is a reasonable
        starting point (see run_integration.py's IMAT_TIME_LIMIT/
        IMAT_MIP_GAP for the values already in use elsewhere).

    Returns
    -------
    fluxes : pd.Series
        Flux values indexed by reaction ID (mmol/gDW/h).

    info : dict
        Run metadata including:
          method          — method used
          predicted_growth — biomass flux (g/gDW/h)
          status          — solver status ("optimal" if successful)
          genes_matched   — number of genes matched to model
          model_name      — name of model file used
          cell_line       — cell line name (if DepMap data)

    Raises
    ------
    ValueError
        If required parameters are missing or files not found.
    RuntimeError
        If solver returns non-optimal status.

    Examples
    --------
    # DepMap expression data with Human1
    fluxes, info = run_transcriptomics_integration(
        model_path="data/models/Human1.xml",
        expression_path="data/expression/OmicsExpression...csv",
        model_csv_path="data/expression/Model.csv",
        gene_csv_path="data/expression/Gene.csv",
        cell_line_name="CAOV3",
        method="continuous_meeson",
        growth_threshold=0.148,)

    # TCGA expression data with Human1 (no gene_csv_path needed — TCGA
    # is already Ensembl-indexed, matching Human1's internal format)
    fluxes, info = run_transcriptomics_integration(
        model_path="data/models/Human1.xml",
        expression_path="data/tcga/tcga_ov_toil_tpm.tsv.gz",
        tcga_sample_barcode="TCGA-25-1319",
        method="continuous_meeson",)

    # TCGA expression data with Recon3D (needs a bridge to Entrez —
    # reuses DepMap's Gene.csv if you have it, no separate download)
    fluxes, info = run_transcriptomics_integration(
        model_path="data/models/Recon3D.xml",
        expression_path="data/tcga/tcga_ov_toil_tpm.tsv.gz",
        tcga_sample_barcode="TCGA-25-1319",
        gene_csv_path="data/expression/Gene.csv",
        method="continuous_meeson",)

    # Generic CSV expression file
    fluxes, info = run_transcriptomics_integration(
        model_path="data/models/Human1.xml",
        expression_path="my_expression_data.csv",
        method="continuous_meeson",)

    # Baseline comparison (no expression data needed)
    fluxes, info = run_transcriptomics_integration(
        model_path="data/models/Human1.xml",
        expression_path=None,
        method="pfba_baseline",)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model file not found: {model_path}")

    # ── Resolve media: None -> per-model default, False -> force none ──────────
    if media is None:
        media = get_default_media(model_path.name)  # may still be None
    elif media is False:
        media = None

    # ── Load model ────────────────────────────────────────────────────────────
    model      = load_model(model_path)
    biomass_id = get_biomass_reaction_id(model)

    # ── Load expression data ──────────────────────────────────────────────────
    expr = None

    if method != "pfba_baseline":
        if expression_path is None:
            raise ValueError(
                f"expression_path is required for method '{method}'. "
                "Only 'pfba_baseline' can run without expression data.")

        expression_path = Path(expression_path)
        if not expression_path.exists():
            raise ValueError(
                f"Expression file not found: {expression_path}")

        # Detect which data source this is: DepMap, TCGA, or generic CSV
        is_depmap = (cell_line_name is not None and
                     model_csv_path is not None)
        is_tcga = tcga_sample_barcode is not None

        if is_depmap and is_tcga:
            raise ValueError(
                "Provide either cell_line_name (DepMap) or "
                "tcga_sample_barcode (TCGA), not both.")

        if is_depmap:
            if model_csv_path is None:
                raise ValueError(
                    "model_csv_path (Model.csv) is required "
                    "when cell_line_name is provided.")
            raw_expr = load_depmap(
                expression_path=expression_path,
                model_csv_path=model_csv_path,
                cell_line_name=cell_line_name,)
            detected_format = "depmap"
        elif is_tcga:
            raw_expr = load_tcga(
                expression_path=expression_path,
                sample_barcode=tcga_sample_barcode,
                sample_type=tcga_sample_type,)
            # Xena's Toil-recompute matrices are always Ensembl-indexed —
            # set explicitly rather than relying on auto-detection, since
            # a single-sample series gives detect_id_format very little
            # to work with.
            detected_format = "ensembl"
        else:
            # Generic CSV — one column of expression values
            raw_expr = load_generic_csv(expression_path)
            detected_format = id_format

        # Normalise gene IDs to match model format
        expr = normalise_to_model_ids(
            raw_expr,
            model,
            id_format=detected_format,
            gene_csv_path=gene_csv_path,
            ensembl_mapping_path=ensembl_mapping_path,)

        if len(expr) == 0:
            raise ValueError(
                "No genes matched between expression data and model. "
                "Check that the expression file contains recognisable "
                "gene IDs and that gene_csv_path is provided for "
                "Human1/Human2 models.")

    # ── Run integration ───────────────────────────────────────────────────────
    extra_kwargs = {}
    if method in ("gimme", "imat") and parsimonious is not None:
        extra_kwargs["parsimonious"] = parsimonious
    if method == "imat":
        if time_limit is not None:
            extra_kwargs["time_limit"] = time_limit
        if mip_gap is not None:
            extra_kwargs["mip_gap"] = mip_gap

    fluxes, run_info = run_integration(
        model=model,
        method=method,
        expr=expr,
        growth_threshold=growth_threshold,
        media=media,
        **extra_kwargs,)

    # ── Build info dict ───────────────────────────────────────────────────────
    info = {
        "method":           method,
        "model_name":       model_path.name,
        "biomass_reaction": biomass_id,
        "cell_line":        cell_line_name or "N/A",
        "tcga_sample":      tcga_sample_barcode or "N/A",
        "genes_matched":    len(expr) if expr is not None else 0,
        "predicted_growth": run_info.get("predicted_growth"),
        "status":           run_info.get("status"),
        **{k: v for k, v in run_info.items()
           if k not in {"method", "status", "predicted_growth"}},}

    return fluxes, info


def get_available_methods() -> list[str]:
    """Return list of available integration methods."""
    return ["pfba_baseline", "continuous_meeson", "gimme", "imat"]


def get_supported_models() -> list[str]:
    """Return list of supported GEM model names."""
    return ["Human1", "Human2", "Recon3D"]