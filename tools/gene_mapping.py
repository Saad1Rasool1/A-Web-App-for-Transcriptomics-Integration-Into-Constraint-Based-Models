"""
Gene identifier normalisation across different transcriptomics data sources.

Sources and their gene ID formats:
  DepMap/CCLE  → "TP53 (7157)"        (HGNC symbol + Entrez ID in brackets)
  TCGA         → "ENSG00000141510.11"  (Ensembl ID with version suffix)
  ArrayExpress → varies by platform
  Generic CSV  → assumes HGNC symbols

Human GEMs (Human1, Human2) use Ensembl IDs internally.
Recon3D uses Entrez + isoform suffix e.g. '7157_AT1'.
This module normalises any format appropriately per model.
"""
import re
import pandas as pd
from pathlib import Path


def detect_id_format(gene_ids: pd.Index) -> str:
    """
    Detect gene identifier format from an expression index.
    Returns one of: "depmap", "ensembl", "entrez", "hgnc_symbol"
    """
    sample = [str(g) for g in gene_ids[:50]]
    if any(re.search(r'\(\d+\)$', g) for g in sample):
        return "depmap"
    if any(g.startswith("ENSG") for g in sample):
        return "ensembl"
    if all(re.match(r'^\d+$', g) for g in sample):
        return "entrez"
    return "hgnc_symbol"


def detect_model_gene_format(model) -> str:
    """
    Detect what gene ID format the model uses internally.
    Human1/Human2: Ensembl IDs e.g. 'ENSG00000000419'
    Recon3D:       Entrez + isoform suffix e.g. '7157_AT1'
    Plain entrez:  purely numeric e.g. '7157'
    HGNC symbols:  e.g. 'TP53'
    """
    sample = [g.id for g in list(model.genes)[:20]]
    if any(str(g).startswith("ENSG") for g in sample):
        return "ensembl"
    if any(re.match(r'^\d+_AT\d+$', str(g)) for g in sample):
        return "recon3d"
    if all(re.match(r'^\d+$', str(g)) for g in sample):
        return "entrez"
    return "hgnc_symbol"


def normalise_depmap_ids(gene_ids: pd.Index) -> pd.Index:
    """
    Strip Entrez IDs from DepMap column names.
    "TP53 (7157)" -> "TP53"
    """
    return pd.Index([re.sub(r'\s*\(\d+\)\s*$', '', str(g)).strip() for g in gene_ids])


def extract_entrez_from_depmap(gene_ids: pd.Index) -> pd.Index:
    """
    Extract Entrez IDs from DepMap column names.
    "TP53 (7157)" -> "7157"
    """
    result = []
    for g in gene_ids:
        match = re.search(r'\((\d+)\)$', str(g))
        result.append(match.group(1) if match else str(g))
    return pd.Index(result)


def match_depmap_to_recon3d(expr: pd.Series, model_gene_ids: set) -> pd.Series:
    """
    Match DepMap Entrez IDs to Recon3D gene IDs (format: '7157_AT1').

    For each Recon3D gene, strips the _AT suffix to get the base
    Entrez number, then looks for that number in the expression data.
    Where multiple isoforms exist (7157_AT1, 7157_AT2), they all
    receive the same expression value from the single DepMap entry.
    """
    matched = {}
    for gene_id in model_gene_ids:
        base = re.sub(r'_AT\d+$', '', str(gene_id))
        if base in expr.index:
            matched[gene_id] = expr[base]

    if not matched:
        return pd.Series(dtype=float)

    return pd.Series(matched)


def build_entrez_to_ensembl(gene_csv_path: str | Path) -> dict:
    """
    Build a mapping from Entrez ID (string) to Ensembl gene ID
    using the Gene.csv file downloaded from DepMap.

    Gene.csv columns used: entrez_id, ensembl_gene_id

    Returns dict: {'7105': 'ENSG00000000419', ...}
    """
    df = pd.read_csv(gene_csv_path, low_memory=False)

    entrez_col  = None
    ensembl_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "entrez" in col_lower and entrez_col is None:
            entrez_col = col
        if "ensembl_gene_id" in col_lower and ensembl_col is None:
            ensembl_col = col

    if entrez_col is None or ensembl_col is None:
        print(f"  Available Gene.csv columns: {df.columns.tolist()}")
        raise ValueError("Could not find entrez and ensembl columns in Gene.csv.")

    mapping = {}
    for _, row in df.iterrows():
        entrez  = row[entrez_col]
        ensembl = str(row[ensembl_col]).strip()

        if pd.isna(entrez) or pd.isna(row[ensembl_col]):
            continue
        if ensembl == "nan" or ensembl == "":
            continue

        # Convert float (1.0) to clean integer string ("1")
        # so it matches DepMap Entrez format ("7105" not "7105.0")
        try:
            entrez_str = str(int(float(entrez)))
        except (ValueError, TypeError):
            continue

        mapping[entrez_str] = ensembl

    print(f"  Built Entrez→Ensembl mapping: {len(mapping)} entries")
    sample = list(mapping.items())[:3]
    print(f"  Sample: {sample}")
    return mapping


def build_ensembl_to_entrez(mapping_path: str | Path) -> dict:
    """
    Build a mapping from Ensembl gene ID to Entrez ID (string) using a
    BioMart TSV export (columns: ensembl_gene_id, entrez_id or similar).

    This is the fallback bridge for Ensembl -> Entrez when gene_csv_path
    (DepMap's Gene.csv) isn't available. If it is available, prefer
    inverting build_entrez_to_ensembl() instead — no separate download
    needed.

    Returns dict: {'ENSG00000141510': '7157', ...}
    """
    df = pd.read_csv(mapping_path, sep="\t")

    ensembl_col = None
    entrez_col  = None
    for col in df.columns:
        col_lower = col.lower()
        if "ensembl_gene_id" in col_lower and ensembl_col is None:
            ensembl_col = col
        if "entrez" in col_lower and entrez_col is None:
            entrez_col = col

    if ensembl_col is None or entrez_col is None:
        print(f"  Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find ensembl_gene_id and entrez columns in the "
                         "mapping file. Expected a BioMart export with those columns.")

    mapping = {}
    for _, row in df.iterrows():
        ensembl = str(row[ensembl_col]).strip().split(".")[0]
        entrez  = row[entrez_col]
        if pd.isna(entrez) or ensembl in ("", "nan"):
            continue
        try:
            entrez_str = str(int(float(entrez)))
        except (ValueError, TypeError):
            continue
        mapping[ensembl] = entrez_str

    print(f"  Built Ensembl→Entrez mapping: {len(mapping)} entries")
    return mapping


def normalise_ensembl_ids(gene_ids: pd.Index, mapping_path: str | Path) -> pd.Index:
    """
    Convert Ensembl gene IDs to HGNC symbols using a BioMart TSV.
    Only needed when the target model itself uses HGNC symbols
    internally (none of Human1/Human2/Recon3D do — see
    normalise_to_model_ids for the routes actually used).
    """
    mapping_df = pd.read_csv(mapping_path, sep="\t")
    mapping = mapping_df.set_index("ensembl_gene_id")["hgnc_symbol"].to_dict()
    stripped = [str(g).split(".")[0] for g in gene_ids]
    return pd.Index([mapping.get(g, g) for g in stripped])


def _get_ensembl_to_entrez_map(gene_csv_path: str | Path | None, ensembl_mapping_path: str | Path | None,) -> dict:
    """
    Return an Ensembl -> Entrez dict, preferring gene_csv_path (DepMap's
    Gene.csv — already required for Human1/Human2 support, so this needs
    no extra download) and falling back to a dedicated BioMart TSV.
    """
    if gene_csv_path is not None:
        entrez_to_ensembl = build_entrez_to_ensembl(gene_csv_path)
        return {v: k for k, v in entrez_to_ensembl.items()}
    if ensembl_mapping_path is not None:
        return build_ensembl_to_entrez(ensembl_mapping_path)
    raise ValueError("Recon3D/Entrez-format models need an Ensembl -> Entrez bridge "
                     "for Ensembl-format input (e.g. TCGA). Provide either "
                     "gene_csv_path (DepMap's Gene.csv — no extra download needed) "
                     "or ensembl_mapping_path (a BioMart TSV with columns "
                     "ensembl_gene_id, entrez_id).")


def normalise_to_model_ids(expr: pd.Series, model, id_format: str = "auto", ensembl_mapping_path: str | Path | None = None, gene_csv_path: str | Path | None = None) -> pd.Series:
    """
    Normalise gene IDs to match the model's internal format,
    then filter to genes present in the model.

    Handles all combinations automatically:
      DepMap  → Human1/Human2 (Ensembl): Entrez extracted, mapped to Ensembl
      DepMap  → Recon3D (_AT format):    Entrez extracted, matched to _AT IDs
      DepMap  → plain entrez model:      Entrez extracted directly
      DepMap  → HGNC model:              HGNC symbol stripped from brackets
      Ensembl (e.g. TCGA) → Human1/Human2: version stripped, matched directly
      Ensembl (e.g. TCGA) → Recon3D:       mapped to Entrez, then _AT-matched
      Ensembl (e.g. TCGA) → entrez model:  mapped to Entrez directly
      Ensembl (e.g. TCGA) → HGNC model:    requires ensembl_mapping_path

    Parameters
    ----------
    expr : pd.Series
        Gene expression values, any gene ID format as index.
    model : cobra.Model
    id_format : str
        "auto" detects automatically, or specify:
        "depmap", "ensembl", "entrez", "hgnc_symbol"
    ensembl_mapping_path : str or Path, optional
        BioMart TSV (columns: ensembl_gene_id, entrez_id and/or
        hgnc_symbol). Only needed for Ensembl-format input (e.g. TCGA)
        when gene_csv_path isn't available, or when the model itself
        uses HGNC symbols internally.
    gene_csv_path : str or Path, optional
        Path to Gene.csv from DepMap. Required when expression is DepMap
        format and model uses Ensembl IDs (Human1, Human2). Also serves
        as the preferred Ensembl<->Entrez bridge for Ensembl-format
        input (e.g. TCGA) against Recon3D or an Entrez-format model —
        no separate BioMart download needed if you already have it.

    Returns
    -------
    pd.Series indexed by model gene IDs only.
    """
    if id_format == "auto":
        id_format = detect_id_format(expr.index)
        print(f"  Detected expression ID format: {id_format}")

    model_format = detect_model_gene_format(model)
    print(f"  Model gene ID format:         {model_format}")

    expr = expr.copy()

    if id_format == "depmap":

        if model_format == "ensembl":
            # Human1/Human2: extract Entrez from DepMap, map to Ensembl
            if gene_csv_path is None:
                raise ValueError("gene_csv_path (path to Gene.csv from DepMap) is required "
                                 "when using DepMap expression with a model that uses "
                                 "Ensembl gene IDs (Human1, Human2).")
            print("  Converting DepMap → Entrez → Ensembl (for Human1/Human2)")
            expr.index = extract_entrez_from_depmap(expr.index)

            if expr.index.duplicated().any():
                expr = expr.groupby(level=0).mean()

            entrez_to_ensembl = build_entrez_to_ensembl(gene_csv_path)
            expr.index = pd.Index([entrez_to_ensembl.get(str(g), g) for g in expr.index])

        elif model_format == "recon3d":
            print("  Converting DepMap → Entrez (for Recon3D)")
            expr.index = extract_entrez_from_depmap(expr.index)
            if expr.index.duplicated().any():
                expr = expr.groupby(level=0).mean()
            model_gene_ids = {g.id for g in model.genes}
            expr = match_depmap_to_recon3d(expr, model_gene_ids)
            coverage = len(expr) / len(model_gene_ids) * 100
            print(f"  Gene coverage: {len(expr)}/{len(model_gene_ids)} "
                  f"model genes ({coverage:.1f}%)")
            if coverage < 10.0:
                print("  WARNING: Very low gene coverage (<10%).")
            return expr

        elif model_format == "entrez":
            print("  Converting DepMap → Entrez")
            expr.index = extract_entrez_from_depmap(expr.index)

        else:
            # HGNC symbol model
            print("  Converting DepMap → HGNC symbol")
            expr.index = normalise_depmap_ids(expr.index)

    elif id_format == "ensembl":
        # Strip Ensembl version suffix once, up front, for every route below.
        # "ENSG00000141510.11" -> "ENSG00000141510"
        expr.index = pd.Index([str(g).split(".")[0] for g in expr.index])

        if model_format == "ensembl":
            # Human1/Human2 already use Ensembl gene IDs internally.
            print("  Ensembl input, Ensembl model (Human1/Human2) — "
                  "matching directly, no mapping file needed")

        elif model_format == "recon3d":
            print("  Converting Ensembl → Entrez (for Recon3D)")
            ensembl_to_entrez = _get_ensembl_to_entrez_map(gene_csv_path, ensembl_mapping_path)
            expr.index = pd.Index([ensembl_to_entrez.get(g, g) for g in expr.index])
            if expr.index.duplicated().any():
                expr = expr.groupby(level=0).mean()
            model_gene_ids = {g.id for g in model.genes}
            expr = match_depmap_to_recon3d(expr, model_gene_ids)
            coverage = len(expr) / len(model_gene_ids) * 100
            print(f"  Gene coverage: {len(expr)}/{len(model_gene_ids)} "
                  f"model genes ({coverage:.1f}%)")
            if coverage < 10.0:
                print("  WARNING: Very low gene coverage (<10%).")
            return expr

        elif model_format == "entrez":
            print("  Converting Ensembl → Entrez")
            ensembl_to_entrez = _get_ensembl_to_entrez_map(gene_csv_path, ensembl_mapping_path)
            expr.index = pd.Index([ensembl_to_entrez.get(g, g) for g in expr.index])

        else:
            # HGNC symbol model — none of Human1/Human2/Recon3D, but kept
            # for a future model that does use HGNC symbols internally.
            if ensembl_mapping_path is None:
                raise ValueError("Model uses HGNC symbols internally. Provide "
                                 "ensembl_mapping_path (BioMart TSV with columns "
                                 "ensembl_gene_id, hgnc_symbol) to map Ensembl → HGNC.")
            expr.index = normalise_ensembl_ids(expr.index, ensembl_mapping_path)

    # Handle duplicates
    if expr.index.duplicated().any():
        n_dups = expr.index.duplicated().sum()
        print(f"  Note: {n_dups} duplicate IDs — taking mean.")
        expr = expr.groupby(level=0).mean()

    # Filter to model genes only
    model_gene_ids = {g.id for g in model.genes}
    overlap = expr.index.intersection(list(model_gene_ids))
    coverage = len(overlap) / len(model_gene_ids) * 100

    print(f"  Gene coverage: {len(overlap)}/{len(model_gene_ids)} "
          f"model genes ({coverage:.1f}%)")

    if coverage < 10.0:
        print("  WARNING: Very low gene coverage (<10%). "
              "Check gene ID formats match between data and model.")

    return expr.loc[overlap]