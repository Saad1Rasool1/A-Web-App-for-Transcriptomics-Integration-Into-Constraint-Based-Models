"""
Load transcriptomics expression data from different sources.

All loaders return a raw pd.Series (gene_id -> expression value) before ID normalisation. 
Normalisation is handled separately by gene_mapping.py."""
import pandas as pd
from pathlib import Path

# Cache full expression matrices in memory, keyed by resolved file path.
# Both load_depmap and load_tcga re-read the same large file once per
# sample/cell-line when called in a loop (e.g. run_integration.py's five
# cell lines, or a multi-sample TCGA run) — this avoids re-parsing a
# 300+MB file from disk every time. Trades memory for speed: fine for a
# handful of large files, not intended for many distinct large matrices
# in one process. Call clear_expression_cache() to free the memory back up.
_matrix_cache: dict[str, pd.DataFrame] = {}


def clear_expression_cache() -> None:
    """Free memory held by cached expression matrices (see _matrix_cache)."""
    _matrix_cache.clear()


def _read_matrix_cached(expression_path: str | Path, use_cache: bool, **read_csv_kwargs,) -> pd.DataFrame:
    """Shared read-with-cache helper for load_depmap and load_tcga."""
    key = str(Path(expression_path).resolve())

    if use_cache and key in _matrix_cache:
        print(f"[data_loaders] Using cached matrix: {Path(key).name}")
        return _matrix_cache[key]

    df = pd.read_csv(expression_path, **read_csv_kwargs)

    if use_cache:
        _matrix_cache[key] = df

    return df


def find_model_id(model_csv_path: str | Path, cell_line_name: str) -> str:
    """
    Look up a cell line's DepMap ModelID (ACH-XXXXXX) from Model.csv using its common name.
    """
    model_df = pd.read_csv(model_csv_path)

    mask = (model_df["CellLineName"].str.upper().str.strip() == cell_line_name.upper().strip()) | (model_df["StrippedCellLineName"].str.upper().str.strip() == cell_line_name.upper().strip())

    matches = model_df[mask]

    if len(matches) == 0:
        available = model_df["CellLineName"].dropna().sort_values().tolist()
        raise ValueError(f"Cell line '{cell_line_name}' not found in Model.csv.\n"
                         f"Check spelling. Some examples:\n{available[:20]}")

    return matches.iloc[0]["ModelID"]


def load_depmap(expression_path: str | Path, model_csv_path: str | Path, cell_line_name: str, use_cache: bool = True,) -> pd.Series:
    """
    Load expression for one cell line from the DepMap expression CSV.

    Parameters
    ----------
    expression_path : str or Path
        Path to OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
    model_csv_path : str or Path
        Path to Model.csv
    cell_line_name : str
        Common cell line name e.g. "CAOV3", "59M", "HEYA8"
    use_cache : bool
        Reuse the in-memory expression matrix across calls (e.g. looping
        over multiple cell lines) instead of re-reading the file each
        time. Default True. Set False to force a fresh read.

    Returns
    -------
    pd.Series indexed by raw DepMap gene names ("TP53 (7157)" format).
    """
    print(f"[data_loaders] Loading expression for: {cell_line_name}")

    model_id = find_model_id(model_csv_path, cell_line_name)
    print(f"  Resolved ModelID: {model_id}")

    print(f"  Reading expression file...")
    expr_df = _read_matrix_cached(expression_path, use_cache, index_col=0, low_memory=False)

    if "ModelID" not in expr_df.columns:
        raise ValueError("Expression file does not have a 'ModelID' column. " 
                         "Check you downloaded OmicsExpressionTPMLogp1Human" 
                         "ProteinCodingGenes.csv from DepMap.")

    model_rows = expr_df[expr_df["ModelID"] == model_id]

    if len(model_rows) == 0:
        raise ValueError(f"ModelID '{model_id}' not found in expression file. " 
                         f"The cell line may not have RNA-seq data in this release.")

    if "is_default_entry" in model_rows.columns:
        default_rows = model_rows[model_rows["is_default_entry"] == True]
        if len(default_rows) > 0:
            model_rows = default_rows

    row = model_rows.iloc[0]

    meta_cols = {"ModelID", "is_default_entry", "ProfileID"}
    expr = row.drop(labels=[c for c in meta_cols if c in row.index])
    expr = pd.to_numeric(expr, errors="coerce").dropna()

    print(f"  Loaded {len(expr)} genes.")
    return expr


def load_tcga(expression_path: str | Path, sample_barcode: str, sample_type: str = "01", use_cache: bool = True,) -> pd.Series:
    """
    Load expression for one sample from a TCGA gene-expression matrix, as
    downloaded from UCSC Xena (https://xenabrowser.net/datapages/) — the
    Toil-recompute pipeline specifically, which uses Ensembl gene IDs
    (other Xena TCGA hubs use HUGO symbols instead and won't auto-detect
    correctly here).

    Expected file format (matches Xena's TCGA/GTEx Toil-recompute exports):
      Rows:    Ensembl gene IDs, with or without version suffix
               (e.g. "ENSG00000141510" or "ENSG00000141510.11")
      Columns: TCGA sample barcodes (e.g. "TCGA-25-1319-01")
      Values:  log2(x+1)-transformed expression (same scale as DepMap's
               log2(TPM+1) — normalise_to_model_ids needs no changes here)
      May be gzip-compressed (.gz) or plain — handled automatically
      either way.

    Parameters
    ----------
    expression_path : str or Path
        Path to the Xena expression matrix (.tsv, .tsv.gz, or .csv).
    sample_barcode : str
        TCGA patient/sample barcode, or a prefix of one, e.g.
        "TCGA-25-1319". Matched by substring against column headers.
    sample_type : str
        TCGA sample type code suffix to prefer when a barcode matches
        multiple columns (tumour vs normal vs metastatic, etc).
        Default "01" = primary solid tumour. Use "11" for normal
        tissue, "06" for metastatic. See the TCGA barcode reference:
        https://docs.gdc.cancer.gov/Encyclopedia/pages/TCGA_Barcode/
    use_cache : bool
        Reuse the in-memory matrix across calls (e.g. looping over
        multiple samples from the same file) instead of re-reading a
        potentially 300+MB file each time. Default True.

    Returns
    -------
    pd.Series indexed by raw Ensembl gene IDs (version suffix intact —
    stripped later by normalise_to_model_ids), for the matched sample.
    """
    print(f"[data_loaders] Loading TCGA expression for: {sample_barcode}")

    expression_path = Path(expression_path)
    suffix = "".join(expression_path.suffixes)  # handles ".tsv.gz"
    sep = "\t" if ".tsv" in suffix or ".gz" in suffix else ","
    compression = "gzip" if expression_path.suffix == ".gz" else "infer"

    print("  Reading TCGA expression matrix (large file, may take a "
          "while on first read)...")
    df = _read_matrix_cached(expression_path, use_cache, index_col=0, sep=sep, compression=compression, low_memory=False,)

    matches = [c for c in df.columns if sample_barcode.upper() in c.upper()]

    if len(matches) == 0:
        raise ValueError(f"No sample matching '{sample_barcode}' found in " 
                         f"{expression_path.name}. Check the barcode, or use " 
                         f"search_tcga_samples() to list available barcodes.")

    if len(matches) > 1:
        type_matches = [c for c in matches if f"-{sample_type}" in c]
        if type_matches:
            matches = type_matches
        print(f"  Multiple columns matched '{sample_barcode}': {matches}. "
              f"Using: {matches[0]}")

    chosen_col = matches[0]
    expr = pd.to_numeric(df[chosen_col], errors="coerce").dropna()

    # Xena occasionally has duplicate Ensembl IDs (different transcript
    # versions collapsed to the same gene). Same handling as DepMap path.
    if expr.index.duplicated().any():
        n_dups = expr.index.duplicated().sum()
        print(f"  Note: {n_dups} duplicate gene IDs — taking mean.")
        expr = expr.groupby(level=0).mean()

    print(f"  Loaded {len(expr)} genes for sample: {chosen_col}")
    return expr


def search_tcga_samples(expression_path: str | Path, search_term: str, max_results: int = 20) -> list[str]:
    """
    List column headers (sample barcodes) in a TCGA/Xena expression matrix
    matching a search term. Useful for finding the exact barcode to pass
    to load_tcga() without reading the full file into memory.

    Usage:
        python -c "
        from tools.data_loaders import search_tcga_samples
        print(search_tcga_samples('DATA/TCGA/tcga_ov_tpm.tsv.gz', 'TCGA-25'))
        "
    """
    expression_path = Path(expression_path)
    suffix = "".join(expression_path.suffixes)
    sep = "\t" if ".tsv" in suffix or ".gz" in suffix else ","
    compression = "gzip" if expression_path.suffix == ".gz" else "infer"

    header = pd.read_csv(expression_path, sep=sep, compression=compression, nrows=0)
    matches = [c for c in header.columns if search_term.upper() in c.upper()]
    return matches[:max_results]


def load_generic_csv(filepath: str | Path, sample_col: int | str = 0) -> pd.Series:
    """
    Load expression from any generic CSV or Excel file.

    Expected format:
      Row index: gene identifiers (any supported format)
      Columns:   one or more sample expression values

    Parameters
    ----------
    filepath : str or Path
    sample_col : int or str
        Column to use. 0 = first data column (default).

    Returns
    -------
    pd.Series indexed by gene IDs (raw, not yet normalised).
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath, index_col=0)
    elif suffix == ".tsv":
        df = pd.read_csv(filepath, index_col=0, sep="\t")
    else:
        df = pd.read_csv(filepath, index_col=0)

    expr = df.iloc[:, sample_col] if isinstance(sample_col, int) \
        else df[sample_col]

    print(f"[data_loaders] Loaded {len(expr)} genes from {filepath.name}")
    return expr


def search_cell_lines(model_csv_path: str | Path, search_term: str) -> pd.DataFrame:
    """
    Search Model.csv for cell lines matching a search term.
    Useful for finding correct name spellings.

    Usage:
        python -c "
        from tools.data_loaders import search_cell_lines
        print(search_cell_lines('DATA/Expression/Model.csv', 'ovarian'))
        "
    """
    df = pd.read_csv(model_csv_path)
    mask = (df["CellLineName"].str.contains(search_term, case=False, na=False) | df["OncotreeLineage"].str.contains(search_term, case=False, na=False) | df["OncotreePrimaryDisease"].str.contains(search_term, case=False, na=False))
    cols = ["ModelID", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype"]
    return df[mask][cols].reset_index(drop=True)
