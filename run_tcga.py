"""
Single-sample TCGA integration run.

Mirrors run_integration.py's structure and style, but pulls expression
data from a TCGA/Xena expression matrix instead of DepMap, via
tools.pipeline.run_transcriptomics_integration() — the same clean entry
point Streamlit interface uses, so this also doubles as a real
test of that code path.

BEFORE RUNNING:
  1. Place Human1.xml (or Human2.xml / Recon3D.xml) wherever
     run_integration.py already expects it (DATA/Models/).
  2. Place TCGA-OV.star_tpm.tsv.gz in data/tcga/ (or wherever you prefer
     — just update TCGA_EXPRESSION_PATH below).
  3. Find a real barcode first — see find_a_barcode() at the bottom of
     this file, or just run:
         python -c "
         from tools.data_loaders import search_tcga_samples
         print(search_tcga_samples('data/tcga/TCGA-OV.star_tpm.tsv.gz', 'TCGA'))
         "
  4. Set TCGA_SAMPLE_BARCODE below to one of those, then:
         python run_tcga_integration.py
"""
from pathlib import Path
from tools.pipeline import run_transcriptomics_integration
from tools.data_loaders import search_tcga_samples

# =============================================================================
# USER INPUT SECTION — only edit values in this block
# =============================================================================

MODEL_PATH = Path(r"C:\Users\Saad\Documents\Project 2\DATA\Models\Recon3D.xml")
TCGA_EXPRESSION_PATH = Path(r"C:\Users\Saad\Documents\Project 2\data\tcga\TCGA-OV.star_tpm.tsv")

# Only needed if MODEL_PATH is Recon3D (bridges Ensembl -> Entrez).
# Reuses the same Gene.csv already set up for run_integration.py.
GENE_CSV_PATH = Path(r"C:\Users\Saad\Documents\Project 2\DATA\Expression\Gene.csv")

# ── Sample ────────────────────────────────────────────────────────────────
# Leave blank first, run once — the script will list available barcodes
# and stop, so you can pick a real one and re-run.
TCGA_SAMPLE_BARCODE = "TCGA-24-1104"    # or "TCGA-24-1104-01A" for the exact one
TCGA_SAMPLE_TYPE = "01"     # 01 = primary tumour, 11 = normal tissue

# ── Method ────────────────────────────────────────────────────────────────
METHOD = "imat"   # pfba_baseline | continuous_meeson | gimme | imat
# See pipeline.py's run_transcriptomics_integration docstring.
# CONFIRMED via a real run: PARSIMONIOUS=True on iMAT at genome scale
# (Recon3D, ~10600 reactions) returned a false "INFEASIBLE" after ~10
# minutes with no time limit set -- GLPK numerical fragility at scale,
# not a real contradiction (the pre-parsimony solution remains a valid
# feasible point mathematically). Left OFF by default until this is
# better understood; don't flip this back on without also setting
# IMAT_TIME_LIMIT/IMAT_MIP_GAP below, so a repeat runs bounded instead
# of unbounded.
PARSIMONIOUS = False
IMAT_TIME_LIMIT = 120.0    # seconds -- only applies when METHOD == "imat"
IMAT_MIP_GAP    = 0.05     # accept within 5% of optimal

# ── Baseline comparison ─────────────────────────────────────────────────────
# growth alone doesn't tell you much for gimme/imat (see run_integration.py's
# subtype-differential-flux diagnostic and everything discussed alongside it
# — both methods' growth bound derives from an expression-INDEPENDENT
# wt_growth reference, so the scalar is largely uninformative about what the
# expression data actually did). For a single TCGA sample there's no other
# cell line to diff against, so the natural comparison is against
# pfba_baseline (the no-expression benchmark) on the SAME model+media: which
# reactions did this sample's expression actually move, and by how much.
# Set False to skip this and only run METHOD (faster, matches old behaviour).
COMPARE_TO_BASELINE = True
# Minimum absolute flux difference (mmol/gDW/h) for a reaction to count as
# "moved" — same spirit as run_integration.py's differential-flux magnitude
# threshold, filters out reactions that technically differ but are near
# numerical noise level.
BASELINE_DIFF_THRESHOLD = 0.5

RESULTS_DIR = Path(r"C:\Users\Saad\Documents\Project 2\results\tcga")

# =============================================================================
# End of user input — do not edit below this line
# =============================================================================


def main():
    if not TCGA_SAMPLE_BARCODE:
        print("TCGA_SAMPLE_BARCODE is not set. Available barcodes "
              "(first 20 matches for 'TCGA'):\n")
        for b in search_tcga_samples(TCGA_EXPRESSION_PATH, "TCGA", max_results=20):
            print(f"  {b}")
        print("\nSet TCGA_SAMPLE_BARCODE to one of these (barcode prefix is "
              "enough, e.g. 'TCGA-25-1319') and re-run.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    is_recon3d = "recon3d" in MODEL_PATH.stem.lower()

    print("=" * 60)
    print("TCGA Integration Run")
    print(f"Sample:  {TCGA_SAMPLE_BARCODE}")
    print(f"Model:   {MODEL_PATH.name}")
    print(f"Method:  {METHOD}")
    print("=" * 60)

    fluxes, info = run_transcriptomics_integration(
        model_path=MODEL_PATH,
        expression_path=TCGA_EXPRESSION_PATH,
        tcga_sample_barcode=TCGA_SAMPLE_BARCODE,
        tcga_sample_type=TCGA_SAMPLE_TYPE,
        gene_csv_path=GENE_CSV_PATH if is_recon3d else None,
        method=METHOD,
        parsimonious=PARSIMONIOUS if METHOD in ("gimme", "imat") else None,
        time_limit=IMAT_TIME_LIMIT if METHOD == "imat" else None,
        mip_gap=IMAT_MIP_GAP if METHOD == "imat" else None,)

    print(f"\nStatus:         {info['status']}")
    print(f"Predicted growth: {info['predicted_growth']}")
    print(f"Genes matched:  {info['genes_matched']}")

    out = RESULTS_DIR / f"{TCGA_SAMPLE_BARCODE}_{METHOD}_fluxes.csv"
    fluxes.to_csv(out, header=["flux_mmol_gDW_h"])
    print(f"Saved: {out}")

    # ── Flux-vs-baseline comparison ─────────────────────────────────────────
    if COMPARE_TO_BASELINE and METHOD != "pfba_baseline":
        print(f"\nRunning pfba_baseline for comparison (same model+media, "
              f"no expression)...")
        baseline_fluxes, baseline_info = run_transcriptomics_integration(
            model_path=MODEL_PATH,
            expression_path=TCGA_EXPRESSION_PATH,
            tcga_sample_barcode=TCGA_SAMPLE_BARCODE,
            tcga_sample_type=TCGA_SAMPLE_TYPE,
            gene_csv_path=GENE_CSV_PATH if is_recon3d else None,
            method="pfba_baseline",)

        # gimme/imat go through MEWpy's sim object (an SBML round-trip),
        # whose reaction IDs are "R_"-prefixed relative to the underlying
        # cobra model's own IDs (see run_gimme's env_cond-building code:
        # r_id = "R_" + rxn.id). pfba_baseline/continuous_meeson use plain
        # cobrapy directly, unprefixed. Strip the prefix so the two
        # indices actually line up for comparison -- confirmed necessary
        # via a real "0 of 0 compared" result before this fix (not "no
        # differences found", the indices genuinely didn't overlap at all).
        compare_fluxes = fluxes
        if METHOD in ("gimme", "imat"):
            compare_fluxes = fluxes.rename(lambda k: k[2:] if isinstance(k, str) and k.startswith("R_") else k)

        common = compare_fluxes.index.intersection(baseline_fluxes.index)
        if len(common) == 0:
            if len(compare_fluxes) == 0:
                print(
                    f"\n{METHOD} run returned an empty flux vector (status "
                    f"was likely INFEASIBLE — check the Status line above) "
                    f"-- nothing to compare against baseline. Not a "
                    f"prefix-mismatch issue, just no solution to diff."
                )
            else:
                print(
                    f"\nWARNING: still 0 reactions in common after R_-prefix "
                    f"stripping ({len(compare_fluxes)} {METHOD} reactions, "
                    f"{len(baseline_fluxes)} baseline reactions) -- the ID "
                    f"mismatch is something other than the R_ prefix. Don't "
                    f"trust this comparison; share both index samples "
                    f"(e.g. list(fluxes.index[:5]) and "
                    f"list(baseline_fluxes.index[:5])) before relying on it."
                )
        else:
            diff = (compare_fluxes[common] - baseline_fluxes[common]).abs()
            moved = diff[diff >= BASELINE_DIFF_THRESHOLD].sort_values(ascending=False)

            print(f"\nReactions differing from pfba_baseline by >= "
                  f"{BASELINE_DIFF_THRESHOLD} mmol/gDW/h: {len(moved)} "
                  f"(of {len(common)} compared)")
            if len(moved) > 0:
                print("Top 20 by magnitude:")
                for rxn_id, d in moved.head(20).items():
                    print(f"  {rxn_id}: {METHOD}={compare_fluxes[rxn_id]:.4f}  "
                          f"baseline={baseline_fluxes[rxn_id]:.4f}  diff={d:.4f}")

            comparison_df = compare_fluxes[common].to_frame(name=f"{METHOD}_flux")
            comparison_df["baseline_flux"] = baseline_fluxes[common]
            comparison_df["abs_diff"] = diff
            comparison_out = RESULTS_DIR / f"{TCGA_SAMPLE_BARCODE}_{METHOD}_vs_baseline.csv"
            comparison_df.sort_values("abs_diff", ascending=False).to_csv(comparison_out)
            print(f"Full comparison saved: {comparison_out}")


if __name__ == "__main__":
    main()