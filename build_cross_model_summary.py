"""
Cross-model comparison summary: growth ceilings and subtype-differential-
flux counts for Human1, Human2, and Recon3D side by side, read directly
from the actual result files each run_integration.py run already
produced -- not retyped from console output, so this stays correct even
if you re-run any of the three later.

CAVEAT worth keeping in mind when reading the output: the growth
ceiling is directly comparable WITHIN the Human1/Human2 family (both
use Human-GEM's MAR-prefixed biomass reaction, same units, same
convention) but is NOT necessarily comparable to Recon3D's ceiling --
different reconstructions can define biomass differently even in the
same nominal units (mmol/gDW/h), so "Recon3D's ceiling is ~2.3x
Human1/2's" is not the same claim as "Recon3D grows 2.3x faster" --
see the media provenance notes in tools/dmem_media_recon3d.py.

Usage: python build_cross_model_summary.py
"""

# --- Path bootstrap: lets this script find tools/ regardless of
# where it's run from (this file lives one level below the repo
# root, where tools/ actually is). ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT2_DATA_DIR", Path(__file__).resolve().parent))
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS = ["Human1", "Human2", "Recon3D"]
DIFF_METHODS = ["continuous_meeson", "gimme", "imat"]


def load_growth_ceiling(model_name: str) -> dict:
    """Reads growth_summary.csv -- all cell lines show the same value
    per method (see run_integration.py's discussion of why), so the
    first row is representative; included as a sanity check that this
    is still true rather than assumed blindly.

    Uses a tolerance-based check (max-min > 1e-6), not nunique():
    nunique() does exact floating-point equality, and CSV round-
    tripping (write then read back) can introduce bit-level
    representation differences that are numerically meaningless but
    still count as "different" to exact equality -- confirmed via a
    real false positive on Recon3D (nunique() said non-uniform;
    directly inspecting the values showed max-min == 0.000000)."""
    path = RESULTS_DIR / model_name / "growth_summary.csv"
    if not path.exists():
        return {method: None for method in ["pfba_baseline"] + DIFF_METHODS}

    df = pd.read_csv(path, index_col=0)
    non_uniform = [col for col in df.columns
                   if (df[col].max() - df[col].min()) > 1e-6]
    if non_uniform:
        print(f"  NOTE ({model_name}): growth is NOT uniform across cell "
              f"lines for {non_uniform} (max-min > 1e-6) -- using the "
              f"mean instead of assuming a single representative value. "
              f"Worth a closer look at why this differs from the other "
              f"models' runs, which were all uniform.")
        return {col: df[col].mean() for col in df.columns}
    return {col: df[col].iloc[0] for col in df.columns}


def load_differential_flux_count(model_name: str, method: str) -> int | None:
    path = RESULTS_DIR / model_name / f"{method}_subtype_differential_flux.csv"
    if not path.exists():
        return None
    return len(pd.read_csv(path, index_col=0))


def main():
    rows = []
    for model_name in MODELS:
        growth = load_growth_ceiling(model_name)
        row = {"model": model_name,
            "growth_ceiling_pfba_baseline": growth.get("pfba_baseline"),
            "growth_gimme": growth.get("gimme"),
            "growth_imat": growth.get("imat"),}
        for method in DIFF_METHODS:
            row[f"differential_flux_{method}"] = load_differential_flux_count(model_name, method)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")

    print("=" * 70)
    print("Cross-model comparison")
    print("=" * 70)
    print(summary.to_string())
    print(
        "\nCAVEAT: growth ceilings are directly comparable WITHIN "
        "Human1/Human2 (same reconstruction family, same biomass "
        "convention) but not necessarily comparable to Recon3D's -- see "
        "this script's own docstring.")

    missing = summary.isna().any().any()
    if missing:
        print(
            "\nSome values are missing (None) -- likely because that "
            "model/method combination hasn't been run yet, or its "
            "results directory doesn't exist. Run run_integration.py "
            "for the missing model(s) first.")

    out_path = RESULTS_DIR / "cross_model_comparison.csv"
    summary.to_csv(out_path)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
