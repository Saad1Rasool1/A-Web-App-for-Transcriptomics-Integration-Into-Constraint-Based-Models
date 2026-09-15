# Transcriptomics-to-GEM Integration Pipeline

Computational backend for integrating transcriptomic data into genome-scale metabolic models (GEMs). 
This is the analytical backend for the Streamlit app. Everything here is called through `tools/pipeline.py`

## What this does

- Four integration methods: `pfba_baseline`, `continuous_meeson`, `gimme`, `imat`
- Three models: **Human1**, **Human2**, **Recon3D**
- Two expression data sources: **CCLE/DepMap** (cell lines) and **TCGA** (tumour samples)
- Runs entirely on **GLPK** (free, open-source) 

## Getting started

Set up a fresh virtual environment:

```bash
# from the repo root (NOTE: must be running python version 3.11)
py -3.11 -m venv venv311 
```
```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
source venv/bin/activate
```
```bash
pip install -r requirements.txt
```

Then set up `DATA/` (see below), and confirm everything's working:

```bash
pytest tests/ -v
```

You should see `40 passed`. If anything fails, don't proceed to run the example scripts until it's sorted.

GLPK itself needs to be installed at the system level for `optlang`'s GLPK 
interface to work (usually already satisfied on most systems `cobra`/`optlang`
are tested on, but if `import optlang; optlang.available_solvers` doesn't
show `'GLPK': True`, install it via your OS package manager first).

**Do not install `gurobipy` or `PySCIPOpt` in the deployment environment.**


## Data setup

Data files are **not** included in this repository (large, and DepMap/CCLE files carry their own terms of use — see [depmap.org](https://depmap.org/portal/download/all/)). 
Expected layout:

```
DATA/
├── Models/
│   ├── Human1.xml
│   ├── Human2.xml
│   └── Recon3D.xml
└── Expression/
    ├── OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv   # DepMap
    ├── Model.csv                                             # DepMap cell-line metadata
    ├── Gene.csv                                               # Entrez<->Ensembl bridge
    └── CRISPRGeneDependency.csv                               # only needed for run_gene_essentiality_validation.py
```

- **Human1 / Human2**: [SysBioChalmers/Human-GEM](https://github.com/SysBioChalmers/Human-GEM)
- **Recon3D**: [BiGG Models](http://bigg.ucsd.edu/models/Recon3D)
- **DepMap files**: [depmap.org/portal/download](https://depmap.org/portal/download/all/)
- **TCGA expression** (only needed if using TCGA as a data source, not DepMap):
  [UCSC Xena](https://xenabrowser.net/datapages/), Toil-recompute pan-cancer matrices

## Usage

```python
from tools.pipeline import run_transcriptomics_integration

fluxes, info = run_transcriptomics_integration(
    model_path="DATA/Models/Human1.xml",
    expression_path="DATA/Expression/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
    cell_line_name="CAOV3",
    model_csv_path="DATA/Expression/Model.csv",
    gene_csv_path="DATA/Expression/Gene.csv",   # only needed for Recon3D (Ensembl<->Entrez bridge)
    method="continuous_meeson",                  # pfba_baseline | continuous_meeson | gimme | imat
    media=None,                                  # None = auto-select the right medium per model (recommended)
)

print(info["predicted_growth"], info["status"])
fluxes.to_csv("output.csv")
```

For a TCGA sample instead of a DepMap cell line, pass `tcga_sample_barcode=` instead of `cell_line_name=` (don't pass both)

`media=None` auto-selects a medium per model
(see `tools/default_media.py`). Pass `media=False` to force no medium
constraints, or a dict to supply your own — but `None` is almost always what you want.

See `run_integration.py`, `run_tcga.py`, and `run_gene_essentiality_validation.py`
for fuller worked scripts (batch runs across multiple cell lines, TCGA
single-sample runs, gene-essentiality validation). Each has a USER
INPUT block near the top with hardcoded paths and settings (data
locations, which cell lines, which model) that you may need to edit for your own setup before running.


## Running the tests

```bash
pytest tests/ -v
```

40 tests covering gene-ID mapping, medium application, solver selection, and the numerical behaviour of each integration method.
Run this before and after integrating into the app to confirm nothing regressed.

## Known limitations

- **iMAT without a secondary flux-minimisation objective** can leave a
  meaningful number of reactions sitting at the raw ±1000 flux bound rather
  than a calibrated value, since iMAT's own objective doesn't distinguish
  between the two. A `parsimonious=True` option exists to fix this but is
  disabled by default: on GLPK, at genome scale, it has been observed to
  return an incorrect "infeasible" result rather than a slower-but-correct
  one. Treat `imat` results with this in mind, or enable `parsimonious=True`
  with a `time_limit`/`mip_gap` set and verify the result before trusting it.

- **`continuous_meeson`'s growth-threshold correction can fail** for cell
  lines whose experimental growth threshold exceeds what the medium can
  support at all (confirmed for two specific cell lines — HEYA8, CAOV3 — on
  both Human1 and Human2). `info["threshold_unreachable"]` reports this
  explicitly when it happens rather than failing silently.

- **Recon3D's medium** has a smaller, more narrowly-scoped gap-fill set than
  Human1's (2 components vs. Human1's ~28), found via a targeted search
  rather than the more extensive essential-reaction analysis behind
  Human1's medium. It produces sensible growth, but hasn't had the same
  depth of curation.

- Growth rate alone is **not informative** for `gimme`/`imat` specifically —
  both derive their growth bound from an expression-independent reference
  value, so identical growth across different inputs is expected, not a
  bug. Use flux-level comparison (see `build_cross_model_summary.py`
  for the pattern) to see what expression data actually changed.

## Credits

The `continuous_meeson` method and the original Human1 medium composition
are adapted from Meeson & Schwartz (2024), *Constraint-based modelling
predicts metabolic signatures of low and high-grade serous ovarian cancer*,
npj Systems Biology and Applications 10:96.
