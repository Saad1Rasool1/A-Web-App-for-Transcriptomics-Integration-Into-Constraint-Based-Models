"""
DMEM media for Recon3D.

Every reaction ID below was picked from the REAL candidate list produced by find_recon3d_media_candidates.py 
(run against the actual loaded Recon3D.xml, not guessed from BiGG naming convention) — see that script's output for the full candidate set each ID was chosen from.
For each of the 24 DMEM_PLAIN components there was exactly one unambiguous free-metabolite/vitamin exchange reaction; 
the rest of each candidate list was di/tripeptides and complex lipids that happened to share a name substring 
(e.g. searching "threonine" also matches "Threonyl-Seryl-Arginine") and are NOT what DMEM media actually contains.

UPTAKE RATES: reused directly from Human1's DMEM_PLAIN (dmem_media.py)
— DMEM's composition is a fixed real chemical formulation independent of which GEM it's applied to;
the numeric uptake bound for e.g. glucose describes the actual medium, not anything Human1-specific. 
Legitimate specifically because both halves (Supervisor's real DMEM values, and Recon3D's real reaction IDs below) are independently verified from real sources."""
from tools.dmem_media import DMEM_PLAIN

# Recon3D reaction ID for each DMEM_PLAIN component, confirmed against the real model via find_recon3d_media_candidates.py.
_RECON3D_EXCHANGE_ID = {
    "arginine":      "EX_arg__L_e",
    "cysteine":      "EX_cys__L_e",    # see caveat 1 above
    "glutamine":     "EX_gln__L_e",
    "glycine":       "EX_gly_e",
    "histidine":     "EX_his__L_e",
    "isoleucine":    "EX_ile__L_e",
    "leucine":       "EX_leu__L_e",
    "lysine":        "EX_lys__L_e",
    "methionine":    "EX_met__L_e",
    "phenylalanine": "EX_phe__L_e",
    "serine":        "EX_ser__L_e",
    "threonine":     "EX_thr__L_e",
    "tryptophan":    "EX_trp__L_e",
    "tyrosine":      "EX_tyr__L_e",
    "valine":        "EX_val__L_e",
    "choline":       "EX_chol_e",
    "folate":        "EX_fol_e",
    "inositol":      "EX_inost_e",
    "nicotinamide":  "EX_ncam_e",
    "pantothenate":  "EX_pnto__R_e",
    "pyridoxine":    "EX_pydxn_e",
    "riboflavin":    "EX_ribflv_e",
    "thiamin":       "EX_thm_e",
    "glucose":       "EX_glc__D_e",
}

# DMEM_PLAIN keys are named identically to the components above (see dmem_media.py's own inline comments), so this rekeys Human1's real uptake bounds onto Recon3D's real reaction IDs, component-by-component.
_HUMAN1_ID_TO_COMPONENT = {
    "MAR09066": "arginine", "MAR09065": "cysteine", "MAR09063": "glutamine",
    "MAR09067": "glycine", "MAR09038": "histidine", "MAR09039": "isoleucine",
    "MAR09040": "leucine", "MAR09041": "lysine", "MAR09042": "methionine",
    "MAR09043": "phenylalanine", "MAR09069": "serine", "MAR09044": "threonine",
    "MAR09045": "tryptophan", "MAR09064": "tyrosine", "MAR09046": "valine",
    "MAR09083": "choline", "MAR09146": "folate", "MAR09361": "inositol",
    "MAR09378": "nicotinamide", "MAR09145": "pantothenate",
    "MAR09144": "pyridoxine", "MAR09143": "riboflavin", "MAR09159": "thiamin",
    "MAR09034": "glucose",
}

RECON3D_DMEM_PLAIN = {_RECON3D_EXCHANGE_ID[component]: DMEM_PLAIN[human1_id] for human1_id, component in _HUMAN1_ID_TO_COMPONENT.items()}

# Gap-fill: found via find_recon3d_gapfill_manual.py -- a real two-phase
# greedy-reopen-then-shrink search (basic slim_optimize()/bound-setting
# only, nothing relying on any library's internal search scope) against
# the actual loaded Recon3D model with RECON3D_DMEM_PLAIN applied. Of
# 246 candidate boundary reactions invisible to model.exchanges (the
# same SK_/DM_ class of reaction that was silently leaking before the
# model.boundary fix in _apply_media), only these 2 were found
# necessary to reach nonzero growth; the rest were redundant. Mirrors
# Human1's own GAP_FILL_ONLY in spirit (reactions found necessary via a
# real essential-reaction search, not derived from DMEM's textbook
# recipe) but is Recon3D's own independently-verified result, not a
# copy of Human1's specific entries.
#
# CAVEAT: this is a locally-minimal set (see find_recon3d_gapfill_manual.py's
# own docstring) -- a different candidate search order could in
# principle find a different, equally-valid minimal set. -1000.0 bounds
# (fully open, not a calibrated real uptake rate) match the convention
# GAP_FILL_ONLY already uses for Human1, for the same reason: these
# represent "must be available in principle" rather than a measured
# physiological concentration.

RECON3D_GAP_FILL_ONLY = { "SK_tag_hs_c":  -1000.0,   # Triacylglycerol (homo sapiens)
                          "SK_tchola_c":  -1000.0,   # Taurocholic acid
                          }

# Combined default for run_tcga.py / pipeline.py's Recon3D media lookup.
RECON3D_DMEM_HIGH_GLUCOSE = {**RECON3D_DMEM_PLAIN, **RECON3D_GAP_FILL_ONLY}
