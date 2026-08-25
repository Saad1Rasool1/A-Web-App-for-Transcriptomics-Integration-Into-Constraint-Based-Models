"""
Per-model default media lookup.

Implements Supervisor's requested pattern: "each model should have a default media formulation, then let the user change it if they want."
pipeline.run_transcriptomics_integration() falls back to whatever is registered here when media=None (the default) is passed 
An explicit media dict, or media=False to force no constraints, still overrides this.

Human1: full DMEM_HIGH_GLUCOSE (24 core components + FBS trace + gap-fill), the original, most thoroughly verified entry.

Human2: reuses Human1's DMEM_HIGH_GLUCOSE directly — confirmed by check_human2_media_compatibility.py 
(all 52 reaction IDs exist in Human2.xml) and check_human2_metabolite_identity.py (spot-checked IDs map to the same metabolite in both models, not just the same ID string), so this is a verified reuse, not an assumption.

Recon3D: RECON3D_DMEM_HIGH_GLUCOSE 
The 24 core DMEM components plus 2 gap-fill reactions (SK_tag_hs_c, SK_tchola_c), found via a real two-phase search against the actual loaded Recon3D model (find_recon3d_gapfill_manual.py)
see tools/dmem_media_recon3d.py's own docstring for full provenance). 
Verified end-to-end: growth went 75.5 (no media) -> 37 (core media only, later found to still leak via 95 SK_/DM_ reactions model.exchanges doesn't catch) -> 0.0 (leak fixed, but genuinely needs gap-fill) -> 0.19 (gap-fill added)
A biologically plausible number, same order of magnitude as Human1's own ceiling."""

from tools.dmem_media import DMEM_HIGH_GLUCOSE
from tools.dmem_media_recon3d import RECON3D_DMEM_HIGH_GLUCOSE

DEFAULT_MEDIA_BY_MODEL = {"Human1": DMEM_HIGH_GLUCOSE, "Human2": DMEM_HIGH_GLUCOSE, "Recon3D": RECON3D_DMEM_HIGH_GLUCOSE,}

def get_default_media(model_filename: str) -> dict | None:
    """Look up the default media for a model by filename (e.g. "Human1.xml").
    Returns None if no default is registered for that model — callers should treat this as "no media constraints", not an error."""
    for name, media in DEFAULT_MEDIA_BY_MODEL.items():
        if name.lower() in model_filename.lower():
            return media
    return None
