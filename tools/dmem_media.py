"""
Media definitions for Human1.

Two DISTINCT real recipes exist across Supervisor's repos:
  DMEM_PLAIN — matches the paper for the exact five ovarian cell lines
               used in this project (Meeson & Schwartz 2024).
  DMEM_F12   — a different medium, used in her OV56 tutorial notebook.
Use DMEM_PLAIN (via DMEM_HIGH_GLUCOSE below) unless you have a specific
reason to match the OV56 tutorial's recipe instead.

WARNING on GAP_FILL_ONLY below: keep this as the full 22-entry set.
A trimmed-down version (1 entry, beta-carotene only) was tried based on
a single find_essential_reactions() run reporting the other 21 as
non-essential — that result did NOT reproduce on a second run (same
code, model, media dict, fresh process), which points to a stale
__pycache__ artifact rather than a real result. The full set below has
multiple independently-reproduced runs behind it (growth=0.084541
across all five cell lines, all four methods, repeated separately).
Don't re-trim without clearing all __pycache__ dirs AND reproducing
"not essential" at least twice in separate fresh processes.
"""

# ── DEFAULT: plain DMEM (matches this project's actual cell lines) ─────────
DMEM_PLAIN = {
    "MAR09066": -0.084,    # arginine
    "MAR09065": -0.063,    # cysteine
    "MAR09063": -0.584,    # glutamine
    "MAR09067": -0.03,     # glycine
    "MAR09038": -0.042,    # histidine
    "MAR09039": -0.105,    # isoleucine
    "MAR09040": -0.105,    # leucine
    "MAR09041": -0.146,    # lysine
    "MAR09042": -0.03,     # methionine
    "MAR09043": -0.066,    # phenylalanine
    "MAR09069": -0.042,    # serine
    "MAR09044": -0.095,    # threonine
    "MAR09045": -0.016,    # tryptophan
    "MAR09064": -0.104,    # tyrosine
    "MAR09046": -0.094,    # valine
    "MAR09083": -0.004,    # choline
    "MAR09146": -0.004,    # folate
    "MAR09361": -0.0072,   # inositol
    "MAR09378": -0.004,    # nicotinamide
    "MAR09145": -0.004,    # pantothenate
    "MAR09144": -0.004,    # pyridoxine
    "MAR09143": -0.0004,   # riboflavin
    "MAR09159": -0.004,    # thiamin
    "MAR09034": -4.5,      # glucose
}

# ── Alternative: DMEM/F12 (OV56 tutorial, different paper) ─────────────────
DMEM_F12 = {
    "MAR09034": -3.151,     # glucose
    "MAR09035": -0.000042,  # linoleate
    "MAR09038": -0.03148,   # histidine
    "MAR09039": -0.05447,   # isoleucine
    "MAR09040": -0.05905,   # leucine
    "MAR09041": -0.09125,   # lysine
    "MAR09042": -0.01724,   # methionine
    "MAR09043": -0.03548,   # phenylalanine
    "MAR09044": -0.05345,   # threonine
    "MAR09045": -0.00902,   # tryptophan
    "MAR09046": -0.05285,   # valine
    "MAR09061": -0.00445,   # alanine
    "MAR09062": -0.0075,    # asparagine
    "MAR09063": -0.365,     # glutamine
    "MAR09064": -0.05579,   # tyrosine
    "MAR09065": -0.03129,   # cysteine
    "MAR09066": -0.1475,    # arginine
    "MAR09067": -0.01875,   # glycine
    "MAR09068": -0.01725,   # proline
    "MAR09069": -0.02625,   # serine
    "MAR09070": -0.00665,   # aspartate
    "MAR09071": -0.00735,   # glutamate
    "MAR09083": -0.00898,   # choline
    "MAR09107": -1000.0,    # heme
    "MAR09109": -0.0000035, # biotin
    "MAR09143": -0.000219,  # riboflavin
    "MAR09144": -0.002013,  # pyridoxine
    "MAR09145": -0.00224,   # pantothenate
    "MAR09146": -0.00265,   # folate
    "MAR09152": -1000.0,    # alpha-tocotrienol
    "MAR09154": -1000.0,    # gamma-tocotrienol
    "MAR09159": -0.00217,   # thiamin
    "MAR09167": -0.000105,  # lipoic acid
    "MAR09269": -1000.0,    # aquacob(III)alamin (vitamin B12)
    "MAR09358": -0.00239,   # hypoxanthine
    "MAR09361": -0.0126,    # inositol
    "MAR09378": -0.00202,   # nicotinamide
    "MAR09423": -0.000365,  # thymidine
}

# ── FBS-related trace components — confirmed relevant by Meeson &
# Schwartz's own paper (heme specifically named as an FBS component
# they had to reopen for HEYA8).
FBS_TRACE_COMPONENTS = {
    "MAR09107": -1000.0,    # heme
    "MAR09109": -0.0000035, # biotin
    "MAR09152": -1000.0,    # alpha-tocotrienol
    "MAR09154": -1000.0,    # gamma-tocotrienol
    "MAR09167": -0.000105,  # lipoic acid
    "MAR09269": -1000.0,    # aquacob(III)alamin (vitamin B12)
}

# ── Full gap-fill set — see module docstring for why this must stay
# at all 22 entries rather than the trimmed 1-entry version.
GAP_FILL_ONLY = {
    "MAR09105": -1000.0,  # thiamin-P
    "MAR09215": -1000.0,  # calcidiol (vitamin D)
    "MAR09240": -1000.0,  # heptaglutamyl-folate(THF)
    "MAR09276": -1000.0,  # beta-carotene
    "MAR09376": -1000.0,  # NAD+
    "MAR09456": -1000.0,  # ADP-glucose
    "MAR09691": -1000.0,  # pyridoxal-phosphate
    "MAR00569": -1000.0,  # ATP
    "MAR01939": -1000.0,  # FAD
    "MAR04129": -1000.0,  # plasminogen
    "MAR04886": -1000.0,  # malonyl-carnitine
    "MAR08966": -1000.0,  # ubiquinone
    "MAR09026": -1000.0,  # SAH
    "MAR09923": -1000.0,  # octadecenoylcarnitine(5)
    "MAR10250": -1000.0,  # hexanoyl-CoA
    "MAR10254": -1000.0,  # linoleic-carnitine
    "MAR10265": -1000.0,  # N-(omega)-hydroxyarginine
    "MAR10496": -1000.0,  # tetrahydrobiopterin
    "MAR11344": -1000.0,  # de-Fuc form of PA6
    "MAR09051": -1000.0,  # LDL
    "MAR09055": -1000.0,  # LDL remnant
    "MAR11999": -1000.0,  # apoE
}

# MAR10024 ("biomass") is a separate essential reaction (matches supervisor's
# own find_essential_reactions() output on her model too). Does NOT
# need an entry here — strict media only zeroes the uptake side.

# ── Combined default for run_integration.py / pipeline.py ──────────────────
DMEM_HIGH_GLUCOSE = {**DMEM_PLAIN, **FBS_TRACE_COMPONENTS, **GAP_FILL_ONLY}
