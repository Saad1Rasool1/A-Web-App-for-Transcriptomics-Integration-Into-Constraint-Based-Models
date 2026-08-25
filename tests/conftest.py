"""
Shared fixtures for the regression test suite.

These toy models are deliberately minimal (3-5 reactions), but each one
is built to reproduce a specific bug found while developing the pipeline against the real Human1/Recon3D models 
See each test file for which bug each fixture targets."""
import gzip
import pandas as pd
import cobra
import pytest


@pytest.fixture
def linear_chain_model():
    """A -> B -> C -> BIOMASS, via reactions R1 (high-expr), R2 (low-expr). 
    Human1-format gene IDs (Ensembl). 
    Used for: basic iMAT low/high branch behaviour, epsilon handling, growth floor mechanics."""
    m = cobra.Model("linear_chain")
    m.compartments = {"c": "cytosol"}
    a = cobra.Metabolite("A", compartment="c")
    b = cobra.Metabolite("B", compartment="c")
    c = cobra.Metabolite("C", compartment="c")

    r_ex = cobra.Reaction("EX_A")
    r_ex.lower_bound, r_ex.upper_bound = -1000, 1000
    r1 = cobra.Reaction("R1")
    r1.lower_bound, r1.upper_bound = -1000, 1000
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("R2")
    r2.lower_bound, r2.upper_bound = -1000, 1000
    r2.gene_reaction_rule = "g2"
    biomass = cobra.Reaction("BIOMASS")
    biomass.lower_bound, biomass.upper_bound = 0, 1000
    biomass.gene_reaction_rule = "g3"

    r_ex.add_metabolites({a: -1})
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1, c: 1})
    biomass.add_metabolites({c: -1})
    m.add_reactions([r_ex, r1, r2, biomass])
    m.objective = "BIOMASS"
    return m


@pytest.fixture
def forced_negative_flux_model():
    """Reproduces exact structure found in Human1's real IIS (M_MAM00247c / MAR04368 / MAR04371 conflict): 
    a growth floor forces a positive term into a metabolite balance, which can only be cancelled by a high-expression, reversible reaction (R1) going negative. 
    This is the model that caught a y_fwd sign bug which a simpler model (e.g. linear_chain_model) wouldn't, because the solver can satisfy it without R1 ever needing to go negative."""
    m = cobra.Model("forced_negative")
    m.compartments = {"c": "cytosol"}
    q = cobra.Metabolite("Q", compartment="c")
    g = cobra.Metabolite("G", compartment="c")
    j = cobra.Metabolite("J", compartment="c")

    ex_q = cobra.Reaction("EX_Q")
    ex_q.lower_bound, ex_q.upper_bound = -1000, 1000
    r_pos = cobra.Reaction("R_POS")  # no gene: always available, forced >0 by biomass demand
    r_pos.lower_bound, r_pos.upper_bound = 0, 1000
    biomass = cobra.Reaction("BIOMASS")
    biomass.lower_bound, biomass.upper_bound = 0, 1000
    r1 = cobra.Reaction("R1")  # HIGH expr, reversible — must go negative to balance
    r1.lower_bound, r1.upper_bound = -1000, 1000
    r1.gene_reaction_rule = "g1"
    r2 = cobra.Reaction("R2")  # HIGH expr, reversible, unrelated — control
    r2.lower_bound, r2.upper_bound = -1000, 1000
    r2.gene_reaction_rule = "g2"

    ex_q.add_metabolites({q: -1})
    r_pos.add_metabolites({q: -1, g: 1, j: 1})
    biomass.add_metabolites({g: -1})
    r1.add_metabolites({j: 1})
    r2.add_metabolites({})

    m.add_reactions([ex_q, r_pos, biomass, r1, r2])
    m.objective = "BIOMASS"
    return m


@pytest.fixture
def human1_like_model():
    """Ensembl-format gene IDs — mirrors Human1/Human2's internal format."""
    m = cobra.Model("human1_like")
    m.compartments = {"c": "cytosol"}
    a = cobra.Metabolite("A", compartment="c")
    b = cobra.Metabolite("B", compartment="c")
    c = cobra.Metabolite("C", compartment="c")

    r_ex = cobra.Reaction("MAR_EX")
    r_ex.lower_bound, r_ex.upper_bound = -1000, 1000
    r1 = cobra.Reaction("MAR001")
    r1.lower_bound, r1.upper_bound = -1000, 1000
    r1.gene_reaction_rule = "ENSG00000141510"
    r2 = cobra.Reaction("MAR002")
    r2.lower_bound, r2.upper_bound = -1000, 1000
    r2.gene_reaction_rule = "ENSG00000012048"
    biomass = cobra.Reaction("MARBIOMASS")
    biomass.lower_bound, biomass.upper_bound = 0, 1000

    r_ex.add_metabolites({a: -1})
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1, c: 1})
    biomass.add_metabolites({c: -1})
    m.add_reactions([r_ex, r1, r2, biomass])
    m.objective = "MARBIOMASS"
    return m


@pytest.fixture
def recon3d_like_model():
    """Entrez+isoform ('_AT') gene IDs — mirrors Recon3D's internal format."""
    m = cobra.Model("recon3d_like")
    m.compartments = {"c": "cytosol"}
    a = cobra.Metabolite("A", compartment="c")
    b = cobra.Metabolite("B", compartment="c")
    c = cobra.Metabolite("C", compartment="c")

    r_ex = cobra.Reaction("EX_A")
    r_ex.lower_bound, r_ex.upper_bound = -1000, 1000
    r1 = cobra.Reaction("R1")
    r1.lower_bound, r1.upper_bound = -1000, 1000
    r1.gene_reaction_rule = "7157_AT1"
    r2 = cobra.Reaction("R2")
    r2.lower_bound, r2.upper_bound = -1000, 1000
    r2.gene_reaction_rule = "672_AT1"
    biomass = cobra.Reaction("BIOMASS")
    biomass.lower_bound, biomass.upper_bound = 0, 1000

    r_ex.add_metabolites({a: -1})
    r1.add_metabolites({a: -1, b: 1})
    r2.add_metabolites({b: -1, c: 1})
    biomass.add_metabolites({c: -1})
    m.add_reactions([r_ex, r1, r2, biomass])
    m.objective = "BIOMASS"
    return m


@pytest.fixture
def gene_csv_fixture(tmp_path):
    """Synthetic DepMap Gene.csv — Entrez<->Ensembl bridge table."""
    path = tmp_path / "Gene.csv"
    pd.DataFrame({"entrez_id": [7157, 672], "ensembl_gene_id": ["ENSG00000141510", "ENSG00000012048"],}).to_csv(path, index=False)
    return path


@pytest.fixture
def tcga_matrix_fixture(tmp_path):
    """Synthetic Xena-style TCGA matrix (gzipped tsv, Ensembl IDs w/ version)."""
    path = tmp_path / "tcga_ov.tsv.gz"
    df = pd.DataFrame({"sample": ["ENSG00000141510.11", "ENSG00000012048.20", "ENSG00000999999.3"], "TCGA-25-1319-01": [5.2, 3.1, 0.4], "TCGA-25-1320-01": [6.8, 2.9, 0.1]}).set_index("sample")
    df.to_csv(path, sep="\t", compression="gzip")
    return path


@pytest.fixture
def depmap_expression_fixture(tmp_path):
    """Synthetic DepMap expression CSV (Model.csv + expression matrix)."""
    expr_path = tmp_path / "expression.csv"
    model_csv_path = tmp_path / "Model.csv"

    pd.DataFrame({"ModelID": ["ACH-000001"], "TP53 (7157)": [5.2], "CD44 (672)": [3.1],}).set_index("ModelID").reset_index().to_csv(expr_path, index=False)

    pd.DataFrame({"ModelID": ["ACH-000001"], "CellLineName": ["TESTLINE"], "StrippedCellLineName": ["TESTLINE"],}).to_csv(model_csv_path, index=False)

    return expr_path, model_csv_path


@pytest.fixture
def two_carbon_sources_model():
    """Two independent, redundant carbon sources (glucose, fructose), each sufficient alone to sustain growth. 
    Used to test that _apply_media's strict mode actually closes unlisted exchange reactions rather than just opening the ones named in the media dict.
    Bug found when a real DMEM media run produced byte-for-byte identical growth to the unconstrained baseline (fructose, unlisted, silently kept the model's original wide-open bound and bypassed the glucose cap entirely)."""
    m = cobra.Model("two_carbon")
    m.compartments = {"c": "cytosol", "e": "extracellular"}
    glc = cobra.Metabolite("glc_e", compartment="e")
    fru = cobra.Metabolite("fru_e", compartment="e")
    bp = cobra.Metabolite("bp_c", compartment="c")

    ex_glc = cobra.Reaction("EX_glucose")
    ex_glc.lower_bound, ex_glc.upper_bound = -1000, 1000
    ex_fru = cobra.Reaction("EX_fructose")
    ex_fru.lower_bound, ex_fru.upper_bound = -1000, 1000
    r_glc = cobra.Reaction("USE_GLC")
    r_glc.lower_bound, r_glc.upper_bound = 0, 1000
    r_fru = cobra.Reaction("USE_FRU")
    r_fru.lower_bound, r_fru.upper_bound = 0, 1000
    biomass = cobra.Reaction("BIOMASS")
    biomass.lower_bound, biomass.upper_bound = 0, 1000

    ex_glc.add_metabolites({glc: -1})
    ex_fru.add_metabolites({fru: -1})
    r_glc.add_metabolites({glc: -1, bp: 1})
    r_fru.add_metabolites({fru: -1, bp: 1})
    biomass.add_metabolites({bp: -1})
    m.add_reactions([ex_glc, ex_fru, r_glc, r_fru, biomass])
    m.objective = "BIOMASS"
    return m
