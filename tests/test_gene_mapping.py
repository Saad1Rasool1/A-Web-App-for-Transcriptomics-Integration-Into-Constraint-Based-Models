"""
Regression tests for gene_mapping.py.

Locks in the fix for: Ensembl-format input (e.g. TCGA) was always being
converted to HGNC symbols regardless of the target model's actual
internal gene ID format, silently producing ~0% coverage against
Human1/Human2 (which are Ensembl-format) and Recon3D (Entrez-format).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.gene_mapping import normalise_to_model_ids
import pandas as pd


def _tcga_expr():
    return pd.Series({"ENSG00000141510.11": 5.2, "ENSG00000012048.20": 3.1, "ENSG00000999999.1": 0.1},  # absent from both fixture models
        name="expr",)


class TestEnsemblInputRouting:
    """The core bug: Ensembl input must route by the MODEL's format, not always convert to HGNC."""

    def test_ensembl_to_ensembl_model_direct_match_no_mapping_file(self, human1_like_model):
        """TCGA -> Human1/Human2: must match directly, no gene_csv_path or ensembl_mapping_path required at all."""
        result = normalise_to_model_ids(_tcga_expr(), human1_like_model, id_format="ensembl")
        assert len(result) == 2
        assert set(result.index) == {"ENSG00000141510", "ENSG00000012048"}
        assert result["ENSG00000141510"] == 5.2

    def test_ensembl_to_recon3d_via_gene_csv_bridge(self, recon3d_like_model, gene_csv_fixture):
        """TCGA -> Recon3D: bridges through gene_csv_path (reuses DepMap's Gene.csv, no separate BioMart download needed)."""
        result = normalise_to_model_ids(_tcga_expr(), recon3d_like_model, id_format="ensembl", gene_csv_path=gene_csv_fixture,)
        assert len(result) == 2
        assert set(result.index) == {"7157_AT1", "672_AT1"}

    def test_ensembl_to_recon3d_without_any_bridge_raises_clear_error(self, recon3d_like_model):
        """Without gene_csv_path or ensembl_mapping_path, must fail loudly rather than silently produce near-zero coverage."""
        import pytest
        with pytest.raises(ValueError, match="Ensembl -> Entrez"):
            normalise_to_model_ids(_tcga_expr(), recon3d_like_model, id_format="ensembl",)

    def test_version_suffix_stripped_correctly(self, human1_like_model):
        """'ENSG00000141510.11' must match model gene 'ENSG00000141510'."""
        result = normalise_to_model_ids(_tcga_expr(), human1_like_model, id_format="ensembl")
        assert "ENSG00000141510.11" not in result.index
        assert "ENSG00000141510" in result.index


class TestDepmapInputRouting:
    """Sanity check the pre-existing DepMap path wasn't disturbed by the Ensembl-routing changes."""

    def test_depmap_to_human1_via_gene_csv(self, human1_like_model, gene_csv_fixture):
        depmap_expr = pd.Series({"TP53 (7157)": 5.2, "CD44 (672)": 3.1})
        result = normalise_to_model_ids(depmap_expr, human1_like_model, id_format="depmap", gene_csv_path=gene_csv_fixture,)
        assert len(result) == 2
        assert set(result.index) == {"ENSG00000141510", "ENSG00000012048"}

    def test_depmap_to_recon3d_direct(self, recon3d_like_model):
        depmap_expr = pd.Series({"TP53 (7157)": 5.2, "CD44 (672)": 3.1})
        result = normalise_to_model_ids(depmap_expr, recon3d_like_model, id_format="depmap",)
        assert len(result) == 2
        assert set(result.index) == {"7157_AT1", "672_AT1"}
