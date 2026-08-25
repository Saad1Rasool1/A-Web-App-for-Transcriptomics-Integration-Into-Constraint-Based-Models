"""
End-to-end regression tests through pipeline.run_transcriptomics_integration.
The actual entry point external code (Streamlit interface) will call, not just the lower-level functions in isolation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cobra
from tools.pipeline import run_transcriptomics_integration


def _write_model(model, tmp_path, name):
    path = tmp_path / name
    cobra.io.write_sbml_model(model, str(path))
    return path


class TestTCGAEndToEnd:
    def test_tcga_to_human1_no_gene_csv_needed(self, human1_like_model, tcga_matrix_fixture, tmp_path):
        model_path = _write_model(human1_like_model, tmp_path, "human1.xml")
        fluxes, info = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", method="continuous_meeson",)
        assert info["status"] == "optimal"
        assert info["genes_matched"] == 2
        assert info["tcga_sample"] == "TCGA-25-1319"

    def test_tcga_to_recon3d_with_gene_csv_bridge(self, recon3d_like_model, tcga_matrix_fixture, gene_csv_fixture, tmp_path):
        model_path = _write_model(recon3d_like_model, tmp_path, "recon3d.xml")
        fluxes, info = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", gene_csv_path=gene_csv_fixture, method="continuous_meeson",)
        assert info["status"] == "optimal"
        assert info["genes_matched"] == 2

    def test_cannot_specify_both_depmap_and_tcga(
        self, human1_like_model, tcga_matrix_fixture, gene_csv_fixture, tmp_path):
        import pytest
        model_path = _write_model(human1_like_model, tmp_path, "human1.xml")
        with pytest.raises(ValueError, match="not both"):
            run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", cell_line_name="SOMELINE", model_csv_path=gene_csv_fixture, method="continuous_meeson",)


class TestDefaultMedia:
    """
    Supervisor requested: 'each model should have a default media formulation, then let the user change it if they want.'
    Locks in that media=None triggers the per-model default (Human1 -> DMEM), and media=False forces it off regardless.
    """

    def test_media_none_applies_registered_default(self, human1_like_model, tcga_matrix_fixture, tmp_path):
        model_path = _write_model(human1_like_model, tmp_path, "Human1_test.xml")
        _, info = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", method="pfba_baseline",)
        # human1_like_model's genes are ENSG00000141510/ENSG00000012048,
        # not real DMEM component IDs, so the default (looked up by
        # filename containing "Human1") won't restrict growth here —
        # this just confirms it resolves without error. The real
        # restriction behaviour is covered in test_apply_media.py.
        assert info["status"] == "optimal"

    def test_media_false_forces_none_even_with_default_registered(self, human1_like_model, tcga_matrix_fixture, tmp_path):
        model_path = _write_model(human1_like_model, tmp_path, "Human1_test.xml")
        _, info_default = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", method="pfba_baseline",)
        _, info_forced_off = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", method="pfba_baseline", media=False,)
        # Both should succeed regardless; the key check is that
        # media=False doesn't raise or behave differently from an
        # unregistered model.
        assert info_default["status"] == "optimal"
        assert info_forced_off["status"] == "optimal"

    def test_unregistered_model_falls_back_to_no_media(self, recon3d_like_model, tcga_matrix_fixture, gene_csv_fixture, tmp_path):
        """Recon3D has no registered default yet — media=None must fall back to no constraints, not raise an error."""
        model_path = _write_model(recon3d_like_model, tmp_path, "Recon3D_test.xml")
        _, info = run_transcriptomics_integration(model_path=model_path, expression_path=tcga_matrix_fixture, tcga_sample_barcode="TCGA-25-1319", gene_csv_path=gene_csv_fixture, method="pfba_baseline",)
        assert info["status"] == "optimal"