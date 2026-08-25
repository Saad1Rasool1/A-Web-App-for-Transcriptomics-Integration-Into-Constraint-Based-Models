"""
Regression tests for _apply_media (integrate_omics.py).

Fix for: media only ever SET bounds for reactions explicitly listed in the media dict, never CLOSED anything.
A media dict listing only glucose left every other exchange reaction (e.g. fructose) at its original wide-open bound, silently bypassing the intended nutrient restriction entirely. 
Caught when a real DMEM run produced growth byte-for-byte identical to the unconstrained baseline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.integrate_omics import run_pfba_baseline


class TestApplyMediaStrictMode:
    def test_unlisted_alternative_exchange_is_closed(self, two_carbon_sources_model):
        """Fructose isn't in the media dict, but strict mode must still close it so glucose's cap actually binds."""
        media = {"EX_glucose": 10.0}
        _, info = run_pfba_baseline(two_carbon_sources_model, media=media)
        assert info["predicted_growth"] == 10.0

    def test_growth_drops_relative_to_unconstrained_baseline(self, two_carbon_sources_model):
        _, unconstrained = run_pfba_baseline(two_carbon_sources_model)
        _, constrained = run_pfba_baseline(two_carbon_sources_model, media={"EX_glucose": 10.0})
        assert constrained["predicted_growth"] < unconstrained["predicted_growth"]