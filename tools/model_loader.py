"""
Load and cache GEM models from XML/SBML files.
Supports Human1, Human2, Recon3D, and any SBML-compatible model.
"""
import cobra
from pathlib import Path

_cache: dict[str, cobra.Model] = {}


def load_model(path: str | Path, use_cache: bool = True) -> cobra.Model:
    """
    Load a GEM from an SBML/XML file.

    Parameters
    ----------
    path : str or Path
        Path to .xml or .sbml model file.
    use_cache : bool
        Return cached model if already loaded. Avoids reloading a
        13,000-reaction model on every function call.

    Returns
    -------
    cobra.Model
    """
    path = str(Path(path).resolve())

    if use_cache and path in _cache:
        print(f"[model_loader] Using cached: {Path(path).name}")
        return _cache[path]

    print(f"[model_loader] Loading: {Path(path).name} ...")
    model = cobra.io.read_sbml_model(path)
    print(f"  Reactions:   {len(model.reactions)}")
    print(f"  Metabolites: {len(model.metabolites)}")
    print(f"  Genes:       {len(model.genes)}")

    if use_cache:
        _cache[path] = model

    return model


def clear_model_cache() -> None:
    """Free memory held by cached models."""
    _cache.clear()


def get_biomass_reaction_id(model: cobra.Model) -> str:
    """
    Return the ID of the reaction set as the model objective (biomass).
    Works across Human1 (MAR13082), Human2, Recon3D etc. automatically.

    Raises ValueError if no objective reaction is found.
    """
    for rxn in model.reactions:
        if rxn.objective_coefficient != 0:
            return rxn.id
    raise ValueError("No objective reaction found. Ensure the model has a biomass "
                     "reaction set as its objective.")
