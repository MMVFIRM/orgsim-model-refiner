"""OrgSim: deterministic enterprise digital-twin simulation infrastructure."""
from .engine import SimulationEngine, SimError
from .spec import load_spec, validate_spec

__version__ = "0.2.4"
__all__ = ["SimulationEngine", "SimError", "load_spec", "validate_spec"]
