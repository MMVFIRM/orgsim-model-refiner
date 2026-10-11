"""OrgSim × Model Refiner: scoped, active, executable policy learning.

The only ERP client is `erp.dolibarr_live`: a loopback-only oracle for DISPOSABLE test
tenants that creates real records there (never point it at production). Model-provider
integration is optional; the core learner makes no LLM calls.
"""
__version__ = '0.8.0'
from .types import Scope, Field, Schema, Observation
from .learner import ActivePolicyLearner, LearnConfig
from .artifact import PolicyArtifact
__all__ = ['Scope','Field','Schema','Observation','ActivePolicyLearner','LearnConfig','PolicyArtifact']
