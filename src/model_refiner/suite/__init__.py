"""Frozen black-box adaptation suite. Provider weights remain unchanged."""
from .artifact import Recipe,SuiteArtifact
from .config import ReleasePolicy,TrainConfig
from .dataset import Dataset,Preference,Record,load_records
from .runtime import SuiteRuntime
from .tasks import TaskSpec
from .training import TrainResult,plan_fit,train
from .weights import ArtificialWeights
from .sdk import fit

__all__=["Recipe","SuiteArtifact","ReleasePolicy","TrainConfig","Dataset","Preference","Record","load_records",
         "SuiteRuntime","TaskSpec","TrainResult","plan_fit","train","ArtificialWeights","fit"]
