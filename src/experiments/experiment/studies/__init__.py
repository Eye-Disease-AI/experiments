from optuna import Study

from experiments.experiment.studies._names import StudyType
from experiments.experiment.studies.convnext_study import ConvNextStudy
from experiments.experiment.studies.study import StudyConfig

_StudyTypes: dict[StudyType, Study] = {
    "ConvnextStudy": ConvNextStudy
}

def init_study(config: StudyConfig) -> Study:
    """For initializing studies using logged configs"""
    return _StudyTypes[config.name](config)
