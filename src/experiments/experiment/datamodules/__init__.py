from .datamodule import DataModule, DataModuleConfig
from .default_nuclear_cataract import NuclearCataractDataModule
from ._names import DataModuleType

_DatamoduleTypes: dict[DataModuleType, DataModule] = {
    "DefaultNuclearCataractDatamodule": NuclearCataractDataModule
}

def init_datamodule(config: DataModuleConfig) -> DataModule:
    """For initializing datamodules using logged configs"""
    return _DatamoduleTypes[config.name](config)
