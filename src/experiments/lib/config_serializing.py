from abc import ABC
import importlib
from importlib.metadata import PackagePath
from typing import Any
from dataclasses import dataclass, fields, is_dataclass


@dataclass(frozen=True, kw_only=True)
class ClassConfig(ABC):
    configured_class: PackagePath | None = None

    @staticmethod
    def get_configured_class(): ...

    def post_init_checks(self): ...

    def __post_init__(self):
        object.__setattr__(
            self,
            "configured_class",
            serialize_class(self.get_configured_class()),
        )

    def _config_to_dict(config):
        if is_dataclass(config):
            return {
                f.name: config._config_to_dict(getattr(config, f.name))
                for f in fields(config)
            }
        if isinstance(config, (list, tuple)):
            return list(config)
        return config

    @staticmethod
    def _flatten_dict(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if v is None:
                continue  # optional params skipped instead of logging lots of None
            if isinstance(v, dict):
                out.update(ClassConfig._flatten_dict(v, key + "."))
            else:
                out[key] = v
        return out

    def serialize_config(self):
        return ClassConfig._flatten_dict(ClassConfig._config_to_dict(self))

    def build(self):
        self.post_init_checks()
        if not self.configured_class:
            raise Exception("fConfigured class not set (={self.configured_class}).")
        return deserialize_class(self.configured_class)(self)


def deserialize_class(path: PackagePath) -> Any:
    """'package.module.ClassName' -> object"""
    if path is None:
        return None
    module_path, _, attr_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def serialize_class(cls) -> PackagePath:
    """object -> 'package.module.ClassName'"""
    if cls is None:
        return None
    return f"{cls.__module__}.{cls.__qualname__}"
