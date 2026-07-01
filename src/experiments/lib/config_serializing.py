import importlib
import json
from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from importlib.metadata import PackagePath
from typing import Any, Literal


@dataclass(frozen=True)
class OptunaOptimised:
    """Marks a config field as something to be optimised by Optuna.
    `kwargs` are forwarded to `trial.suggest_<kind>(name, **kwargs)`.
    """

    kind: Literal["float", "int", "categorical"]
    kwargs: dict


def resolved[T](value: T | OptunaOptimised) -> T:
    assert not isinstance(value, OptunaOptimised)
    return value


@dataclass(frozen=True, kw_only=True)
class ClassConfig(ABC):
    configured_class: PackagePath | None = None

    @staticmethod
    def get_configured_class(): ...

    def class_name(self) -> str:
        """Friendly class name without the whole path"""
        return str(self.configured_class).split(".")[-1]

    def post_init_checks(self): ...

    def __post_init__(self):
        object.__setattr__(
            self,
            "configured_class",
            serialize_class(self.get_configured_class()),
        )
        self.post_init_checks()

    def to_dict(self, save_class=False):
        if is_dataclass(self):
            d = {
                f.name: ClassConfig.to_dict(getattr(self, f.name), save_class)
                for f in fields(self)
            }
            if save_class:
                d |= {"__cfg__": serialize_class(type(self))}
            return d
        if isinstance(self, (list, tuple)):
            return [ClassConfig.to_dict(x, save_class) for x in self]

        if isinstance(self, Enum):
            return {"__enum__": serialize_class(type(self)), "value": self.value}
        return self

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
        return ClassConfig._flatten_dict(ClassConfig.to_dict(self))

    def build(self):
        self.post_init_checks()
        if not self.configured_class:
            raise Exception("fConfigured class not set (={self.configured_class}).")
        return deserialize_class(self.configured_class)(self)

    @staticmethod
    def from_dict(d, _first_recursion=True):
        if isinstance(d, dict) and "__cfg__" in d:
            # Then it's a ClassConfig dict that can be deserialized into a ClassConfig
            cls = deserialize_class(d["__cfg__"])
            kwds = {}
            for k, v in d.items():
                if k == "__cfg__":
                    # skip the __cfg__ field only useful for deserialising
                    continue
                kwds[k] = ClassConfig.from_dict(v, _first_recursion=False)
            return cls(**kwds)

        # If its the first recursion and the dict has no __cfg__ then it must
        # have been called on an invalid dict that cannot be used to instantiate
        # a ClassConfig
        assert not _first_recursion

        if isinstance(d, (list, tuple)):
            return [ClassConfig.from_dict(x, _first_recursion=False) for x in d]

        if isinstance(d, dict) and "__enum__" in d:
            return deserialize_class(d["__enum__"])(d["value"])
        return d

    def to_json(self, save_class=False):
        return json.dumps(ClassConfig.to_dict(self, save_class))

    @staticmethod
    def from_json(s) -> ClassConfig:
        c = ClassConfig.from_dict(json.loads(s))
        assert isinstance(c, ClassConfig)
        return c


def deserialize_class(path: str | None) -> Any:
    """'package.module.ClassName' -> object"""
    if path is None:
        return None
    module_path, _, attr_name = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def serialize_class(cls) -> str | None:
    """object -> 'package.module.ClassName'"""
    if cls is None:
        return None
    return must_serialize_class(cls)


def must_serialize_class(cls) -> str:
    assert cls is not None
    return f"{cls.__module__}.{cls.__qualname__}"
