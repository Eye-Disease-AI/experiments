from enum import Enum, StrEnum


class T(StrEnum):
    a = "asd"
    b = "fdsf"


c = T.a

print(isinstance(c, Enum))
