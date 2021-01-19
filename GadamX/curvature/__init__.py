import torchvision
#if torchvision.__version__ <= "0.2.1":
#   print("Older torchvision found")
#   from . import data as data
#else:
from. import data2 as data

from . import (
    methods,
    models,
    losses,
    utils,
)

__all__ = [
    'methods',
    'models',
    'data',
    'losses',
    'utils',
]
