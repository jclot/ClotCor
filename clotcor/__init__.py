NAME = "clotcor"

__all__ = ["NAME", "Prediccion"]


def __getattr__(name):
    if name == "Prediccion":
        from clotcor.base import Prediccion

        return Prediccion
    raise AttributeError(f"module 'clotcor' has no attribute '{name}'")
