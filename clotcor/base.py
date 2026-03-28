"""Backward-compatible exports for legacy imports."""

from clotcor.modeling.predictor import Prediccion, PredictionPayload

NAME = "clotcor"

__all__ = ["NAME", "Prediccion", "PredictionPayload"]
