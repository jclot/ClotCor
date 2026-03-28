import html
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    categorical: List[str]
    numeric: List[str]
    model_input: List[str]
    user_input: List[str]


class CrimeDataPreprocessor:
    def __init__(self) -> None:
        self.feature_spec = FeatureSpec(
            target="Delito",
            categorical=[
                "SubDelito",
                "Victima",
                "SubVictima",
                "Edad",
                "Sexo",
                "Nacionalidad",
                "Provincia",
                "Canton",
                "Distrito",
            ],
            numeric=[
                "Hora",
                "DiaSemana",
                "Mes",
                "FinDeSemana",
                "HoraSin",
                "HoraCos",
                "MesSin",
                "MesCos",
            ],
            model_input=[
                "SubDelito",
                "Victima",
                "SubVictima",
                "Edad",
                "Sexo",
                "Nacionalidad",
                "Provincia",
                "Canton",
                "Distrito",
                "Hora",
                "DiaSemana",
                "Mes",
                "FinDeSemana",
                "HoraSin",
                "HoraCos",
                "MesSin",
                "MesCos",
            ],
            user_input=[
                "SubDelito",
                "Victima",
                "SubVictima",
                "Edad",
                "Sexo",
                "Nacionalidad",
                "Provincia",
                "Canton",
                "Distrito",
                "Hora",
                "DiaSemana",
                "Mes",
            ],
        )

    @staticmethod
    def _parse_hour(value: object) -> int:
        if pd.isna(value):
            return 0
        if isinstance(value, (int, float)):
            return int(np.clip(int(value), 0, 23))
        raw = str(value).strip()
        if raw == "":
            return 0
        match = re.search(r"\d{1,2}", raw)
        if match is None:
            return 0
        return int(np.clip(int(match.group(0)), 0, 23))

    @staticmethod
    def _to_int_series(values: pd.Series, lower: int, upper: int, default: int) -> pd.Series:
        casted = pd.to_numeric(values, errors="coerce").fillna(default).astype(int)
        return casted.clip(lower=lower, upper=upper)

    @staticmethod
    def _clean_text(series: pd.Series) -> pd.Series:
        return (
            series.fillna("DESCONOCIDO")
            .astype(str)
            .map(html.unescape)
            .str.strip()
            .replace("", "DESCONOCIDO")
        )

    def clean_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        frame = raw_frame.copy()
        frame = frame.loc[:, ~frame.columns.astype(str).str.startswith("Unnamed")]

        if "Fecha" in frame.columns:
            frame["Fecha"] = pd.to_datetime(frame["Fecha"], errors="coerce")
        else:
            frame["Fecha"] = pd.NaT

        if "Hora" not in frame.columns:
            frame["Hora"] = 0
        frame["Hora"] = frame["Hora"].apply(self._parse_hour)

        if "DiaSemana" not in frame.columns or frame["DiaSemana"].isna().all():
            frame["DiaSemana"] = frame["Fecha"].dt.dayofweek
        if "Mes" not in frame.columns or frame["Mes"].isna().all():
            frame["Mes"] = frame["Fecha"].dt.month

        frame["DiaSemana"] = self._to_int_series(frame["DiaSemana"], lower=0, upper=6, default=0)
        frame["Mes"] = self._to_int_series(frame["Mes"], lower=1, upper=12, default=1)
        frame["FinDeSemana"] = frame["DiaSemana"].isin([5, 6]).astype(int)
        frame["HoraSin"] = np.sin(2 * np.pi * frame["Hora"] / 24)
        frame["HoraCos"] = np.cos(2 * np.pi * frame["Hora"] / 24)
        frame["MesSin"] = np.sin(2 * np.pi * frame["Mes"] / 12)
        frame["MesCos"] = np.cos(2 * np.pi * frame["Mes"] / 12)

        for column in self.feature_spec.categorical:
            if column not in frame.columns:
                frame[column] = "DESCONOCIDO"
            frame[column] = self._clean_text(frame[column])

        if self.feature_spec.target in frame.columns:
            frame[self.feature_spec.target] = self._clean_text(frame[self.feature_spec.target])

        return frame

    def prepare_training_data(self, raw_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        clean_frame = self.clean_frame(raw_frame)
        target_name = self.feature_spec.target
        clean_frame = clean_frame[clean_frame[target_name].notna()].copy()
        x_data = clean_frame[self.feature_spec.model_input].copy()
        y_data = clean_frame[target_name].copy()
        return x_data, y_data, clean_frame

    def prepare_inference_data(self, payload: Dict[str, object] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(payload, dict):
            raw = pd.DataFrame([payload])
        else:
            raw = payload.copy()
        clean_frame = self.clean_frame(raw)
        for column in self.feature_spec.model_input:
            if column not in clean_frame.columns:
                clean_frame[column] = 0 if column in self.feature_spec.numeric else "DESCONOCIDO"
        return clean_frame[self.feature_spec.model_input].copy()
