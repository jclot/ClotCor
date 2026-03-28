from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class StatisticalAnalyzer:
    frame: pd.DataFrame

    def class_distribution(self) -> pd.DataFrame:
        counts = self.frame["Delito"].value_counts().rename_axis("Delito").reset_index(name="Casos")
        counts["Porcentaje"] = (counts["Casos"] / counts["Casos"].sum()) * 100
        return counts

    def temporal_trend(self) -> pd.DataFrame:
        if "Fecha" not in self.frame.columns:
            return pd.DataFrame(columns=["Periodo", "Casos"])
        timeline = self.frame.dropna(subset=["Fecha"]).copy()
        if timeline.empty:
            return pd.DataFrame(columns=["Periodo", "Casos"])
        timeline["Periodo"] = timeline["Fecha"].dt.to_period("M").dt.to_timestamp()
        trend = timeline.groupby("Periodo").size().reset_index(name="Casos")
        trend["PromedioMovil3"] = trend["Casos"].rolling(window=3, min_periods=1).mean()
        return trend

    def hour_weekday_heatmap(self) -> pd.DataFrame:
        if self.frame.empty:
            return pd.DataFrame()
        heat = (
            self.frame.pivot_table(
                index="DiaSemana",
                columns="Hora",
                values="Delito",
                aggfunc="count",
                fill_value=0,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        return heat

    def province_hotspots(self, top_n: int = 10) -> pd.DataFrame:
        ranking = self.frame["Provincia"].value_counts().head(top_n)
        return ranking.rename_axis("Provincia").reset_index(name="Casos")

    def summary(self) -> Dict[str, pd.DataFrame]:
        return {
            "class_distribution": self.class_distribution(),
            "temporal_trend": self.temporal_trend(),
            "hour_weekday_heatmap": self.hour_weekday_heatmap(),
            "province_hotspots": self.province_hotspots(),
        }
