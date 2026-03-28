from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor


@dataclass
class RiskReport:
    filters: Dict[str, Optional[str]]
    area_risk: pd.DataFrame
    dangerous_dates_areas: pd.DataFrame
    forecast: pd.DataFrame
    predictive_heatmap: pd.DataFrame
    answers: Dict[str, object]


class SpatioTemporalRiskEngine:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.copy()
        if "Fecha" in self.frame.columns:
            self.frame["Fecha"] = pd.to_datetime(self.frame["Fecha"], errors="coerce")
        self.frame = self.frame.dropna(subset=["Fecha"]).copy()

    def _filter_frame(
        self,
        delito: Optional[str] = None,
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        view = self.frame.copy()
        if delito:
            view = view[view["Delito"] == str(delito)]
        if provincia:
            view = view[view["Provincia"] == str(provincia)]
        if canton:
            view = view[view["Canton"] == str(canton)]
        if distrito:
            view = view[view["Distrito"] == str(distrito)]
        if start_date:
            start_dt = pd.to_datetime(start_date, errors="coerce")
            if not pd.isna(start_dt):
                view = view[view["Fecha"] >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date, errors="coerce")
            if not pd.isna(end_dt):
                view = view[view["Fecha"] <= end_dt]
        return view

    @staticmethod
    def _series_to_features(index: pd.Series, base_date: pd.Timestamp) -> pd.DataFrame:
        offset = (index - base_date).dt.days.astype(float)
        dow = index.dt.dayofweek.astype(float)
        month = index.dt.month.astype(float)
        return pd.DataFrame(
            {
                "t": offset,
                "dow_sin": np.sin(2 * np.pi * dow / 7),
                "dow_cos": np.cos(2 * np.pi * dow / 7),
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
            },
            index=index.index,
        )

    def forecast_daily_incidents(
        self,
        delito: Optional[str] = None,
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        horizon_days: int = 30,
    ) -> pd.DataFrame:
        horizon_days = int(max(7, min(horizon_days, 180)))
        view = self._filter_frame(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
        )
        if view.empty:
            return pd.DataFrame(columns=["Fecha", "Forecast", "Lower80", "Upper80"])

        daily = (
            view.groupby(view["Fecha"].dt.floor("D"), observed=True)
            .size()
            .rename("Casos")
            .reset_index()
            .sort_values("Fecha")
            .reset_index(drop=True)
        )
        min_date = daily["Fecha"].min()
        max_date = daily["Fecha"].max()
        date_index = pd.date_range(min_date, max_date, freq="D")
        aligned = daily.set_index("Fecha").reindex(date_index, fill_value=0).rename_axis("Fecha").reset_index()
        y_train = aligned["Casos"].astype(float).values

        if len(aligned) < 21 or y_train.sum() == 0:
            rolling = pd.Series(y_train).rolling(7, min_periods=1).mean().iloc[-1]
            last_day = aligned["Fecha"].max()
            future_dates = pd.date_range(last_day + pd.Timedelta(days=1), periods=horizon_days, freq="D")
            baseline = float(max(rolling, 0.0))
            output = pd.DataFrame(
                {
                    "Fecha": future_dates,
                    "Forecast": baseline,
                    "Lower80": max(baseline * 0.7, 0.0),
                    "Upper80": baseline * 1.3,
                }
            )
            return output

        base_date = aligned["Fecha"].min()
        x_train = self._series_to_features(aligned["Fecha"], base_date)
        model = PoissonRegressor(alpha=0.05, max_iter=800)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        residual = y_train - train_pred
        sigma = float(np.std(residual)) if residual.size else 0.0

        last_date = aligned["Fecha"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        x_future = self._series_to_features(pd.Series(future_dates), base_date)
        forecast = np.clip(model.predict(x_future), a_min=0.0, a_max=None)
        output = pd.DataFrame({"Fecha": future_dates, "Forecast": forecast})
        output["Lower80"] = np.clip(output["Forecast"] - (1.28 * sigma), a_min=0.0, a_max=None)
        output["Upper80"] = output["Forecast"] + (1.28 * sigma)
        return output

    def most_probable_areas(
        self,
        delito: str,
        area_level: str = "Canton",
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        area_level = area_level if area_level in {"Provincia", "Canton", "Distrito"} else "Canton"
        view = self._filter_frame(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
        )
        if view.empty:
            return pd.DataFrame(columns=[area_level, "Casos", "Probabilidad"])

        counts = view[area_level].value_counts().head(max(1, int(top_n)))
        probs = counts / counts.sum()
        output = (
            pd.DataFrame({area_level: counts.index, "Casos": counts.values, "Probabilidad": probs.values})
            .sort_values("Probabilidad", ascending=False)
            .reset_index(drop=True)
        )
        return output

    def predictive_hour_weekday_heatmap(
        self,
        delito: Optional[str] = None,
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        horizon_days: int = 30,
    ) -> pd.DataFrame:
        view = self._filter_frame(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
        )
        if view.empty:
            return pd.DataFrame()

        matrix = (
            view.pivot_table(
                index="DiaSemana",
                columns="Hora",
                values="Delito",
                aggfunc="count",
                fill_value=0,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        total = matrix.to_numpy().sum()
        if total <= 0:
            return matrix

        distribution = matrix / total
        forecast = self.forecast_daily_incidents(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
        )
        mean_daily = float(forecast["Forecast"].mean()) if not forecast.empty else float(view.groupby("Fecha").size().mean())
        expected = distribution * mean_daily
        return expected

    def dangerous_dates_and_areas(
        self,
        delito: str,
        area_level: str = "Canton",
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        horizon_days: int = 30,
        top_areas: int = 5,
        top_rows: int = 20,
    ) -> pd.DataFrame:
        area_risk = self.most_probable_areas(
            delito=delito,
            area_level=area_level,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            top_n=top_areas,
        )
        forecast = self.forecast_daily_incidents(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
        )
        if area_risk.empty or forecast.empty:
            return pd.DataFrame(columns=["Fecha", area_level, "RiesgoEsperado", "ProbabilidadArea", "PronosticoDiario"])

        rows = []
        for _, date_row in forecast.iterrows():
            for _, area_row in area_risk.iterrows():
                rows.append(
                    {
                        "Fecha": date_row["Fecha"],
                        area_level: area_row[area_level],
                        "RiesgoEsperado": float(date_row["Forecast"] * area_row["Probabilidad"]),
                        "ProbabilidadArea": float(area_row["Probabilidad"]),
                        "PronosticoDiario": float(date_row["Forecast"]),
                    }
                )
        ranking = pd.DataFrame(rows).sort_values("RiesgoEsperado", ascending=False).head(max(1, int(top_rows)))
        return ranking.reset_index(drop=True)

    def build_report(
        self,
        delito: str,
        area_level: str = "Canton",
        provincia: Optional[str] = None,
        canton: Optional[str] = None,
        distrito: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        horizon_days: int = 30,
    ) -> RiskReport:
        area_risk = self.most_probable_areas(
            delito=delito,
            area_level=area_level,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            top_n=10,
        )
        forecast = self.forecast_daily_incidents(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
        )
        heatmap = self.predictive_hour_weekday_heatmap(
            delito=delito,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
        )
        dangerous = self.dangerous_dates_and_areas(
            delito=delito,
            area_level=area_level,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
            top_areas=5,
            top_rows=15,
        )

        top_area = area_risk.iloc[0].to_dict() if not area_risk.empty else {}
        top_danger = dangerous.iloc[0].to_dict() if not dangerous.empty else {}
        answers = {
            "most_probable_area_for_crime": top_area,
            "most_dangerous_date_area": top_danger,
            "forecast_total_next_horizon": float(forecast["Forecast"].sum()) if not forecast.empty else 0.0,
        }
        return RiskReport(
            filters={
                "delito": delito,
                "area_level": area_level,
                "provincia": provincia,
                "canton": canton,
                "distrito": distrito,
                "start_date": start_date,
                "end_date": end_date,
                "horizon_days": str(horizon_days),
            },
            area_risk=area_risk,
            dangerous_dates_areas=dangerous,
            forecast=forecast,
            predictive_heatmap=heatmap,
            answers=answers,
        )
