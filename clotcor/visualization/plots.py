from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CrimePlotFactory:
    def __init__(self) -> None:
        sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fb", "figure.facecolor": "#f8f9fb"})

    @staticmethod
    def _safe_barplot(data: pd.DataFrame, x: str, y: str, palette: str, axis) -> None:
        # Seaborn >=0.14 deprecates palette without hue.
        sns.barplot(data=data, x=x, y=y, hue=y, palette=palette, dodge=False, legend=False, ax=axis)

    def confusion_matrix_figure(self, matrix: np.ndarray, labels: Sequence[str]) -> plt.Figure:
        figure, axis = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            ax=axis,
        )
        axis.set_title("Normalized Confusion Matrix", fontsize=13, fontweight="bold")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        figure.tight_layout()
        return figure

    def feature_importance_figure(self, feature_importance: pd.DataFrame, top_n: int = 15) -> plt.Figure:
        ranking = feature_importance.sort_values("importance", ascending=False).head(top_n).copy()
        figure, axis = plt.subplots(figsize=(10, 7))
        self._safe_barplot(ranking, x="importance", y="feature", palette="crest", axis=axis)
        axis.set_title("Feature Importance", fontsize=13, fontweight="bold")
        axis.set_xlabel("Relative Importance")
        axis.set_ylabel("Feature")
        figure.tight_layout()
        return figure

    def probabilities_figure(self, labels: Iterable[str], scores: Iterable[float]) -> plt.Figure:
        labels_list = list(labels)
        scores_list = list(scores)
        plot_frame = pd.DataFrame({"Delito": labels_list, "Probabilidad": scores_list})
        figure, axis = plt.subplots(figsize=(8, 4))
        self._safe_barplot(plot_frame, x="Probabilidad", y="Delito", palette="mako", axis=axis)
        axis.set_xlim(0, 1)
        axis.set_title("Top Class Probabilities", fontsize=12, fontweight="bold")
        axis.set_xlabel("Probability")
        axis.set_ylabel("Crime")
        for index, value in enumerate(scores_list):
            axis.text(min(value + 0.01, 0.98), index, f"{value:.2%}", va="center", fontsize=9)
        figure.tight_layout()
        return figure

    def trend_figure(self, trend_frame: pd.DataFrame) -> plt.Figure:
        figure, axis = plt.subplots(figsize=(10, 4))
        if trend_frame.empty:
            axis.text(0.5, 0.5, "No temporal data available", ha="center", va="center")
            axis.axis("off")
            return figure
        sns.lineplot(data=trend_frame, x="Periodo", y="Casos", marker="o", color="#0f766e", ax=axis)
        sns.lineplot(
            data=trend_frame,
            x="Periodo",
            y="PromedioMovil3",
            color="#ff8800",
            linewidth=2.2,
            ax=axis,
        )
        axis.set_title("Monthly Trend and 3-Month Moving Average", fontsize=12, fontweight="bold")
        axis.set_xlabel("Period")
        axis.set_ylabel("Cases")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        return figure

    def forecast_figure(self, forecast_frame: pd.DataFrame) -> plt.Figure:
        figure, axis = plt.subplots(figsize=(10, 4))
        if forecast_frame.empty:
            axis.text(0.5, 0.5, "No forecast data available", ha="center", va="center")
            axis.axis("off")
            return figure
        sns.lineplot(data=forecast_frame, x="Fecha", y="Forecast", marker="o", color="#1d4ed8", ax=axis)
        if {"Lower80", "Upper80"}.issubset(forecast_frame.columns):
            axis.fill_between(
                forecast_frame["Fecha"],
                forecast_frame["Lower80"],
                forecast_frame["Upper80"],
                color="#93c5fd",
                alpha=0.35,
                label="80% band",
            )
            axis.legend(loc="upper left")
        axis.set_title("Daily Forecast (Future Horizon)", fontsize=12, fontweight="bold")
        axis.set_xlabel("Date")
        axis.set_ylabel("Expected cases")
        axis.tick_params(axis="x", rotation=35)
        figure.tight_layout()
        return figure

    def area_risk_figure(self, area_frame: pd.DataFrame, area_column: str) -> plt.Figure:
        figure, axis = plt.subplots(figsize=(8, 4))
        if area_frame.empty:
            axis.text(0.5, 0.5, "No area risk data available", ha="center", va="center")
            axis.axis("off")
            return figure
        plot_frame = area_frame.copy().head(10)
        self._safe_barplot(plot_frame, x="Probabilidad", y=area_column, palette="viridis", axis=axis)
        axis.set_xlim(0, min(1.0, max(0.1, plot_frame["Probabilidad"].max() * 1.2)))
        axis.set_title(f"Most Probable {area_column} for Selected Crime", fontsize=12, fontweight="bold")
        axis.set_xlabel("Probability")
        axis.set_ylabel(area_column)
        figure.tight_layout()
        return figure

    def hour_weekday_heatmap_figure(self, matrix: pd.DataFrame, title: str = "Hour/Weekday Heatmap") -> plt.Figure:
        figure, axis = plt.subplots(figsize=(10, 4))
        if matrix.empty:
            axis.text(0.5, 0.5, "No heatmap data available", ha="center", va="center")
            axis.axis("off")
            return figure
        day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        index_labels = [day_labels[idx] if idx < len(day_labels) else str(idx) for idx in matrix.index]
        sns.heatmap(matrix, cmap="rocket_r", linewidths=0.2, linecolor="#ffffff", ax=axis)
        axis.set_title(title, fontsize=12, fontweight="bold")
        axis.set_xlabel("Hour")
        axis.set_ylabel("Day")
        axis.set_yticklabels(index_labels, rotation=0)
        figure.tight_layout()
        return figure
