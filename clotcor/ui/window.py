import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:  # pragma: no cover
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clotcor.modeling import Prediccion


APP_STYLESHEET = """
QMainWindow, QDialog, QWidget {
    background: #f1f5f9;
    color: #0f172a;
}
QGroupBox {
    font-weight: 600;
    border: 1px solid #cbd5e1;
    margin-top: 10px;
    padding-top: 8px;
    background: #ffffff;
    color: #0f172a;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #0f172a;
}
QLabel {
    color: #0f172a;
}
QPushButton {
    background: #0f766e;
    color: white;
    border: 1px solid #0f766e;
    border-radius: 6px;
    padding: 8px 10px;
    font-weight: 600;
}
QPushButton:hover {
    background: #0d9488;
    border-color: #0d9488;
}
QPushButton:pressed {
    background: #0f766e;
}
QComboBox, QLineEdit, QSpinBox, QTableWidget, QTextEdit {
    background: #f8fafc;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px;
}
QComboBox QAbstractItemView {
    background: #ffffff;
    color: #0f172a;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QHeaderView::section {
    background: #e2e8f0;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    padding: 6px;
    font-weight: 600;
}
QTableWidget {
    gridline-color: #e2e8f0;
    selection-background-color: #dbeafe;
    selection-color: #0f172a;
}
QTabWidget::pane {
    border: 1px solid #cbd5e1;
    background: #ffffff;
}
QTabBar::tab {
    background: #e2e8f0;
    color: #0f172a;
    padding: 8px 12px;
    border: 1px solid #cbd5e1;
    border-bottom: none;
    min-width: 120px;
}
QTabBar::tab:selected {
    background: #ffffff;
    font-weight: 700;
}
QToolTip {
    background: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
}
"""


class FigurePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._canvas: Optional[FigureCanvas] = None

    def set_figure(self, figure) -> None:
        if self._canvas is not None:
            self._layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
        self._canvas = FigureCanvas(figure)
        self._layout.addWidget(self._canvas)
        self._canvas.draw()
        plt.close(figure)


class DashboardDialog(QDialog):
    def __init__(self, predictor: Prediccion, parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.setWindowTitle("ClotCor | Analytics Dashboard")
        self.resize(1220, 820)

        root = QVBoxLayout(self)
        intro = QLabel(
            "This dashboard explains historical behavior, model quality, and short-term risk projections.\n"
            "Use it to understand trends before taking decisions."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("padding: 8px; background: #eef2ff; border: 1px solid #c7d2fe;")
        root.addWidget(intro)

        tabs = QTabWidget()
        root.addWidget(tabs, 1)

        figures = self.predictor.get_dashboard_figures()
        descriptions = {
            "Trend": "Historical monthly case volume and moving average.",
            "Hour / Day": "Where incidents concentrate by hour and weekday.",
            "Confusion": "How the model confuses classes (normalized by actual class).",
            "Importance": "Which variables contribute most to class separation.",
            "Forecast": "Projected daily incident volume for the next horizon.",
            "Predictive Heatmap": "Expected future concentration by hour and weekday.",
            "Area Risk": "Most probable areas for selected crime behavior.",
        }
        tab_mapping = [
            ("Trend", figures["trend"]),
            ("Hour / Day", figures["hour_weekday_heatmap"]),
            ("Confusion", figures["confusion"]),
            ("Importance", figures["feature_importance"]),
            ("Forecast", figures["forecast"]),
            ("Predictive Heatmap", figures["predictive_heatmap"]),
            ("Area Risk", figures["area_risk"]),
        ]
        for title, figure in tab_mapping:
            page = QWidget()
            page_layout = QVBoxLayout(page)
            label = QLabel(descriptions[title])
            label.setWordWrap(True)
            label.setStyleSheet("color: #334155;")
            page_layout.addWidget(label)
            canvas = FigurePanel()
            canvas.set_figure(figure)
            page_layout.addWidget(canvas, 1)
            tabs.addTab(page, title)


class FutureRiskDialog(QDialog):
    def __init__(self, predictor: Prediccion, defaults: Dict[str, str], parent=None):
        super().__init__(parent)
        self.predictor = predictor
        self.defaults = defaults
        self.setWindowTitle("ClotCor | Future Risk Assistant")
        self.resize(1240, 840)
        self._build_ui()
        self.compute_report()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        help_text = QLabel(
            "Goal: identify where and when risk is expected to be higher in the near future.\n"
            "Step 1: choose a crime type. Step 2: optionally filter area/date. Step 3: click 'Compute Risk Report'."
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("padding: 8px; background: #fef9c3; border: 1px solid #fde68a;")
        root.addWidget(help_text)

        controls = QGroupBox("Filters")
        controls_layout = QGridLayout(controls)
        root.addWidget(controls)

        delitos = self.predictor.get_available_crimes()
        self.crime_combo = QComboBox()
        self.crime_combo.addItems(delitos)
        self.area_level_combo = QComboBox()
        self.area_level_combo.addItems(["Provincia", "Canton", "Distrito"])
        self.province_edit = QLineEdit(self.defaults.get("Provincia", ""))
        self.canton_edit = QLineEdit("")
        self.district_edit = QLineEdit("")
        self.start_date_edit = QLineEdit("")
        self.end_date_edit = QLineEdit("")
        self.horizon_spin = QSpinBox()
        self.horizon_spin.setRange(7, 180)
        self.horizon_spin.setValue(30)

        controls_layout.addWidget(QLabel("Crime"), 0, 0)
        controls_layout.addWidget(self.crime_combo, 0, 1)
        controls_layout.addWidget(QLabel("Area level"), 0, 2)
        controls_layout.addWidget(self.area_level_combo, 0, 3)

        controls_layout.addWidget(QLabel("Province"), 1, 0)
        controls_layout.addWidget(self.province_edit, 1, 1)
        controls_layout.addWidget(QLabel("Canton"), 1, 2)
        controls_layout.addWidget(self.canton_edit, 1, 3)

        controls_layout.addWidget(QLabel("District"), 2, 0)
        controls_layout.addWidget(self.district_edit, 2, 1)
        controls_layout.addWidget(QLabel("Start date (YYYY-MM-DD)"), 2, 2)
        controls_layout.addWidget(self.start_date_edit, 2, 3)

        controls_layout.addWidget(QLabel("End date (YYYY-MM-DD)"), 3, 0)
        controls_layout.addWidget(self.end_date_edit, 3, 1)
        controls_layout.addWidget(QLabel("Horizon (days)"), 3, 2)
        controls_layout.addWidget(self.horizon_spin, 3, 3)

        self.compute_button = QPushButton("Compute Risk Report")
        self.compute_button.clicked.connect(self.compute_report)
        controls_layout.addWidget(self.compute_button, 4, 0, 1, 4)

        answers_box = QGroupBox("Direct Answers")
        answers_layout = QVBoxLayout(answers_box)
        self.answer_area = QLabel("Most probable area: --")
        self.answer_danger = QLabel("Most dangerous date-area: --")
        self.answer_forecast = QLabel("Expected total incidents in horizon: --")
        for label in (self.answer_area, self.answer_danger, self.answer_forecast):
            label.setWordWrap(True)
            answers_layout.addWidget(label)
        root.addWidget(answers_box)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.forecast_panel = FigurePanel()
        self.heatmap_panel = FigurePanel()
        self.area_panel = FigurePanel()
        self.dangerous_table = QTableWidget(0, 3)
        self.dangerous_table.setHorizontalHeaderLabels(["Date", "Area", "Expected Risk"])
        self.dangerous_table.horizontalHeader().setStretchLastSection(True)

        for title, widget in [
            ("Forecast", self.forecast_panel),
            ("Predictive Heatmap", self.heatmap_panel),
            ("Area Risk", self.area_panel),
            ("Dangerous Date-Area", self.dangerous_table),
        ]:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.addWidget(widget)
            self.tabs.addTab(page, title)

    @staticmethod
    def _blank_to_none(value: str) -> Optional[str]:
        value = value.strip()
        return value if value else None

    def compute_report(self) -> None:
        try:
            report = self.predictor.get_spatiotemporal_report(
                delito=self.crime_combo.currentText(),
                area_level=self.area_level_combo.currentText(),
                provincia=self._blank_to_none(self.province_edit.text()),
                canton=self._blank_to_none(self.canton_edit.text()),
                distrito=self._blank_to_none(self.district_edit.text()),
                start_date=self._blank_to_none(self.start_date_edit.text()),
                end_date=self._blank_to_none(self.end_date_edit.text()),
                horizon_days=int(self.horizon_spin.value()),
            )
            area_level = self.area_level_combo.currentText()
            area_info = report.answers.get("most_probable_area_for_crime", {})
            area_name = area_info.get(area_level, "--")
            area_prob = float(area_info.get("Probabilidad", 0.0))
            self.answer_area.setText(f"Most probable area for this crime: {area_name} ({area_prob:.2%})")

            danger_info = report.answers.get("most_dangerous_date_area", {})
            danger_date = str(danger_info.get("Fecha", "--"))[:10]
            danger_area = danger_info.get(area_level, "--")
            danger_score = float(danger_info.get("RiesgoEsperado", 0.0))
            self.answer_danger.setText(
                f"Most dangerous date and area in selected horizon: {danger_date} | {danger_area} | expected risk {danger_score:.2f}"
            )

            forecast_total = float(report.answers.get("forecast_total_next_horizon", 0.0))
            self.answer_forecast.setText(f"Expected total incidents in horizon: {forecast_total:.1f}")

            forecast_fig = self.predictor.plot_factory.forecast_figure(report.forecast)
            heatmap_fig = self.predictor.plot_factory.hour_weekday_heatmap_figure(
                report.predictive_heatmap, title=f"Predictive Heatmap - {self.crime_combo.currentText()}"
            )
            area_fig = self.predictor.plot_factory.area_risk_figure(report.area_risk, area_level)
            self.forecast_panel.set_figure(forecast_fig)
            self.heatmap_panel.set_figure(heatmap_fig)
            self.area_panel.set_figure(area_fig)

            self.dangerous_table.setRowCount(0)
            for _, row in report.dangerous_dates_areas.iterrows():
                idx = self.dangerous_table.rowCount()
                self.dangerous_table.insertRow(idx)
                self.dangerous_table.setItem(idx, 0, QTableWidgetItem(str(row["Fecha"])[:10]))
                self.dangerous_table.setItem(idx, 1, QTableWidgetItem(str(row.get(area_level, "--"))))
                self.dangerous_table.setItem(idx, 2, QTableWidgetItem(f"{float(row['RiesgoEsperado']):.2f}"))
        except Exception as error:
            QMessageBox.critical(self, "Future Risk Error", str(error))


class Window:
    def __init__(self) -> None:
        self.prediccion = Prediccion()
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.app.setStyle("Fusion")
        self.app.setFont(QFont("Segoe UI", 10))
        self.app.setStyleSheet(APP_STYLESHEET)
        self.main = QMainWindow()
        self.main.setWindowTitle("ClotCor | Predictive Crime Assistant")
        self.main.resize(1360, 860)

        self.day_options = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"]
        self.day_to_index = {name: index for index, name in enumerate(self.day_options)}
        self.month_options = [
            "1 - Enero",
            "2 - Febrero",
            "3 - Marzo",
            "4 - Abril",
            "5 - Mayo",
            "6 - Junio",
            "7 - Julio",
            "8 - Agosto",
            "9 - Septiembre",
            "10 - Octubre",
            "11 - Noviembre",
            "12 - Diciembre",
        ]
        self.month_to_index = {name: int(name.split(" - ")[0]) for name in self.month_options}

        self.inputs: Dict[str, QComboBox] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        header = QLabel("ClotCor - Guided Crime Prediction for Non-Technical Users")
        header.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        root.addWidget(header)

        sub = QLabel(
            "Use Train first, then Predict. Confidence is probability, not certainty. Use Dashboard and Future Risk for context."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("background: #dbeafe; border: 1px solid #bfdbfe; padding: 8px;")
        root.addWidget(sub)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter, 1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 840])
        left_scroll.setMinimumWidth(460)
        right_scroll.setMinimumWidth(620)

        self._build_input_panel(left_layout)
        self._build_help_panel(left_layout)

        self._build_result_panel(right_layout)
        self._build_action_bar(root)

        self.main.setCentralWidget(central)

    def _field_options(self, field: str, raw_values):
        if field == "Hora":
            return [f"{hour:02d}" for hour in range(0, 24)]
        if field == "DiaSemana":
            return self.day_options
        if field == "Mes":
            return self.month_options
        return sorted({str(value).strip() for value in raw_values if str(value).strip()})

    def _build_input_panel(self, parent_layout: QVBoxLayout) -> None:
        box = QGroupBox("Incident Input (What happened?)")
        layout = QFormLayout(box)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        options = self.prediccion.get_selection_options()
        input_fields = self.prediccion.feature_order

        for field in input_fields:
            combo = QComboBox()
            combo.addItems(self._field_options(field, options.get(field, [])))
            combo.setToolTip(f"Select the value for {field}.")
            combo.setMinimumWidth(260)
            self.inputs[field] = combo
            layout.addRow(QLabel(field), combo)
        parent_layout.addWidget(box)

    def _build_help_panel(self, parent_layout: QVBoxLayout) -> None:
        help_box = QGroupBox("How to use this screen")
        layout = QVBoxLayout(help_box)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setMinimumHeight(220)
        text.setPlainText(
            "Step A - Train / Update:\n"
            "Builds the predictive model from historical records. Run this after updating data.\n\n"
            "Step B - Predict:\n"
            "Uses your selected incident profile and returns top probable crimes.\n\n"
            "Step C - Dashboard:\n"
            "Shows historical trends, model quality, and predictive visual context.\n\n"
            "Step D - Future Risk:\n"
            "Answers where and when risk is expected to be higher in the near future.\n\n"
            "Interpretation note:\n"
            "Confidence is a probability estimate. A lower confidence means uncertainty is higher."
        )
        layout.addWidget(text)
        parent_layout.addWidget(help_box)

    def _build_result_panel(self, parent_layout: QVBoxLayout) -> None:
        summary = QGroupBox("Prediction Output")
        grid = QGridLayout(summary)
        self.predicted_label = QLabel("--")
        self.confidence_label = QLabel("--")
        self.model_label = QLabel("--")
        for lbl in (self.predicted_label, self.confidence_label, self.model_label):
            lbl.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))

        grid.addWidget(QLabel("Predicted crime"), 0, 0)
        grid.addWidget(self.predicted_label, 0, 1)
        grid.addWidget(QLabel("Confidence"), 1, 0)
        grid.addWidget(self.confidence_label, 1, 1)
        grid.addWidget(QLabel("Model"), 2, 0)
        grid.addWidget(self.model_label, 2, 1)
        parent_layout.addWidget(summary)

        table_box = QGroupBox("Top probabilities")
        table_layout = QVBoxLayout(table_box)
        self.prob_table = QTableWidget(0, 2)
        self.prob_table.setHorizontalHeaderLabels(["Crime", "Probability"])
        self.prob_table.horizontalHeader().setStretchLastSection(True)
        self.prob_table.verticalHeader().setVisible(False)
        self.prob_table.setAlternatingRowColors(True)
        table_layout.addWidget(self.prob_table)
        parent_layout.addWidget(table_box)

        chart_box = QGroupBox("Probability chart")
        chart_layout = QVBoxLayout(chart_box)
        self.prob_chart = FigurePanel()
        chart_layout.addWidget(self.prob_chart)
        parent_layout.addWidget(chart_box)

        self.interpretation = QLabel(
            "Interpretation: the first row is the most likely crime class under the current inputs."
        )
        self.interpretation.setWordWrap(True)
        self.interpretation.setStyleSheet("background: #ecfeff; border: 1px solid #a5f3fc; padding: 8px;")
        parent_layout.addWidget(self.interpretation)
        parent_layout.addStretch(1)

    def _build_action_bar(self, root_layout: QVBoxLayout) -> None:
        actions = QFrame()
        layout = QHBoxLayout(actions)

        self.train_btn = QPushButton("Train / Update")
        self.predict_btn = QPushButton("Predict")
        self.dashboard_btn = QPushButton("Dashboard")
        self.future_btn = QPushButton("Future Risk")
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.main.close)

        self.train_btn.clicked.connect(self.train_model)
        self.predict_btn.clicked.connect(self.get_prediction)
        self.dashboard_btn.clicked.connect(self.show_dashboard)
        self.future_btn.clicked.connect(self.show_future_risk)

        for btn in (self.train_btn, self.predict_btn, self.dashboard_btn, self.future_btn, self.close_btn):
            layout.addWidget(btn)
        root_layout.addWidget(actions)

        self.status_label = QLabel("Status: model not loaded.")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(44)
        self.status_label.setStyleSheet("padding: 8px; background: #e2e8f0; border: 1px solid #cbd5e1;")
        root_layout.addWidget(self.status_label)

    def _collect_user_payload(self):
        payload = {}
        for field in self.prediccion.feature_order:
            combo = self.inputs[field]
            value = combo.currentText().strip()
            if not value:
                raise ValueError(f"Field '{field}' cannot be empty.")
            if field == "Hora":
                payload[field] = int(value)
            elif field == "DiaSemana":
                payload[field] = self.day_to_index[value]
            elif field == "Mes":
                payload[field] = self.month_to_index[value]
            else:
                payload[field] = value
        return payload

    def _set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")
        self.app.processEvents()

    def train_model(self) -> None:
        try:
            self._set_status("Training model, tuning hyperparameters, calibrating probabilities...")
            metrics = self.prediccion.run(force_retrain=True)
            audit = metrics.get("data_audit", {})
            test_metrics = metrics.get("test", {})
            dropped = ", ".join(audit.get("dropped_leaky_features", [])) or "none"
            self.model_label.setText(str(self.prediccion.best_model_name))
            self._set_status(
                f"Done | F1={test_metrics.get('f1_weighted', 0.0):.3f} | split={audit.get('split_strategy')} | "
                f"calibration={audit.get('calibration_status')} | leak-guard dropped={dropped}"
            )
        except Exception as error:
            QMessageBox.critical(self.main, "Training Error", str(error))
            self._set_status("Training failed.")

    def get_prediction(self) -> None:
        try:
            self._set_status("Generating prediction...")
            payload = self._collect_user_payload()
            prediction_payload = self.prediccion.predict_new_data(payload, top_n=6, verbose=False)

            self.predicted_label.setText(prediction_payload.predicted_label)
            self.confidence_label.setText(f"{prediction_payload.confidence:.2%}")
            self.model_label.setText(str(self.prediccion.best_model_name or "--"))

            self.prob_table.setRowCount(0)
            for row, entry in enumerate(prediction_payload.top_probabilities):
                self.prob_table.insertRow(row)
                self.prob_table.setItem(row, 0, QTableWidgetItem(entry["delito"]))
                self.prob_table.setItem(row, 1, QTableWidgetItem(f"{entry['probabilidad']:.2%}"))

            fig = self.prediccion.get_probability_figure(prediction_payload)
            self.prob_chart.set_figure(fig)

            top_prob = prediction_payload.top_probabilities[0]["probabilidad"] if prediction_payload.top_probabilities else 0.0
            self.interpretation.setText(
                f"Interpretation: '{prediction_payload.predicted_label}' is the top estimate ({top_prob:.2%}). "
                "Review the second and third options too if they are close."
            )
            self._set_status("Prediction complete.")
        except Exception as error:
            QMessageBox.critical(self.main, "Prediction Error", str(error))
            self._set_status("Prediction failed.")

    def show_dashboard(self) -> None:
        try:
            self._set_status("Building analytics dashboard...")
            dialog = DashboardDialog(self.prediccion, self.main)
            dialog.exec()
            self._set_status("Dashboard closed.")
        except Exception as error:
            QMessageBox.critical(self.main, "Dashboard Error", str(error))
            self._set_status("Dashboard failed.")

    def show_future_risk(self) -> None:
        try:
            self._set_status("Opening future risk assistant...")
            defaults = {
                "Provincia": self.inputs.get("Provincia").currentText() if "Provincia" in self.inputs else "",
            }
            dialog = FutureRiskDialog(self.prediccion, defaults, self.main)
            dialog.exec()
            self._set_status("Future risk assistant closed.")
        except Exception as error:
            QMessageBox.critical(self.main, "Future Risk Error", str(error))
            self._set_status("Future risk assistant failed.")

    def window_content(self) -> None:
        return None

    def option_menu(self) -> None:
        return None

    def run(self) -> None:
        self.main.show()
        self.app.exec()
