from pathlib import Path

import pandas as pd

from clotcor.modeling.predictor import Prediccion


def _build_sample_dataset(path: Path) -> None:
    rows = []
    base_rows = [
        ("ASALTO", "ARMA BLANCA", "PERSONA", "PEATON [PERSONA]", "Mayor de edad", "HOMBRE"),
        ("HURTO", "DESCUIDO", "VEHICULO", "AUTOMOVIL [VEHICULO]", "Mayor de edad", "MUJER"),
        ("ROBO", "FORZADURA", "VIVIENDA", "NO APLICA [VIVIENDA]", "Mayor de edad", "HOMBRE"),
    ]
    for class_index, base in enumerate(base_rows):
        for row_index in range(26):
            rows.append(
                {
                    "Delito": base[0],
                    "SubDelito": base[1],
                    "Fecha": f"2024-0{(row_index % 6) + 1}-{(row_index % 27) + 1:02d}",
                    "Hora": f"{(row_index + class_index) % 24:02d}:00:00 - {(row_index + class_index) % 24:02d}:59:59",
                    "Victima": base[2],
                    "SubVictima": base[3],
                    "Edad": base[4],
                    "Sexo": base[5],
                    "Nacionalidad": "COSTA RICA",
                    "Provincia": "SAN JOSE",
                    "Canton": "SAN JOSE",
                    "Distrito": "CARMEN",
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_predictor_train_and_predict(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    model_path = tmp_path / "artifacts" / "model.joblib"
    _build_sample_dataset(dataset_path)

    predictor = Prediccion(dataset_path=dataset_path, model_artifact_path=model_path)
    metrics = predictor.run(force_retrain=True)

    payload = {
        "SubDelito": "ARMA BLANCA",
        "Victima": "PERSONA",
        "SubVictima": "PEATON [PERSONA]",
        "Edad": "Mayor de edad",
        "Sexo": "HOMBRE",
        "Nacionalidad": "COSTA RICA",
        "Provincia": "SAN JOSE",
        "Canton": "SAN JOSE",
        "Distrito": "CARMEN",
        "Hora": 14,
        "DiaSemana": 3,
        "Mes": 6,
    }

    prediction = predictor.predict_new_data(payload, top_n=3)

    assert prediction.predicted_label in {"ASALTO", "HURTO", "ROBO"}
    assert 0 <= prediction.confidence <= 1
    assert len(prediction.top_probabilities) == 3
    assert model_path.exists()
    assert "data_audit" in metrics
    assert "dropped_leaky_features" in metrics["data_audit"]
    assert "calibration_tuning" in metrics
    assert isinstance(metrics["calibration_tuning"], list)


def test_spatiotemporal_report(tmp_path):
    dataset_path = tmp_path / "dataset.csv"
    model_path = tmp_path / "artifacts" / "model.joblib"
    _build_sample_dataset(dataset_path)

    predictor = Prediccion(dataset_path=dataset_path, model_artifact_path=model_path)
    predictor.run(force_retrain=True)
    report = predictor.get_spatiotemporal_report(delito="ASALTO", area_level="Canton", horizon_days=14)

    assert "most_probable_area_for_crime" in report.answers
    assert "most_dangerous_date_area" in report.answers
    assert "forecast_total_next_horizon" in report.answers
    assert report.forecast.shape[0] == 14
