import pandas as pd

from clotcor.data.preprocessing import CrimeDataPreprocessor


def test_preprocessor_creates_temporal_features():
    preprocessor = CrimeDataPreprocessor()
    frame = pd.DataFrame(
        [
            {
                "Delito": "ASALTO",
                "SubDelito": "ARMA BLANCA",
                "Fecha": "2024-01-15",
                "Hora": "12:00:00 - 14:59:59",
                "Victima": "PERSONA",
                "SubVictima": "PEATON [PERSONA]",
                "Edad": "Mayor de edad",
                "Sexo": "MUJER",
                "Nacionalidad": "COSTA RICA",
                "Provincia": "SAN JOSE",
                "Canton": "SAN JOSE",
                "Distrito": None,
            }
        ]
    )

    x_data, y_data, clean = preprocessor.prepare_training_data(frame)

    assert x_data.loc[0, "Hora"] == 12
    assert x_data.loc[0, "DiaSemana"] == 0
    assert x_data.loc[0, "Mes"] == 1
    assert "HoraSin" in x_data.columns
    assert "HoraCos" in x_data.columns
    assert "MesSin" in x_data.columns
    assert "MesCos" in x_data.columns
    assert y_data.iloc[0] == "ASALTO"
    assert clean.loc[0, "Distrito"] == "DESCONOCIDO"
