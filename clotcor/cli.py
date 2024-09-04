from clotcor.base import Prediccion

def main():  # pragma: no cover
    file_path = "./data/Estadisticas.csv"
    pre1 = Prediccion(file_path)
    pre1.run()