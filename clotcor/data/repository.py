import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from clotcor.config import DATASET_PATH, UNIQUE_VALUES_PATH


class CrimeDataRepository:
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        unique_values_path: Path = UNIQUE_VALUES_PATH,
    ) -> None:
        self.dataset_path = dataset_path
        self.unique_values_path = unique_values_path

    def load_dataset(self) -> pd.DataFrame:
        return pd.read_csv(self.dataset_path)

    def get_unique_values(
        self,
        columns: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
        if self.unique_values_path.exists():
            with open(self.unique_values_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            if columns is None:
                return payload
            return {column: payload.get(column, []) for column in columns}

        frame = self.load_dataset()
        target_columns = list(columns) if columns is not None else frame.columns.tolist()
        values: Dict[str, List[str]] = {}
        for column in target_columns:
            if column not in frame.columns:
                values[column] = []
                continue
            non_null = frame[column].dropna().astype(str).str.strip()
            unique_sorted = sorted(set(non_null.tolist()))
            values[column] = unique_sorted
        return values
