from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, TypedDict, cast

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from clotcor.analytics import StatisticalAnalyzer
from clotcor.config import DATASET_PATH, MODEL_ARTIFACT_PATH, RANDOM_STATE
from clotcor.data import CrimeDataPreprocessor, CrimeDataRepository
from clotcor.modeling.risk_engine import RiskReport, SpatioTemporalRiskEngine
from clotcor.visualization import CrimePlotFactory


class ProbabilityEntry(TypedDict):
    delito: str
    probabilidad: float


class SupportsPredictProba(Protocol):
    def predict_proba(self, x_data: pd.DataFrame) -> np.ndarray: ...


@dataclass
class PredictionPayload:
    predicted_label: str
    confidence: float
    top_probabilities: List[ProbabilityEntry]


class Prediccion:
    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
        model_artifact_path: Path = MODEL_ARTIFACT_PATH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.repository = CrimeDataRepository(dataset_path=dataset_path)
        self.preprocessor = CrimeDataPreprocessor()
        self.plot_factory = CrimePlotFactory()
        self.model_artifact_path = model_artifact_path
        self.random_state = random_state

        self.model: Optional[object] = None
        self.base_model: Optional[Pipeline] = None
        self.best_model_name: Optional[str] = None
        self.metrics: Dict[str, object] = {}
        self.class_report: Dict[str, object] = {}
        self.confusion_matrix_normalized: Optional[np.ndarray] = None
        self.feature_importance: pd.DataFrame = pd.DataFrame(columns=["feature", "importance"])
        self.analysis_frame: Optional[pd.DataFrame] = None

        self.selection_sample_size = 20000
        self.leakage_threshold = 0.95
        self.split_strategy = "not_trained"
        self.calibration_status = "not_trained"
        self.probability_smoothing = 0.06

        self.active_categorical_features = list(self.preprocessor.feature_spec.categorical)
        self.active_numeric_features = list(self.preprocessor.feature_spec.numeric)
        self.leakage_scores: Dict[str, float] = {}
        self.dropped_leaky_features: List[str] = []
        self.calibration_tuning_table: List[Dict[str, object]] = []

        self.class_prior_labels: List[str] = []
        self.class_prior: Optional[np.ndarray] = None

    @property
    def feature_order(self) -> List[str]:
        return self.preprocessor.feature_spec.user_input

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def _dataset_is_newer_than_model(self) -> bool:
        if not self.model_artifact_path.exists():
            return True
        data_mtime = self.repository.dataset_path.stat().st_mtime
        model_mtime = self.model_artifact_path.stat().st_mtime
        return data_mtime > model_mtime

    @staticmethod
    def _build_one_hot_encoder() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", self._build_one_hot_encoder()),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("cat", categorical_pipeline, self.active_categorical_features),
                ("num", numeric_pipeline, self.active_numeric_features),
            ],
            remainder="drop",
        )

    def _build_pipeline(self, estimator: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("prep", self._build_preprocessor()),
                ("clf", estimator),
            ]
        )

    def _candidate_estimators(self) -> Dict[str, object]:
        return {
            "extra_trees_lite": ExtraTreesClassifier(
                n_estimators=70,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=22,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=self.random_state,
            ),
            "sgd_logistic_v1": SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=0.0002,
                l1_ratio=0.10,
                max_iter=3000,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "sgd_logistic_v2": SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                alpha=0.0005,
                l1_ratio=0.20,
                max_iter=2800,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "sgd_logistic_v3": SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=0.0003,
                max_iter=2800,
                tol=1e-4,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight="balanced",
                random_state=self.random_state,
            ),
        }

    @staticmethod
    def _class_distribution(y_data: pd.Series) -> Dict[str, float]:
        dist = y_data.value_counts(normalize=True)
        return {str(label): float(prob) for label, prob in dist.items()}

    def _compute_leakage_scores(self, clean_frame: pd.DataFrame) -> Dict[str, float]:
        target = self.preprocessor.feature_spec.target
        scores: Dict[str, float] = {}
        for feature in self.preprocessor.feature_spec.categorical:
            if feature not in clean_frame.columns:
                scores[feature] = 0.0
                continue
            table = (
                clean_frame.groupby([feature, target], observed=True)
                .size()
                .rename("count")
                .reset_index()
            )
            if table.empty:
                scores[feature] = 0.0
                continue
            totals = table.groupby(feature, observed=True)["count"].sum().rename("total")
            merged = table.merge(totals, on=feature, how="left")
            merged["p"] = merged["count"] / merged["total"]
            max_per_group = merged.groupby(feature, observed=True)["p"].max()
            weighted = (max_per_group * totals / totals.sum()).sum()
            scores[feature] = float(weighted)
        return scores

    def _activate_leakage_guard(self, clean_frame: pd.DataFrame) -> None:
        self.leakage_scores = self._compute_leakage_scores(clean_frame)
        guard_candidates = [
            feature
            for feature in self.preprocessor.feature_spec.categorical
            if feature not in {"Provincia", "Canton", "Distrito"}
        ]
        self.dropped_leaky_features = [
            feature
            for feature in guard_candidates
            if self.leakage_scores.get(feature, 0.0) >= self.leakage_threshold
        ]
        active = [feature for feature in self.preprocessor.feature_spec.categorical if feature not in self.dropped_leaky_features]
        if not active:
            active = list(self.preprocessor.feature_spec.categorical)
            self.dropped_leaky_features = []
        self.active_categorical_features = active
        self.active_numeric_features = list(self.preprocessor.feature_spec.numeric)

    def _prepare_splits(
        self, clean_frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        target = self.preprocessor.feature_spec.target
        features = self.preprocessor.feature_spec.model_input
        all_classes = set(clean_frame[target].unique())

        temporal = clean_frame.dropna(subset=["Fecha"]).sort_values("Fecha").reset_index(drop=True)
        if len(temporal) >= 4000:
            train_end = int(len(temporal) * 0.70)
            val_end = int(len(temporal) * 0.85)
            if train_end > 0 and val_end > train_end and val_end < len(temporal):
                train_frame = temporal.iloc[:train_end]
                val_frame = temporal.iloc[train_end:val_end]
                test_frame = temporal.iloc[val_end:]
                if (
                    set(train_frame[target].unique()) == all_classes
                    and set(val_frame[target].unique()) == all_classes
                    and set(test_frame[target].unique()) == all_classes
                ):
                    self.split_strategy = "temporal_holdout_70_15_15"
                    return (
                        train_frame[features],
                        train_frame[target],
                        val_frame[features],
                        val_frame[target],
                        test_frame[features],
                        test_frame[target],
                    )

        x_data = clean_frame[features]
        y_data = clean_frame[target]
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_data,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y_train_val,
        )
        self.split_strategy = "random_stratified_60_20_20_fallback"
        return x_train, y_train, x_val, y_val, x_test, y_test

    def _subsample_for_tuning(self, x_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        if len(x_train) <= self.selection_sample_size:
            return x_train, y_train
        try:
            x_sub, _, y_sub, _ = train_test_split(
                x_train,
                y_train,
                train_size=self.selection_sample_size,
                random_state=self.random_state,
                stratify=y_train,
            )
            return x_sub, y_sub
        except ValueError:
            return x_train, y_train

    @staticmethod
    def _evaluate_candidate(model: Pipeline, x_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, float]:
        y_pred = model.predict(x_val)
        y_prob = model.predict_proba(x_val)
        return {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            "log_loss": float(log_loss(y_val, y_prob, labels=model.classes_)),
        }

    @staticmethod
    def _confidence_profile(probabilities: np.ndarray) -> Dict[str, float]:
        max_prob = probabilities.max(axis=1)
        return {
            "mean_top_probability": float(max_prob.mean()),
            "p95_top_probability": float(np.quantile(max_prob, 0.95)),
            "pct_top_over_95": float((max_prob >= 0.95).mean()),
            "pct_top_over_90": float((max_prob >= 0.90).mean()),
            "pct_top_over_80": float((max_prob >= 0.80).mean()),
        }

    @staticmethod
    def _get_classes(model: object) -> np.ndarray:
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "estimator"):
            classes = getattr(model.estimator, "classes_", None)
        if classes is None and hasattr(model, "base_estimator"):
            classes = getattr(model.base_estimator, "classes_", None)
        if classes is None:
            return np.array([])
        return np.asarray(classes)

    def _make_prefit_calibrator(self, fitted_model: Pipeline, method: str, x_cal: pd.DataFrame, y_cal: pd.Series) -> Optional[object]:
        try:
            calibrator = CalibratedClassifierCV(estimator=fitted_model, method=method, cv="prefit")
        except TypeError:
            calibrator = CalibratedClassifierCV(base_estimator=fitted_model, method=method, cv="prefit")
        try:
            calibrator.fit(x_cal, y_cal)
            return calibrator
        except Exception:
            return None

    def _apply_smoothing(self, probabilities: np.ndarray, alpha: Optional[float] = None) -> np.ndarray:
        if self.class_prior is None or len(self.class_prior) != probabilities.shape[1]:
            return probabilities
        value = self.probability_smoothing if alpha is None else alpha
        value = float(np.clip(value, 0.0, 0.4))
        priors = np.asarray(self.class_prior, dtype=float).reshape(1, -1)
        smoothed = ((1.0 - value) * probabilities) + (value * priors)
        row_sum = smoothed.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return smoothed / row_sum

    @staticmethod
    def _predict_proba(model: object, x_data: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            raise RuntimeError("Current model does not expose predict_proba.")
        predictor = cast(SupportsPredictProba, model)
        return np.asarray(predictor.predict_proba(x_data))

    def _tune_calibration_and_smoothing(
        self, base_model: Pipeline, x_val: pd.DataFrame, y_val: pd.Series
    ) -> Tuple[str, float, List[Dict[str, object]]]:
        options: List[Tuple[str, object]] = [("none", base_model)]
        for method in ("sigmoid", "isotonic"):
            calibrated = self._make_prefit_calibrator(base_model, method, x_val, y_val)
            if calibrated is not None:
                options.append((method, calibrated))

        smoothing_candidates = [0.00, 0.03, 0.06, 0.10]
        table: List[Dict[str, object]] = []
        best_tuple: Optional[Tuple[float, float, float, str, float]] = None
        best_choice = ("none", self.probability_smoothing)

        for method_name, model in options:
            classes = self._get_classes(model)
            if classes.size == 0:
                continue
            raw_prob = self._predict_proba(model, x_val)
            for alpha in smoothing_candidates:
                prob = self._apply_smoothing(raw_prob, alpha=alpha)
                pred = classes[np.argmax(prob, axis=1)]
                f1w = float(f1_score(y_val, pred, average="weighted"))
                ll = float(log_loss(y_val, prob, labels=classes))
                conf = self._confidence_profile(prob)
                record = {
                    "calibration": method_name,
                    "smoothing_alpha": float(alpha),
                    "f1_weighted": f1w,
                    "log_loss": ll,
                    "pct_top_over_95": conf["pct_top_over_95"],
                }
                table.append(record)
                score_tuple = (f1w, -ll, -conf["pct_top_over_95"], method_name, float(alpha))
                if best_tuple is None or score_tuple > best_tuple:
                    best_tuple = score_tuple
                    best_choice = (method_name, float(alpha))
        return best_choice[0], best_choice[1], table

    def _extract_feature_importance(self, trained_model: Pipeline) -> pd.DataFrame:
        prep = trained_model.named_steps["prep"]
        clf = trained_model.named_steps["clf"]

        if hasattr(prep, "get_feature_names_out"):
            feature_names = prep.get_feature_names_out()
        else:
            feature_names = np.array(
                [f"feature_{index}" for index in range(len(self.preprocessor.feature_spec.model_input))]
            )

        if hasattr(clf, "feature_importances_"):
            scores = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            coefficients = np.asarray(clf.coef_)
            scores = np.mean(np.abs(coefficients), axis=0)
        else:
            scores = np.zeros(shape=(len(feature_names),), dtype=float)

        if len(scores) != len(feature_names):
            scores = np.resize(scores, len(feature_names))
        output = pd.DataFrame({"feature": feature_names, "importance": scores})
        return output.sort_values("importance", ascending=False).reset_index(drop=True)

    def _ensure_analysis_frame(self) -> pd.DataFrame:
        if self.analysis_frame is None:
            raw = self.repository.load_dataset()
            _, _, clean = self.preprocessor.prepare_training_data(raw)
            self.analysis_frame = clean
        return self.analysis_frame

    def _save_artifacts(self) -> None:
        self.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "base_model": self.base_model,
            "best_model_name": self.best_model_name,
            "metrics": self.metrics,
            "class_report": self.class_report,
            "confusion_matrix_normalized": self.confusion_matrix_normalized,
            "feature_importance": self.feature_importance,
            "active_categorical_features": self.active_categorical_features,
            "active_numeric_features": self.active_numeric_features,
            "leakage_scores": self.leakage_scores,
            "dropped_leaky_features": self.dropped_leaky_features,
            "split_strategy": self.split_strategy,
            "calibration_status": self.calibration_status,
            "probability_smoothing": self.probability_smoothing,
            "calibration_tuning_table": self.calibration_tuning_table,
            "class_prior_labels": self.class_prior_labels,
            "class_prior": self.class_prior,
        }
        joblib.dump(payload, self.model_artifact_path)

    def _load_artifacts(self) -> None:
        payload = joblib.load(self.model_artifact_path)
        self.model = payload.get("model")
        self.base_model = payload.get("base_model")
        self.best_model_name = payload.get("best_model_name")
        self.metrics = payload.get("metrics", {})
        self.class_report = payload.get("class_report", {})
        self.confusion_matrix_normalized = payload.get("confusion_matrix_normalized")
        self.feature_importance = payload.get("feature_importance", pd.DataFrame(columns=["feature", "importance"]))
        self.active_categorical_features = payload.get(
            "active_categorical_features", list(self.preprocessor.feature_spec.categorical)
        )
        self.active_numeric_features = payload.get("active_numeric_features", list(self.preprocessor.feature_spec.numeric))
        self.leakage_scores = payload.get("leakage_scores", {})
        self.dropped_leaky_features = payload.get("dropped_leaky_features", [])
        self.split_strategy = payload.get("split_strategy", "unknown")
        self.calibration_status = payload.get("calibration_status", "unknown")
        self.probability_smoothing = float(payload.get("probability_smoothing", 0.06))
        self.calibration_tuning_table = payload.get("calibration_tuning_table", [])
        self.class_prior_labels = payload.get("class_prior_labels", [])
        class_prior_payload = payload.get("class_prior")
        self.class_prior = None if class_prior_payload is None else np.asarray(class_prior_payload, dtype=float)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        raw_frame = self.repository.load_dataset()
        x_data, y_data, _ = self.preprocessor.prepare_training_data(raw_frame)
        return train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_data,
        )

    def train_model(self) -> Tuple[pd.DataFrame, pd.Series]:
        raw_frame = self.repository.load_dataset()
        _, _, clean_frame = self.preprocessor.prepare_training_data(raw_frame)
        self.analysis_frame = clean_frame
        self._activate_leakage_guard(clean_frame)

        x_train, y_train, x_val, y_val, x_test, y_test = self._prepare_splits(clean_frame)
        x_tune, y_tune = self._subsample_for_tuning(x_train, y_train)

        candidate_estimators = self._candidate_estimators()
        candidate_scores: Dict[str, Dict[str, float]] = {}
        for model_name, estimator in candidate_estimators.items():
            pipeline = self._build_pipeline(estimator)
            pipeline.fit(x_tune, y_tune)
            candidate_scores[model_name] = self._evaluate_candidate(pipeline, x_val, y_val)

        self.best_model_name = max(
            candidate_scores,
            key=lambda name: (
                candidate_scores[name]["f1_weighted"],
                candidate_scores[name]["accuracy"],
                -candidate_scores[name]["log_loss"],
            ),
        )

        selected_estimator = candidate_estimators[self.best_model_name]
        base_train_model = self._build_pipeline(selected_estimator)
        base_train_model.fit(x_train, y_train)

        val_classes = self._get_classes(base_train_model)
        prior_series_train = y_train.value_counts(normalize=True)
        self.class_prior = prior_series_train.reindex(val_classes, fill_value=0.0).values
        self.class_prior_labels = [str(item) for item in val_classes.tolist()]

        best_cal_method, best_smoothing, calibration_grid = self._tune_calibration_and_smoothing(
            base_train_model, x_val, y_val
        )
        self.probability_smoothing = best_smoothing
        self.calibration_tuning_table = calibration_grid

        if best_cal_method == "none":
            final_model = base_train_model
            self.calibration_status = "none_prefit"
        else:
            calibrated = self._make_prefit_calibrator(base_train_model, best_cal_method, x_val, y_val)
            if calibrated is None:
                final_model = base_train_model
                self.calibration_status = f"fallback_no_calibration_after_{best_cal_method}_error"
            else:
                final_model = calibrated
                self.calibration_status = f"{best_cal_method}_prefit"

        final_classes = self._get_classes(final_model)
        prior_series = y_train.value_counts(normalize=True)
        self.class_prior = prior_series.reindex(final_classes, fill_value=0.0).values
        self.class_prior_labels = [str(item) for item in final_classes.tolist()]

        self.base_model = base_train_model
        self.model = final_model

        y_prob_raw = final_model.predict_proba(x_test)
        y_prob = self._apply_smoothing(y_prob_raw, alpha=self.probability_smoothing)
        y_pred = final_classes[np.argmax(y_prob, axis=1)]

        self.metrics = {
            "validation_candidates": candidate_scores,
            "calibration_tuning": calibration_grid,
            "test": {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
                "log_loss": float(log_loss(y_test, y_prob, labels=final_classes)),
            },
            "data_audit": {
                "class_distribution": self._class_distribution(clean_frame[self.preprocessor.feature_spec.target]),
                "leakage_scores": self.leakage_scores,
                "dropped_leaky_features": self.dropped_leaky_features,
                "split_strategy": self.split_strategy,
                "calibration_status": self.calibration_status,
                "probability_smoothing": self.probability_smoothing,
            },
            "confidence_profile_test": self._confidence_profile(y_prob),
        }
        self.class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        self.confusion_matrix_normalized = confusion_matrix(
            y_test,
            y_pred,
            labels=final_classes,
            normalize="true",
        )
        self.feature_importance = self._extract_feature_importance(self.base_model)
        self._save_artifacts()
        return x_test, y_test

    def load_or_train(self, force_retrain: bool = False) -> None:
        if force_retrain or self._dataset_is_newer_than_model():
            self.train_model()
            return
        self._load_artifacts()

    def run(self, force_retrain: bool = False) -> Dict[str, object]:
        self.load_or_train(force_retrain=force_retrain)
        return self.metrics

    def evaluate_model(
        self,
        x_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> Dict[str, object]:
        if not self.is_trained:
            self.load_or_train(force_retrain=False)
        if x_test is None or y_test is None:
            return {
                "best_model": self.best_model_name,
                "metrics": self.metrics,
                "classification_report": self.class_report,
            }
        y_pred, y_prob = self.predict_crime(x_test)
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        report["log_loss"] = float(log_loss(y_test, y_prob, labels=self._get_classes(self.model)))
        return {"classification_report": report}

    def plot_confusion_matrix(
        self,
        y_test: Optional[pd.Series] = None,
        y_pred: Optional[pd.Series] = None,
    ):
        if not self.is_trained:
            self.load_or_train(force_retrain=False)

        if y_test is not None and y_pred is not None:
            labels = sorted(pd.unique(pd.concat([pd.Series(y_test), pd.Series(y_pred)])))
            matrix = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
            return self.plot_factory.confusion_matrix_figure(matrix, labels)

        labels = [str(item) for item in self._get_classes(self.model).tolist()]
        matrix = self.confusion_matrix_normalized
        if matrix is None:
            matrix = np.zeros((len(labels), len(labels)), dtype=float)
        return self.plot_factory.confusion_matrix_figure(matrix, labels)

    def plot_feature_importance(self):
        if not self.is_trained:
            self.load_or_train(force_retrain=False)
        return self.plot_factory.feature_importance_figure(self.feature_importance)

    def predict_crime(self, input_data: Dict[str, object] | pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            self.load_or_train(force_retrain=False)
        prepared = self.preprocessor.prepare_inference_data(input_data)
        model = self.model
        if model is None:
            raise RuntimeError("Model is not available for inference.")
        probabilities = self._predict_proba(model, prepared)
        probabilities = self._apply_smoothing(probabilities, alpha=self.probability_smoothing)
        classes = self._get_classes(model)
        prediction = classes[np.argmax(probabilities, axis=1)]
        return prediction, probabilities

    def predict_new_data(
        self,
        new_data: Dict[str, object] | pd.DataFrame,
        top_n: int = 5,
        verbose: bool = False,
    ) -> PredictionPayload:
        prediction, probabilities = self.predict_crime(new_data)
        labels = self._get_classes(self.model).tolist()
        top_indices = np.argsort(probabilities[0])[::-1][:top_n]
        top_probabilities: List[ProbabilityEntry] = []
        for index in top_indices:
            top_probabilities.append(
                {
                    "delito": str(labels[index]),
                    "probabilidad": float(probabilities[0][index]),
                }
            )
        payload = PredictionPayload(
            predicted_label=str(prediction[0]),
            confidence=float(np.max(probabilities[0])),
            top_probabilities=top_probabilities,
        )

        if verbose:
            print(f"Predicted crime: {payload.predicted_label}")
            print("Top probabilities:")
            for entry in payload.top_probabilities:
                print(f"- {entry['delito']}: {entry['probabilidad']:.2%}")

        return payload

    def get_selection_options(self) -> Dict[str, List[str]]:
        categorical = self.preprocessor.feature_spec.categorical
        values = self.repository.get_unique_values(columns=categorical)
        for column in categorical:
            cleaned = [
                html.unescape(str(item)).strip()
                for item in values.get(column, [])
                if str(item).strip()
            ]
            cleaned = sorted(set(cleaned))
            if not cleaned:
                cleaned = ["DESCONOCIDO"]
            if "DESCONOCIDO" not in cleaned:
                cleaned.insert(0, "DESCONOCIDO")
            values[column] = cleaned
        values["Hora"] = [str(number) for number in range(0, 24)]
        values["DiaSemana"] = [str(number) for number in range(0, 7)]
        values["Mes"] = [str(number) for number in range(1, 13)]
        return values

    def get_available_crimes(self) -> List[str]:
        frame = self._ensure_analysis_frame()
        return sorted(frame["Delito"].dropna().unique().tolist())

    def get_statistical_summary(self) -> Dict[str, pd.DataFrame]:
        frame = self._ensure_analysis_frame()
        analyzer = StatisticalAnalyzer(frame)
        return analyzer.summary()

    def get_probability_figure(self, payload: PredictionPayload):
        labels = [entry["delito"] for entry in payload.top_probabilities]
        scores = [entry["probabilidad"] for entry in payload.top_probabilities]
        return self.plot_factory.probabilities_figure(labels, scores)

    def get_spatiotemporal_report(
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
        frame = self._ensure_analysis_frame()
        engine = SpatioTemporalRiskEngine(frame)
        return engine.build_report(
            delito=delito,
            area_level=area_level,
            provincia=provincia,
            canton=canton,
            distrito=distrito,
            start_date=start_date,
            end_date=end_date,
            horizon_days=horizon_days,
        )

    def get_dashboard_figures(self) -> Dict[str, object]:
        if not self.is_trained:
            self.load_or_train(force_retrain=False)

        stats = self.get_statistical_summary()
        mode_crime = self._ensure_analysis_frame()["Delito"].mode()
        default_crime = str(mode_crime.iloc[0]) if not mode_crime.empty else "HURTO"
        risk_report = self.get_spatiotemporal_report(delito=default_crime, area_level="Canton", horizon_days=30)
        area_col = "Canton" if "Canton" in risk_report.area_risk.columns else risk_report.area_risk.columns[0]
        figures = {
            "trend": self.plot_factory.trend_figure(stats["temporal_trend"]),
            "hour_weekday_heatmap": self.plot_factory.hour_weekday_heatmap_figure(
                stats["hour_weekday_heatmap"], title="Historical Hour/Weekday Concentration"
            ),
            "confusion": self.plot_confusion_matrix(),
            "feature_importance": self.plot_feature_importance(),
            "forecast": self.plot_factory.forecast_figure(risk_report.forecast),
            "predictive_heatmap": self.plot_factory.hour_weekday_heatmap_figure(
                risk_report.predictive_heatmap, title=f"Predictive Hour/Weekday Heatmap ({default_crime})"
            ),
            "area_risk": self.plot_factory.area_risk_figure(risk_report.area_risk, area_col),
        }
        return figures
