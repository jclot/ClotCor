# Methodology (How It Works)

This page explains the internal logic in simple language.

## 1) Data Preparation

Before training, ClotCor cleans and standardizes data:

- removes unnamed/garbage columns,
- parses dates,
- converts hour ranges to hour values (`0-23`),
- fills missing text values with `DESCONOCIDO`,
- creates time features:
  - weekday,
  - month,
  - weekend flag,
  - cyclic encodings for hour and month.

Why this matters:

- model can learn patterns consistently,
- fewer errors from irregular values,
- more stable behavior across updates.

## 2) Leakage Detection

Data leakage means a feature almost directly reveals the answer.

Example: if one input almost always maps to one class, the model can look "perfect" but fail in real situations.

ClotCor computes a leakage score for each categorical feature and automatically excludes very leaky ones (threshold `>= 0.95`, except area keys).

In your current dataset, `SubDelito` is detected as highly leaky and removed from training.

## 3) Train/Validation/Test Split

ClotCor first tries a **temporal split**:

- 70% oldest records for training,
- 15% next records for validation,
- 15% most recent records for test.

This is more realistic than purely random split because it simulates future prediction.

If temporal split cannot preserve all classes, ClotCor uses a stratified random fallback.

## 4) Controlled Tuning

ClotCor tests several model configurations (fast + robust candidates), compares them on validation data, and chooses the best by:

1. weighted F1 (main criterion),
2. accuracy,
3. log loss.

## 5) Probability Calibration

Raw model probabilities are often overconfident.

ClotCor tests calibration options (`none`, `sigmoid`, `isotonic`) on validation data and also tunes probability smoothing.

Target:

- preserve ranking quality,
- reduce unrealistic 95%+ confidence inflation.

## 6) Future Risk Engine

For future-oriented questions, ClotCor adds:

- daily incident forecasting (`PoissonRegressor`),
- area probability ranking for selected crime,
- predictive hour/day heatmap,
- dangerous date-area ranking (combined score from area probability and forecast).

This allows practical questions such as:

- "Where is this crime most likely?"
- "Which date and area are likely most dangerous in the next 30 days?"
