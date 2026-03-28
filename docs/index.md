# ClotCor User Guide

Clotcor is a program designed for data processing and the implementation of geographic crime prediction through statistical analysis. It uses historical criminal activity data to build different models that are capable of projecting and visualizing areas with higher or lower crime probability. Additionally, it can predict the probability of a crime based on a specific input data the user provides, such as geographical location and date/time.

ClotCor helps you:

1. Train a crime prediction model from historical records.
2. Predict the most likely crime class for a selected incident profile.
3. Explore future risk by area and date.
4. Understand results with visual charts (no coding required).

## What You Can Do With ClotCor

- **Estimate likely crime type** from selected inputs (area, hour, victim profile, etc.).
- **See confidence and alternatives** (top probabilities, not only one class).
- **Identify likely risky areas** for a selected crime.
- **Estimate dangerous date-area combinations** for the next days/weeks.
- **Inspect historical behavior** with trend and heatmap charts.


!!! warning "Important Safety Note"
    ClotCor provides **probability estimates**, not guaranteed outcomes.
    
    *   A high probability means "more likely than other options", not certainty.
    *   A low probability difference between top classes means uncertainty is high.

## Recommended Workflow

1. Open the app.
2. Click **Train / Update** after any data update.
3. Fill incident fields.
4. Click **Predict**.
5. Read:
   - Predicted class
   - Confidence
   - Top alternatives
6. Open **Dashboard** for model and historical context.
7. Open **Future Risk** to answer:
   - "Which area is most likely for this crime?"
   - "Which date + area are most dangerous in the next horizon?"

## Current Validation Snapshot

On the current dataset snapshot processed on **March 26, 2026**, with leakage guard and temporal split enabled:

- Weighted F1 (test): around `0.655`
- Confidence over 95%: near `0%` in test
- Leak guard removed: `SubDelito`
- Split strategy: temporal holdout `70/15/15`

These values can change whenever the dataset changes.
