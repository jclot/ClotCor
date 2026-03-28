# Usage (Step by Step)

## Start the Application

```bash
clotcor
```

## Main Buttons Explained

### Train / Update

Use this button when:

- you changed `data/Estadisticas.csv`,
- you want to refresh model behavior,
- you want updated metrics.

What happens internally:

1. data cleaning,
2. leakage audit,
3. temporal split,
4. model tuning,
5. calibration tuning,
6. model artifact save.

What you should read after training:

- selected model name,
- weighted F1,
- split strategy,
- calibration status,
- dropped leaky features.

### Predict

Use after selecting incident fields.

What it returns:

- top predicted class,
- confidence,
- top alternatives with probabilities,
- probability chart.

How to interpret:

- If top probability is much larger than second option, prediction is more stable.
- If top and second are close, uncertainty is high.

### Dashboard

Opens visual diagnostics:

1. **Trend**: monthly behavior over time.
2. **Hour/Day**: historical concentration by weekday/hour.
3. **Confusion**: where model mistakes are concentrated.
4. **Importance**: strongest variables for separation.
5. **Forecast**: expected daily cases in next horizon.
6. **Predictive Heatmap**: expected future concentration by weekday/hour.
7. **Area Risk**: top probable areas for selected behavior.

### Future Risk

Goal: answer operational questions quickly.

Filters:

- crime type,
- area level (province/canton/district),
- optional area/date filters,
- horizon in days.

Direct answers shown:

- most probable area,
- most dangerous date + area,
- expected total incidents in horizon.

## How to Fill Inputs Correctly

- Select each dropdown value intentionally.
- Keep area fields consistent (province/canton/district if possible).
- Choose hour/day/month that best represent the incident context.

If a value is unknown, use the closest valid option (often `DESCONOCIDO`).

## Practical Interpretation Checklist

Before acting on results, check:

1. Is confidence clearly above alternatives?
2. Are dashboard trends consistent with this prediction?
3. Do future risk outputs point to the same area pattern?
4. Is the selected horizon reasonable (example: 14, 30, 60 days)?
