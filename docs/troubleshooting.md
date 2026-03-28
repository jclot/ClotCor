# Troubleshooting

## 1) Confidence Was Always >95%

### Why it happened

Common causes:

- data leakage,
- random-only split,
- uncalibrated probabilities.

### What ClotCor now does

- leakage guard (auto-drop very leaky features),
- temporal holdout split,
- calibration + smoothing tuning.

Check in training metrics:

- `data_audit.dropped_leaky_features`
- `data_audit.split_strategy`
- `confidence_profile_test.pct_top_over_95`

## 2) Seaborn FutureWarning About `palette` and `hue`

This warning has been resolved by using explicit `hue` assignment in bar plots.

## 3) NumPy/SciPy Compatibility Warning

If you see:

`A NumPy version >=1.23.5 and <2.3.0 is required for this version of SciPy`

your environment has incompatible package versions.

Fix:

```bash
pip install -r requirements.txt --upgrade --force-reinstall
```

## 4) `No module named pytest` or `No module named mkdocs`

Install test/docs dependencies:

```bash
pip install -r requirements-test.txt
```

## 5) Qt Launch Problems

If GUI does not start:

1. ensure `PySide6` is installed,
2. run from activated environment,
3. verify command:

```bash
python -c "import PySide6; print('ok')"
```

## 6) `KeyboardInterrupt` in Terminal

If terminal shows `KeyboardInterrupt`, it usually means the process was manually interrupted (for example Ctrl+C).
This is not necessarily a model failure.
