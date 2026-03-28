<div align="center">
  <img src="public/images/icons/Clotcor_Icon.png" alt="ClotCor Logo" width="200" />
  
  <h1>ClotCor</h1>

  [![Python](https://img.shields.io/badge/Python-3.x-blue)](#) 
  [![License](https://img.shields.io/badge/License-MIT-green)](#) 
  [![Domain](https://img.shields.io/badge/Domain-Predictive_Criminology-orange)](#)
</div>

ClotCor is a Python framework focused on predictive crime analytics, statistical exploration, and interactive visualization. It processes historical crime records and builds multi-model classifiers to estimate crime-type probabilities and detect temporal/geographic hotspots.

---

## Overview

The primary focus of ClotCor is to serve as a reliable foundation for executing predictive algorithms in the field of geographic criminology. It is engineered to ingest standardized historical datasets provided by the **OIJ** (Organismo de Investigación Judicial). 

By analyzing the spatial and temporal attributes of past registered criminal events, ClotCor generates projections that estimate the most likely crime category and the probability of each class. The architecture separates data handling, preprocessing, modeling, analytics, plotting, and UI responsibilities to improve maintainability and accuracy.

### Key Characteristics

* **Controlled Model Tuning:** Trains and compares candidate models under temporal validation, then selects the best configuration.
* **Probability Outputs:** Returns top crime probabilities for each prediction request.
* **Leakage Guard + Calibration:** Detects highly leaky features and calibrates probabilities to avoid unrealistic confidence inflation.
* **Robust Preprocessing:** Cleans nulls, handles unknown categories, and engineers temporal cyclic features.
* **Future Risk Engine:** Adds spatio-temporal forecasting, predictive heatmaps, and dangerous date-area ranking.
* **Geographic Crime Statistics:** Produces trend charts, heatmaps, and feature-importance views.
* **Modern Qt GUI:** Guided PySide6 interface designed for non-technical users.
* **Layered Architecture:** Separated modules for `data`, `modeling`, `analytics`, `visualization`, and `ui`.
* **Container-Ready:** Includes a `Containerfile` for seamless Docker/Podman containerization and reproducible deployment of the model.
* **Automated Tooling:** Integrates a comprehensive `Makefile` to streamline development tasks such as model testing, linting, and environment setup.
* **Documentation Built-In:** Pre-configured with MkDocs for maintaining project and model documentation.

---

## Project Structure

The repository follows a standard and highly organized Python package layout:

```text
ClotCor/
├── clotcor/
│   ├── analytics/       # Statistical summaries and descriptive metrics
│   ├── data/            # Data loading and preprocessing pipeline
│   ├── modeling/        # Model training, evaluation, and prediction engine
│   ├── ui/              # PySide6 (Qt) desktop interface
│   ├── visualization/   # Matplotlib/Seaborn chart factory
│   ├── base.py          # Backward-compatible exports
│   └── cli.py           # Application entry point
├── data/                # OIJ datasets and serialized trained models
├── docs/                # MkDocs documentation source files
├── tests/               # Unit and integration test suites
└── setup.py             # Package installation and distribution configuration
```

---

## Installation

It is recommended to use a virtual environment to avoid dependency conflicts on your host system.

### 1. Clone the Repository

```bash
git clone [https://github.com/jclot/ClotCor.git](https://github.com/jclot/ClotCor.git)
cd ClotCor
```

### 2. Set Up the Environment

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required packages, including the dependencies needed for the predictive modeling algorithms:

```bash
pip install -r requirements.txt
```

To install the `clotcor` package locally in editable mode for development:

```bash
pip install -e .
```

---

## Development and Automation

ClotCor utilizes a `Makefile` to automate common development workflows. Ensure `make` is installed on your system.

### Running Tests and Model Validation

To execute the test suite located in the `tests/` directory:

```bash
make test
```

### Containerization

If you prefer to run the predictive model within an isolated container environment, you can build the image using the provided `Containerfile`:

```bash
docker build -t clotcor-env -f Containerfile .
```

---

## Documentation

The project documentation is built using MkDocs. To serve the documentation locally and view it in your browser:

```bash
mkdocs serve
```

---

## License and Authorship

**Author:** Julián Clot Córdoba ([jclot](https://github.com/jclot))

This project is open-source software licensed under the **MIT License**. Please refer to the `LICENSE` file for full terms and conditions.
