<div align="center">
  <img src="public/images/icons/Clotcor_Icon.png" alt="ClotCor Logo" width="150" />
  
  <h1>ClotCor</h1>

  [![Python](https://img.shields.io/badge/Python-3.x-blue)](#) 
  [![License](https://img.shields.io/badge/License-MIT-green)](#) 
  [![Domain](https://img.shields.io/badge/Domain-Predictive_Criminology-orange)](#)
</div>

ClotCor is a dedicated Python framework designed for data processing and the implementation of predictive geographic crime statistical analysis. It utilizes historical registered criminal activity data to build models capable of projecting and visualizing areas with higher or lower delictual probability (crime hotspots).

---

## Overview

The primary focus of ClotCor is to serve as a reliable foundation for executing predictive algorithms in the field of geographic criminology. It is engineered to ingested standardized historical datasets provided by the **OIJ** (Organismo de Investigación Judicial). 

By analyzing the spatial and temporal attributes of past registered criminal events, ClotCor generates statistical projections that identify high-risk versus low-risk geographic areas. By standardizing the project structure, it ensures that sensitive data handling, core modeling logic, and validation tests are strictly separated, adhering to software engineering best practices for Python computational applications.

### Key Characteristics

* **Predictive Hotspotting:** Specialized modules designed to integrate and execute algorithms that predict delictual trends across geographic zones.
* **Geographic Crime Statistics:** Processing and visualization of criminal statistics based on spatial data.
* **Data Segregation:** Dedicated directories for dataset management to keep OIJ historical data logic isolated from the core algorithm.
* **Container-Ready:** Includes a `Containerfile` for seamless Docker/Podman containerization and reproducible deployment of the model.
* **Automated Tooling:** Integrates a comprehensive `Makefile` to streamline development tasks such as model testing, linting, and environment setup.
* **Documentation Built-In:** Pre-configured with MkDocs for maintaining project and model documentation.

---

## Project Structure

The repository follows a standard and highly organized Python package layout:

```text
ClotCor/
├── clotcor/             # Main application package and core predictive logic
├── data/                # Directory for OIJ input datasets and generated outputs
├── docs/                # MkDocs documentation source files
├── tests/               # Unit, integration, and model validation test suites
├── tools/               # Auxiliary scripts and development tools
├── Containerfile        # Container image definition for isolated model execution
├── HISTORY.md           # Changelog and version history
├── Makefile             # Automation script for development workflows
├── mkdocs.yml           # Configuration for the MkDocs documentation generator
├── requirements.txt     # Python package dependencies
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
