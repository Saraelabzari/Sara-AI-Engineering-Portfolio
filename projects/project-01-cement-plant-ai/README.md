# AI Cement Plant Intelligence System

An applied industrial AI project exploring machine-learning-based forecasting and process monitoring for cement plant sensor data.

## Project overview

This project analyzes historical Distributed Control System (DCS) data from cement plant operations and explores prediction of process variables such as calciner outlet temperature and pressure. The notebook uses Python data tools and scikit-learn to prepare sensor readings, train regression models, and evaluate performance against a baseline.

## Objectives

- Inspect and prepare timestamped cement-plant sensor data.
- Explore feature relationships across process variables.
- Train and evaluate regression models for process prediction.
- Compare model performance with a simple persistence baseline.
- Lay groundwork for anomaly detection and decision-support analytics.

## Technologies

Python · Jupyter Notebook · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn

## Dataset

The notebook references the Cement-DCS-data dataset:
https://github.com/mayerzhang/Cement-DCS-data

Review the dataset documentation and licensing before reuse. Do not commit confidential plant data, credentials, or proprietary operating records.

## Repository contents

The project notebook is intended to live in this folder as `AI_Cement_Plant_Intelligence_System.ipynb`. It should be run in Jupyter or Google Colab after the referenced dataset and dependencies are available.

## Run

1. Open the notebook in Google Colab or Jupyter.
2. Run cells in order.
3. Confirm the source dataset is reachable and that all required packages install successfully.
4. Review the reported evaluation metrics and plots.

## Limitations and safety

This is an experimental portfolio prototype for offline analysis. Metrics depend on data quality, time alignment, train/test methodology, target definition, and forecast horizon. The notebook is not an autonomous controller and does not write to a plant DCS/PLC or change operating setpoints. Any future industrial control use would require plant-engineer oversight, rigorous simulation and validation, cybersecurity review, fail-safe design, and formal safety approval.

## Author

**Sara Elabzari** · AI Engineering Portfolio
