# Adaptive Stream Learning with Concept Drift Detection for Real-Time Financial Fraud Detection

Adaptive Stream Learning with Concept Drift Detection for Real-Time Financial Fraud Detection is a machine-learning project for real-time credit-card fraud detection under concept drift. It compares static fraud models with adaptive online models to measure how reliably they detect fraud as transaction behavior evolves.

## Project Title

**Adaptive Stream Learning with Concept Drift Detection for Real-Time Financial Fraud Detection**

## Problem

Digital payment systems must detect suspicious transactions quickly while minimizing false alerts for legitimate customers. Fraud patterns and customer behavior change over time, causing static models trained on historical data to degrade. SentinelFlow evaluates whether adaptive models and drift detection can recover performance more effectively than static models.

## Objectives

- Simulate a chronological stream of credit-card transactions.
- Compare static and adaptive fraud-detection models.
- Detect sudden and gradual concept drift.
- Adapt models after drift events and measure recovery.
- Report precision, recall, F1-score, PR-AUC, false-positive rate, detection delay, and recovery time.

## High-Level Architecture

```text
Public credit-card dataset
        |
        v
Data preparation and chronological batching
        |
        v
Transaction stream simulator
        |
        +-------------------------------+
        |                               |
        v                               v
Static models                    Adaptive online models
- Logistic Regression            - Online Logistic Regression
- Random Forest                  - Adaptive Random Forest
        |                               |
        +---------------+---------------+
                        |
                        v
Performance monitor and drift detectors
- ADWIN / DDM
- Rolling fraud metrics
                        |
                        v
Adaptation controller
- Recent-window retraining
- Model reset or replacement
                        |
                        v
Results, plots, and experiment report
```

## Approach

1. Prepare a public, anonymized credit-card transaction dataset and retain its chronological order.
2. Train static baseline models using an initial historical window.
3. Send later transactions through a batch-based stream simulator.
4. Run static and adaptive models side by side.
5. Feed model errors or losses into drift detectors.
6. When drift is detected, retrain or replace the affected adaptive model using recent observations.
7. Compare model performance before, during, and after drift events.

The initial dataset is suitable for severe class-imbalance experiments, but it spans only two days. SentinelFlow will therefore include controlled sudden- and gradual-drift scenarios and will clearly document this limitation.

## Planned Technology

- Python 3.11+
- pandas and NumPy for preparation and analysis
- scikit-learn for offline baseline models
- River for incremental learning, adaptive models, and drift detection
- Matplotlib and Seaborn for evaluation plots
- Jupyter Notebook for experiments
- GitHub Actions for continuous integration

## Planned Repository Layout

```text
sentinelflow-fraud-detection/
├── data/                 # Local or ignored datasets; never commit raw sensitive data
├── notebooks/            # Exploratory experiments
├── src/                  # Reusable pipeline code
├── tests/                # Automated tests
├── results/              # Metrics and generated figures
├── requirements.txt      # Python dependencies
└── README.md
```

## Dataset and Evaluation

The initial experiment will use the ULB/Kaggle Credit Card Fraud Detection dataset. It contains 284,807 anonymized transactions, including 492 fraud cases, and is highly imbalanced. [Dataset page](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Primary evaluation metrics:

- Precision and recall
- F1-score
- Precision-recall AUC
- False-positive rate
- Detection delay after a drift event
- Recovery time after adaptation

Accuracy is not a primary metric because fraud transactions are rare.

## Development Plan

1. Set up the data pipeline and a chronology-safe train/validation/test split.
2. Implement Logistic Regression and Random Forest baselines.
3. Build the stream simulator and rolling-metric tracker.
4. Implement online models and ADWIN/DDM drift detection.
5. Add controlled drift experiments and adaptation logic.
6. Produce result plots, final report material, and demonstration workflow.

## Continuous Integration

GitHub Actions runs on every push and pull request. The workflow verifies that this README exists, compiles Python source files, and runs tests when a `tests/` directory is present.

## Team

- Sanjay Solararamavu Dev -- Leader
- Hemalasya Annapureddy -- Deputy Leader
- Abhipsa Panda
- Aravindan Chidambaram
- Mihir Rajpathak
- Mitansh Maheshwari
- Rovianty Nugracia
- Thanyathorn Limsuvattanaphong

## Status

Project planning and reference study are in progress. Implementation has not started yet.
