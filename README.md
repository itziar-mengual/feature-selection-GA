# Genetic Feature Selection

This project implements a Genetic Algorithm (GA) for feature selection on classification datasets using Python. It supports multiple classifiers and allows easy customization of GA parameters.

---

## Features

- Feature selection with Genetic Algorithm
- Supports classifiers: RandomForest, SVM, LogisticRegression
- Configurable scoring metrics and cross-validation folds
- Easy-to-use Python function API
- Outputs selected features and class labels for further analysis

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/itziar-mengual/genetic-feature-selection.git
cd genetic-feature-selection
pip install -r requirements.txt
```

---

## Project structure
.
├── main.py
├── genetic-feature_selection.py       # Core GA feature selection logic
├── data/
│   └── wine.csv                  # Example dataset
└── README.md                    # This file


## Usage

Import and use the main function `run_feature_selection_from_df` from `genetic_feature_selection.py` in your Python scripts or interactive sessions.

Example:

```python
import pandas as pd
from genetic_feature_selection import run_feature_selection_from_df

df = pd.read_csv('data/wine.csv')

selected_features, classes = run_feature_selection_from_df(
    df,
    target_col='Type',
    classifier='SVM',
    n_gen=10
)

print("Selected features:", selected_features)
```

## Author
Itziar Mengual, 2025

