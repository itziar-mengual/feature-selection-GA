# Genetic Feature Selection and Optimization

This project implements feature selection using Genetic Algorithms (GA) for classification tasks. It supports multiple classifiers and hyperparameter optimization methods for SVM and Random Forest, as well as Elastic Net-based feature selection.

---

## Features

- Feature selection with Genetic Algorithms
- Supported classifiers: Random Forest, SVM, Elastic Net
- Hyperparameter optimization using GA and Grid Search for RF and SVM
- Flexible configuration with CSV datasets and JSON for storing selected features
- Examples and experiments provided in a Jupyter Notebook (main.ipynb)

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/itziar-mengual/ga-feature-selection-and-optimization.git
cd ga-feature-selection-and-optimization
pip install -r requirements.txt

```

---

## Project structure

```python
.
├── data/
│   ├── wine.csv                     # Example dataset
│   └── selected_features.json       # Stored selected features results
├── feature_selection/
│   ├── elastic_net_feature_selection.py   # Elastic Net feature selection
│   └── ga_feature_selection.py              # Genetic Algorithm feature selection
├── optimization/
│   ├── ga_rf.py                  # GA optimization for Random Forest
│   ├── ga_svm.py                 # GA optimization for SVM
│   ├── grid_search_rf.py         # Grid Search for Random Forest
│   └── grid_search_svm.py        # Grid Search for SVM
├── main.ipynb                   # Notebook with usage examples and experiments
├── README.md                    # Project documentation
└── requirements.txt             # Project dependencies

```

---

## Author

Itziar Mengual, 2025
