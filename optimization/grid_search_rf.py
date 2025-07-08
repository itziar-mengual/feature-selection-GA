import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def run_grid_search_rf(df: pd.DataFrame, selected_features: list, target_col: str):
    X = df[selected_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    rf_grid = {
        'n_estimators': [10, 25, 50, 100],
        'max_depth': [5, 10, 20],
        'min_samples_split': [5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_search = GridSearchCV(rf, rf_grid, cv=5, scoring='accuracy', return_train_score=True)
    rf_search.fit(X_train, y_train)

    rf_best = rf_search.best_estimator_
    rf_acc = accuracy_score(y_test, rf_best.predict(X_test))

    train_scores = rf_search.cv_results_['mean_train_score']
    test_scores = rf_search.cv_results_['mean_test_score']

    # Opcional: plot train vs test scores
    plt.plot(train_scores, label='Train accuracy')
    plt.plot(test_scores, label='Validation accuracy')
    plt.xlabel('Configuración (índice)')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation accuracy Grid Search RF')
    plt.legend()
    plt.show()

    return {
        'best_params': rf_search.best_params_,
        'test_accuracy': rf_acc,
        'train_scores': train_scores,
        'validation_scores': test_scores
    }
