import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def run_grid_search_svm(df: pd.DataFrame, selected_features: list, target_col: str, random_state=42):
    X = df[selected_features]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC())
    ])

    svm_grid = [
        {
            'svm__kernel': ['linear'],
            'svm__C': [0.01, 0.1, 1, 10, 100]
        },
        {
            'svm__kernel': ['rbf'],
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1, 1]
        }
    ]

    svm_search = GridSearchCV(pipe, svm_grid, cv=5, scoring='accuracy',
                              n_jobs=-1, return_train_score=True)
    svm_search.fit(X_train, y_train)

    svm_best = svm_search.best_estimator_
    y_pred = svm_best.predict(X_test)
    svm_acc = accuracy_score(y_test, y_pred)

    train_scores = svm_search.cv_results_['mean_train_score']
    val_scores = svm_search.cv_results_['mean_test_score']

    # Gráfica Train vs Validation accuracy
    plt.plot(train_scores, label='Train accuracy')
    plt.plot(val_scores, label='Validation accuracy')
    plt.xlabel('Configuración (índice)')
    plt.ylabel('Accuracy')
    plt.title('Train vs Validation accuracy Grid Search SVM')
    plt.legend()
    plt.show()

    return {
        'best_params': svm_search.best_params_,
        'test_accuracy': svm_acc,
        'train_scores': train_scores,
        'validation_scores': val_scores
    }
