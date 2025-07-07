# genetic_feature_selection.py

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from deap import base, creator, tools, algorithms
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_classifier(name: str, random_state: int = 42):
    """
    Returns a sklearn classifier instance based on the classifier name.

    Parameters:
    - name (str): 'RandomForest', 'SVM', or 'LogisticRegression'
    - random_state (int): Random seed for reproducibility

    Returns:
    - classifier instance
    """
    if name == "RandomForest":
        return RandomForestClassifier(random_state=random_state)
    elif name == "SVM":
        return SVC(probability=True, random_state=random_state)
    elif name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier: {name}")


def load_and_prepare_data(df: pd.DataFrame, target_col: str):
    """
    Prepare features and encoded target from dataframe.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - target_col (str): Name of target column

    Returns:
    - X (pd.DataFrame): Features
    - y (np.ndarray): Encoded target labels
    - classes (np.ndarray): Original target classes
    """
    df_clean = df.drop(columns=[col for col in df.columns if "Unnamed" in col])
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le.classes_


def evaluate_individual(individual, X_scaled, y, classifier_name, scoring, cv_folds, random_state):
    """
    Fitness function for GA: mean cross-validated score of selected features.

    Parameters:
    - individual (list): Binary mask of selected features
    - X_scaled (np.ndarray): Scaled feature matrix
    - y (np.ndarray): Target array
    - classifier_name (str): Name of classifier to use
    - scoring (str): Scoring metric (e.g., 'accuracy')
    - cv_folds (int): Number of cross-validation folds
    - random_state (int): Random seed

    Returns:
    - tuple: Mean CV score (single-element tuple for DEAP)
    """
    if sum(individual) == 0:
        return 0.0,

    selected = [i for i, bit in enumerate(individual) if bit == 1]
    X_selected = X_scaled[:, selected]

    clf = get_classifier(classifier_name, random_state)
    score = cross_val_score(clf, X_selected, y, cv=cv_folds, scoring=scoring)
    return score.mean(),


def run_genetic_algorithm(X: pd.DataFrame, y: np.ndarray, config: dict):
    """
    Run the genetic algorithm for feature selection.

    Parameters:
    - X (pd.DataFrame): Features
    - y (np.ndarray): Target
    - config (dict): Configuration dictionary with keys:
        'classifier', 'scoring', 'n_gen', 'pop_size', 'cv_folds', 'random_state'

    Returns:
    - list: Best individual (selected features mask)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    n_features = X.shape[1]

    # Prevent DEAP errors if creator is already defined
    try:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    except AttributeError:
        pass

    try:
        creator.create("Individual", list, fitness=creator.FitnessMax)
    except AttributeError:
        pass

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: random.randint(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        lambda ind: evaluate_individual(
            ind, X_scaled, y,
            config['classifier'], config['scoring'],
            config['cv_folds'], config['random_state']
        )
    )
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=config['pop_size'])
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=config['n_gen'],
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    return hof[0]


def get_selected_features(individual, feature_names):
    """
    Return the list of feature names selected by the individual mask.

    Parameters:
    - individual (list): Binary list indicating selected features
    - feature_names (list or pd.Index): Feature names

    Returns:
    - list: Selected feature names
    """
    return [feature_names[i] for i, bit in enumerate(individual) if bit == 1]


def run_feature_selection_from_df(df: pd.DataFrame, target_col: str,
                                  classifier: str = 'RandomForest',
                                  scoring: str = 'accuracy',
                                  n_gen: int = 30,
                                  pop_size: int = 40,
                                  cv_folds: int = 5,
                                  random_state: int = 42):
    """
    Convenience function to run GA feature selection directly from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input dataframe
    - target_col (str): Target column name
    - classifier (str): Classifier name
    - scoring (str): Scoring metric
    - n_gen (int): Number of generations
    - pop_size (int): Population size
    - cv_folds (int): Number of cross-validation folds
    - random_state (int): Random seed

    Returns:
    - tuple: (selected_features (list), classes (np.ndarray))
    """
    X, y, classes = load_and_prepare_data(df, target_col)
    config = {
        'classifier': classifier,
        'scoring': scoring,
        'n_gen': n_gen,
        'pop_size': pop_size,
        'cv_folds': cv_folds,
        'random_state': random_state
    }
    best_individual = run_genetic_algorithm(X, y, config)
    selected_features = get_selected_features(best_individual, X.columns)
    logger.info(f"Selected features: {selected_features}")
    return selected_features, classes
