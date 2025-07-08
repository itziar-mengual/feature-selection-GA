import pandas as pd
import numpy as np
import random
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from deap import base, creator, tools, algorithms


def run_ga_feature_selection(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    classifier='logreg',
    n_gen=20,
    pop_size=30
):
    # Suprimir warnings de convergencia para visualización limpia
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Preprocesamiento
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[target_col].values)
    X = StandardScaler().fit_transform(X)
    n_features = X.shape[1]

    # Modelo según selección
    def get_model():
        if classifier.lower() == 'logreg':
            return LogisticRegression(max_iter=50000, solver='saga', penalty='l2')
        elif classifier.lower() == 'svm':
            return SVC(kernel='linear')
        elif classifier.lower() == 'rf':
            return RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported classifier: {classifier}")

    # Evaluación de individuos
    def evaluate_features(individual):
        if sum(individual) == 0:
            return (0.0,)
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        model = get_model()
        try:
            scores = cross_val_score(model, X[:, selected], y, cv=5, scoring='accuracy')
            return (scores.mean(),)
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            return (0.0,)

    # DEAP setup
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", lambda: random.randint(0, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_features)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen,
                        stats=stats, halloffame=hof, verbose=True)

    best_ind = hof[0]
    selected_features = [feature_cols[i] for i, bit in enumerate(best_ind) if bit == 1]

    return selected_features, best_ind.fitness.values[0]
