import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def evaluate_svm(individual, X, y):
    C = 10 ** individual[0]
    gamma = 10 ** individual[1]
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(C=C, gamma=gamma, kernel='rbf', random_state=RANDOM_SEED))
    ])
    cv_results = cross_validate(model, X, y, cv=3, scoring='accuracy', return_train_score=True, n_jobs=-1)
    train_score = np.mean(cv_results['train_score'])
    val_score = np.mean(cv_results['test_score'])
    return (val_score,), train_score


def check_bounds(individual, bounds):
    for i in range(len(individual)):
        low, up = bounds[i]
        individual[i] = np.clip(individual[i], low, up)


def mutate_and_clip(individual, mu, sigma, indpb, bounds):
    tools.mutGaussian(individual, mu, sigma, indpb)
    check_bounds(individual, bounds)
    return individual,


def generate_individual(bounds):
    return [random.uniform(low, up) for (low, up) in bounds]


def run_ga_svm(df: pd.DataFrame, selected_features: list, target_col: str, n_gen=10, pop_size=20):
    X = df[selected_features].values
    y = LabelEncoder().fit_transform(df[target_col].values)

    bounds = [(-3, 3), (-5, 0)]  # rango log10 para C y gamma

    toolbox = base.Toolbox()
    toolbox.register("attr_float", generate_individual, bounds=bounds)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    train_scores_gen = []
    val_scores_gen = []

    def eval_individual(ind):
        fitness, train_score = evaluate_svm(ind, X, y)
        eval_individual.train_scores_current_gen.append(train_score)
        eval_individual.val_scores_current_gen.append(fitness[0])
        return fitness

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", mutate_and_clip, mu=0, sigma=0.5, indpb=0.2, bounds=bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    for gen in range(n_gen):
        eval_individual.train_scores_current_gen = []
        eval_individual.val_scores_current_gen = []

        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.5, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, len(pop))
        hof.update(pop)

        train_scores_gen.append(np.mean(eval_individual.train_scores_current_gen))
        val_scores_gen.append(np.mean(eval_individual.val_scores_current_gen))

        record = stats.compile(pop)
        print(f"Gen {gen} - Train acc avg: {train_scores_gen[-1]:.4f}, Val acc avg: {val_scores_gen[-1]:.4f}, Max val: {record['max']:.4f}")

    best_ind = hof[0]
    best_score = hof[0].fitness.values[0]
    best_params = {'C': 10 ** best_ind[0], 'gamma': 10 ** best_ind[1]}

    # Graficar evolución train vs val
    plt.plot(train_scores_gen, label='Train accuracy')
    plt.plot(val_scores_gen, label='Validation accuracy')
    plt.xlabel('Generación')
    plt.ylabel('Accuracy')
    plt.title('Evolución Train vs Validation accuracy GA SVM')
    plt.legend()
    plt.show()

    return best_params, best_score
