import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from itertools import product
from scipy.stats import loguniform
from argparse import ArgumentParser
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skfeature.function.similarity_based import fisher_score
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


def apply_feature_selection(algorithm_name, X, Y, params, best_lr, best_rf):
    if algorithm_name == 'Variance Threshold':
        selected_features = VarianceThreshold(threshold=params['threshold']).fit_transform(X)
        selected_features_names = [X.columns.tolist()[i] for i in VarianceThreshold(threshold=params['threshold']).fit(X).get_support(indices=True)]
    elif algorithm_name == 'Fischer Score':
        ranks = fisher_score.fisher_score(X.to_numpy(), Y.to_numpy())
        top_indices = np.argsort(ranks)[::-1][:params['n_features']]
        selected_features = X.iloc[:, top_indices].to_numpy()
        selected_features_names = X.columns[top_indices].tolist()
    elif algorithm_name == 'Information Gain':
        ig = mutual_info_classif(X, Y, n_neighbors=params['n_neighbors'])
        selected_features_names = pd.Series(ig, X.columns).sort_values(ascending=False)[0: params['n_features']].index.tolist()
        selected_features = X[selected_features_names].to_numpy()
    elif algorithm_name in ['Forward Feature Selection', 'Backward Feature Selection', 'Recursive Feature Elimination']:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        if algorithm_name in ['Forward Feature Selection', 'Backward Feature Selection']:
            if algorithm_name == 'Forward Feature Selection':
                sfs = SequentialFeatureSelector(best_lr, k_features=params['n_features'], forward=True, floating=False, scoring="accuracy", cv=5)
            elif algorithm_name == 'Backward Feature Selection':
                sfs = SequentialFeatureSelector(best_lr, k_features=params['n_features'], floating=False, scoring="accuracy", cv=5)

            sfs.fit(X_train, y_train)
            selected_features_names = list(sfs.k_feature_names_)
            selected_features = X[selected_features_names].to_numpy()
        else:
            rfe = RFE(best_lr, n_features_to_select=params['n_features'])
            rfe.fit(X_train, y_train)
            selected_features_names = [X.columns.tolist()[i] for i in rfe.get_support(indices=True)]
            selected_features = X[selected_features_names].to_numpy()
    elif algorithm_name == 'LASSO':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        lasso = Lasso()
        # TODO: Consider 'balanced_accuracy', 'f1', and other scoring measures
        grid_search = GridSearchCV(lasso, {'alpha': np.arange(0.1, 10, 0.1), 'tol': loguniform(1e-6, 1e-1).rvs(5)}, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
    elif algorithm_name == 'Random Forest':
        selected_features_names = pd.Series(best_rf.feature_importances_, X.columns).sort_values(ascending=False)[0: params['n_features']].index.tolist()
        selected_features = X[selected_features_names].to_numpy()

    return selected_features, selected_features_names

def generate_combinations(params_dict):
    keys = params_dict.keys()
    values = params_dict.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

def select_features(X, Y):
    best_lr = tune_lr(X, Y)
    best_rf = tune_rf(X, Y)
    print("[INFO]: Done Tuning Logistic Regression & Random Forest for Feature Selection Algorithms")
    selected_features_dict = {}
    for feature_selection_algorithm in feature_selection_algorithms:
        selected_features_l = []
        for feature_selection_params in generate_combinations(feature_selection_algorithm['params']):
            selected_features, selected_features_names = apply_feature_selection(feature_selection_algorithm['name'],
                                                                                 X,
                                                                                 Y,
                                                                                 feature_selection_params,
                                                                                 best_lr,
                                                                                 best_rf)
            selected_features_l.append({'params': feature_selection_params,
                                        'selected_features_names': selected_features_names})
            selected_features_dict[feature_selection_algorithm['name']] = selected_features_l
        print("Done Selecting Features for Feature Selection Algorithm %s" % feature_selection_algorithm['name'])
    return selected_features_dict

def compute_neighborhood_preservation(X, X_transformed, n_neighbors=10):
    # Fit NearestNeighbors on the Original and Embedding Space
    # TODO: Check other Distances (Manhattan, Euclidean, ..etc)
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    nbrs_transformed = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_transformed)

    # Get the Indicies of the K-Nearest Neighbors
    original_neighbors = nbrs_original.kneighbors(X, return_distance=False)[:, 1:]
    transformed_neighbors = nbrs_transformed.kneighbors(X_transformed, return_distance=False)[:, 1:]

    # Calculate the Neighborhood Preservation Score
    preservation_scores = []
    for i in range(X.shape[0]):
        shared_neighbors = len(set(original_neighbors[i]).intersection(set(transformed_neighbors[i])))
        preservation_score = shared_neighbors / n_neighbors
        preservation_scores.append(preservation_score)

    # TODO: Visualize the Preservation Score; Since we are Averaging, Some Nodes Might Ruin it for others!
    mean_score = np.mean(preservation_scores)
    median_score = np.median(preservation_scores)

    # Trimmed Mean (Remove 10% Lowest and Highest Scores)
    trimmed_mean_score = np.mean(np.sort(preservation_scores)[int(0.1*len(preservation_scores)):int(0.9*len(preservation_scores))])
    return (mean_score, trimmed_mean_score, median_score)

def remove_outliers(scores, threshold=1.5):
    z_scores = np.abs(stats.zscore(scores))
    return scores[z_scores < threshold]

def compute_neighborhood_preservation_outliers(X, X_transformed, n_neighbors=10):
    # Fit NearestNeighbors on the Original and Embedding Space
    # TODO: Check other Distances (Manhattan, Euclidean, ..etc)
    nbrs_original = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    nbrs_transformed = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_transformed)

    # Get the Indicies of the K-Nearest Neighbors
    original_neighbors = nbrs_original.kneighbors(X, return_distance=False)[:, 1:]
    transformed_neighbors = nbrs_transformed.kneighbors(X_transformed, return_distance=False)[:, 1:]

    # Calculate the Neighborhood Preservation Score
    preservation_scores = []
    for i in range(X.shape[0]):
        shared_neighbors = len(set(original_neighbors[i]).intersection(set(transformed_neighbors[i])))
        preservation_score = shared_neighbors / n_neighbors
        preservation_scores.append(preservation_score)

    preservation_scores = np.array(preservation_scores)
    filtered_preservation_scores = remove_outliers(preservation_scores)

    return np.mean(filtered_preservation_scores)

def compute_distance_preservation(X, X_transformed, metric='euclidean'):
    # TODO: Experiment with other Distances (Manhattan, Minkowski, ..etc).
    original_distances = pairwise_distances(X, metric=metric).flatten()
    transformed_distances = pairwise_distances(X_transformed, metric=metric).flatten()

    corr_coef = np.corrcoef(original_distances, transformed_distances)[0, 1]
    return corr_coef

def evaluate_classification_performance(X_transformed, y): # TODO: Need to Refactor this Code to Reduce the Runtime
    # Split the Embedding Space into Training/Testing
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    # Set up the Parameter Grid for Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [None, 30],
        'min_samples_split': [2, 7],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Set up the RandomForestClassifier and GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    # Calculate Performance Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    # precision = precision_score(y_test, y_pred, average='weighted')
    # recall = recall_score(y_test, y_pred, average='weighted')
    precision = 0
    recall = 0
    return accuracy, f1, precision, recall, grid_search.best_params_

def tune_lr(X, Y):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Hyperparameters Tuning for LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lr = LogisticRegression()
    param_grid = {
        'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
        'C' : [100, 10, 1.0, 0.1, 0.01],
        'solver' : ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
        'max_iter' : [1000, 2500, 5000]
    }

    grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_lr = grid_search.best_estimator_

    return best_lr

def tune_rf(X, Y):
    # Hyperparameters Tuning for RandomForest
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # Set up the Parameter Grid for Hyperparameter Tuning
    param_grid = {
        'n_estimators': [100, 200, 1000],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    # Set up the RandomForestClassifier and GridSearchCV
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    return best_rf

def get_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name, index_col=0)
    # Proteomic Dataset
    X = dataset.loc[:, dataset.columns.str.startswith('X')]
    # Standardizing the Dataset
    # scaler = StandardScaler()
    # X_standardized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_standardized = X
    # Phenotype
    Y = dataset['GOLD_STAGE_COPD_SEVERITY']
    return X_standardized, Y

LLE_parameters_dict = {
    'standard': {
        'n_neighbors': [5, 10, 15, 20, 50, 100, 300],
        'n_components': [2, 3, 5, 8, 10],
        'reg': loguniform(1e-4, 1e-1).rvs(4),
        'max_iter': [300],
    },
    'standard-arpack': {
        'n_neighbors': [5, 10, 15, 20, 50, 100, 300],
        'n_components': [2, 3, 5, 8, 10],
        'reg': loguniform(1e-4, 1e-1).rvs(4),
        'eigen_solver': ['arpack'],
        'tol': loguniform(1e-6, 1e-1).rvs(4),
        'max_iter': [300],
    },
    'standard-dense': {
        'n_neighbors': [5, 10, 15, 20, 50, 100, 300],
        'n_components': [2, 3, 5, 8, 10],
        'reg': loguniform(1e-4, 1e-1).rvs(5),
        'eigen_solver': ['dense'],
    },
    'hessian': {
        'n_neighbors': [5, 10, 15, 20, 50, 100, 300],
        'n_components': [2, 3, 5, 8, 10],
        'method': ['hessian'],
        'reg': loguniform(1e-4, 1e-1).rvs(4),
        'eigen_solver': ['dense'],
        'hessian_tol': loguniform(1e-6, 1e-1).rvs(4),
        'max_iter': [300],
    },
    'modified': {
        'n_neighbors': [5, 10, 15, 20, 50, 100, 300],
        'n_components': [2, 3, 5, 8, 10],
        'method': ['modified'],
        'reg': loguniform(1e-4, 1e-1).rvs(4),
        'modified_tol': loguniform(1e-12, 1e-3).rvs(4),
        'max_iter': [300],
    }
}

feature_selection_algorithms = [
    {'name': 'Variance Threshold', 'params': {'threshold': [0.9]}},
    {'name': 'Information Gain', 'params': {'n_features': [10, 15, 20, 25], 'n_neighbors': [5, 10, 50]}},
    {'name': 'Fischer Score', 'params': {'n_features': [10, 15, 20, 25]}},
    {'name': 'Forward Feature Selection', 'params': {'n_features': [10, 15, 20, 25]}},
    {'name': 'Backward Feature Selection', 'params': {'n_features': [10, 15, 20, 25]}},
    {'name': 'Recursive Feature Elimination', 'params': {'n_features': [10, 15, 20, 25]}},
    {'name': 'Random Forest', 'params': {'n_features': [10, 15, 20, 25]}}
]

feature_selection_algorithms = [
    {'name': 'Variance Threshold', 'params': {'threshold': [0.9]}},
    {'name': 'Information Gain', 'params': {'n_features': [10, 16, 20, 25], 'n_neighbors': [5, 10, 50]}},
    {'name': 'Fischer Score', 'params': {'n_features': [10, 16, 20, 25]}},
    {'name': 'Forward Feature Selection', 'params': {'n_features': [10, 16, 20, 25]}},
    {'name': 'Backward Feature Selection', 'params': {'n_features': [10, 16, 20, 25]}},
    {'name': 'Recursive Feature Elimination', 'params': {'n_features': [10, 16, 20, 25]}},
    {'name': 'Random Forest', 'params': {'n_features': [10, 16, 20, 25]}}
]

feature_selection_algorithms = [
    {'name': 'Variance Threshold', 'params': {'threshold': [0.9]}},
    {'name': 'Information Gain', 'params': {'n_features': [10, 20, 25, 30, 50], 'n_neighbors': [5, 10, 50]}},
    {'name': 'Fischer Score', 'params': {'n_features': [10, 20, 25, 30, 50]}},
    {'name': 'Forward Feature Selection', 'params': {'n_features': [10, 20, 25, 30, 50]}},
    {'name': 'Backward Feature Selection', 'params': {'n_features': [10, 20, 25, 30, 50]}},
    {'name': 'Recursive Feature Elimination', 'params': {'n_features': [10, 20, 25, 30, 50]}},
    {'name': 'Random Forest', 'params': {'n_features': [10, 20, 25, 30, 50]}}
]


def make_args():
    parser = ArgumentParser()
    parser.add_argument('--method', dest='method', default='standard', type=str,
                        help='LLE Method Name. Available Options: [standard, standard-arpack, standard-dense, hessian, modified]')
    parser.add_argument('--dataset', dest='dataset', type=str, help='Dataset Name')
    args = parser.parse_args()
    return args

def main():
    args = make_args()
    X, Y = get_dataset(args.dataset)
    results = []
    invalid_parameters_combinations = []
    LLE_parameters_combinations = generate_combinations(LLE_parameters_dict[args.method])
    selected_features_dict = select_features(X, Y)
    print("[INFO]: Done Selecting Features!")
    iter = 0
    for LLE_parameters_combination in LLE_parameters_combinations:
        print("Iteration %s" % iter)
        iter += 1
        print(LLE_parameters_combination)
        for feature_selection_algorithm, feature_selection_algorithm_params in selected_features_dict.items():
            for feature_selection_param in feature_selection_algorithm_params:
                selected_features_names = feature_selection_param['selected_features_names']
                selected_features = X[selected_features_names]

                # Check if the Number of Selected Features Equals the Number of LLE Components
                if len(selected_features_names) == LLE_parameters_combination['n_components']:
                    print("[WARNING]: Number of Selected Features Matches the Number of LLE Components (No Dimensionality Reduction)")
                    continue
                try:
                    # Run LLE
                    lle = LocallyLinearEmbedding(**LLE_parameters_combination)
                    embedding = lle.fit_transform(selected_features)
                    reconstruction_error = lle.reconstruction_error_ # TODO: Add a method to evaluate the generated manifold
                    neighborhood_preservation_score_mean, neighborhood_preservation_score_trimmed_mean, neighborhood_preservation_score_median = compute_neighborhood_preservation(selected_features, embedding)
                    neighborhood_preservation_score_outliers = compute_neighborhood_preservation_outliers(selected_features, embedding)
                    pairwise_distances_preservation_score_euclidean = compute_distance_preservation(selected_features, embedding)
                    pairwise_distances_preservation_score_manhattan = compute_distance_preservation(selected_features, embedding, 'manhattan')
                    pairwise_distances_preservation_score_minkowski = compute_distance_preservation(selected_features, embedding, 'minkowski')
                    prediction_performance = evaluate_classification_performance(embedding, Y)
                    prediction_accuracy = prediction_performance[0]
                    prediction_f1 = prediction_performance[1]
                    prediction_best_params = prediction_performance[-1]
                    # print("Prediction Metrics:\nAccuracy: %s\tF1: %s\tPrecision: %s\tRecall: %s\nBestParams: %s" % evaluate_classification_performance(embedding, Y))
                    results.append([LLE_parameters_combination, feature_selection_algorithm, feature_selection_param, selected_features_names, reconstruction_error, neighborhood_preservation_score_mean, neighborhood_preservation_score_trimmed_mean, neighborhood_preservation_score_median, neighborhood_preservation_score_outliers, pairwise_distances_preservation_score_euclidean, pairwise_distances_preservation_score_manhattan, pairwise_distances_preservation_score_minkowski, prediction_accuracy, prediction_f1, prediction_best_params])
                except Exception as E:
                    print(E)
                    invalid_parameters_combinations.append(LLE_parameters_combination)

            # print("[INFO]: Done with Feature Selection %s" % feature_selection_algorithm)

    results_df = pd.DataFrame(results, columns=['LLE Params', 'Feature Selection Algorithm', 'Feature Selection Params', 'Selected Features', 'Reconstruction Error', 'Neighborhood Preservation Score Mean', 'Neighborhood Preservation Score Trimmed Mean', 'Neighborhood Preservation Score Median', 'Neighborhood Preservation Score Outliers', 'Pairwise Distance Preservation Score Euclidean', 'Pairwise Distance Preservation Score Manhattan', 'Pairwise Distance Preservation Score Minkowski','Prediction Accuracy', 'Prediction F1', 'Prediction Bsest Params'])
    results_df.to_csv('LLE_Results[%s].csv' % args.method, index=False)
    
if __name__ == "__main__":
    main()
