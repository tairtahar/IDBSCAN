from sklearn.metrics import confusion_matrix, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score
import numpy as np


def ARI(ground_truth, prediction):
    return adjusted_rand_score(ground_truth, prediction)


def calinsky_harabasz_index(data: np.ndarray, prediction):
    return calinski_harabasz_score(data, prediction)


def davies_boulding_index(data: np.ndarray, prediction):
    return davies_bouldin_score(data, prediction)


def silhouette_coefficient(data: np.ndarray, prediction):
    return silhouette_score(data, prediction, metric='euclidean')


def confusion_matrix_ground_truth(ground_truth, prediction):
    return confusion_matrix(y_true=ground_truth, y_pred=prediction)
