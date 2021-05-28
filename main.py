import utils
import algorithms
import numpy as np
from sklearn.cluster import DBSCAN
import time
import hdbscan
import sklearn
from st_dbscan import ST_DBSCAN
import vdbscan


def run_main(algos, data_name, flag_calc_neig, flag_save, path):
    print("dataset chosen is ", data_name)
    if data_name == "abalone":  # 4,177 samples
        df, true_class = utils.load_preprocess_abalone()
        eps = 0.2
        minpts = 3

    elif data_name == "mushroom":  # 8,124 samples
        df, true_class = utils.load_preprocess_mushrooms()
        eps = 2.5
        minpts = 4

    elif data_name == "pendigit":  # 10,992 samples
        df, true_class = utils.load_preprocess_pendigit()
        eps = 40
        minpts = 4

    elif data_name == "letter":  # 20,000 samples
        df, true_class = utils.load_preprocess_letters()
        eps = 0.5
        minpts = 8

    elif data_name == "cadata":  # 20,000 samples
        df, true_class = utils.load_preprocess_catadata()
        eps = 200
        minpts = 8

    elif data_name == "sensorless":  # 58,509 samples
        df, true_class = utils.load_preprocess_sensorless()
        eps = 0.3
        minpts = 20

    elif data_name == "shuttle":  # 58,000 samples
        df, true_class = utils.load_preprocess_shuttle()
        eps = 0.03
        minpts = 20

    tau = eps  # for IDBSCAN tau equals epsilon

    clustring = DBSCAN(eps=eps, min_samples=minpts).fit(np.asarray(df))  # sklearn
    predictions_ref = clustring.labels_
    print("baseline sklearn DBSCAN evaluation: ", )
    utils.perform_evaluation(true_class, predictions_ref, True)
    for i in range(len(algos)):
        algo = algos[i]
        start = time.time()

        if algo == "IDBSCAN":
            predictions = algorithms.main_IDBSCAN(df, eps, minpts, tau, flag_save, path, flag_calc_neig)
            print("For my IDBSCAN:")

        elif algo == "DBSCAN":
            predictions = algorithms.DBSCAN(np.asarray(df), eps, minpts)
            print("For my DBSCAN:")

        elif algo == "hdbscan":
            clusterer = hdbscan.HDBSCAN(min_samples=minpts, cluster_selection_epsilon=eps)
            predictions = clusterer.fit(df).labels_
            print("For HDBSCAN:")

        elif algo == "stdbscan":
            st_dbscan = ST_DBSCAN(eps1=eps, eps2=eps, min_samples=minpts)
            # st_dbscan.fit_frame_split(df, frame_size=100)
            st_dbscan.fit(df)
            predictions = st_dbscan.labels
            print("For ST-DBSCAN:")

        elif algo == "vdbscan":
            alg = vdbscan.VDBSCAN()
            alg.fit(df, eps_0=eps, verbose=True)
            predictions = alg.labels_
            print("For V-DBSCAN")

        elif algo == "leader":
            leader_dbscan = algorithms.DensityGeneral(np.asarray(df), eps, minpts, tau, flag_save, path)
            leader_dbscan.leader()
            leader_dbscan.S_data = leader_dbscan.data[leader_dbscan.L]
            leader_dbscan.DBSCAN()
            labels = [0] * len(df)  # place holder
            predictions = leader_dbscan.passing_predictions(labels)
            print("For my Leader:")

        end = time.time()
        time_elapsed = end - start
        print("runtime: " + str(time_elapsed))
        utils.perform_evaluation(predictions_ref, predictions, True)


algos = ["IDBSCAN", "DBSCAN", "stdbscan", "hdbscan", "vdbscan", "leader"]
datasets = ["abalone", "mushroom", "pendigit", "letter", "cadata", "sensorless", "shuttle"]
# algos = ["IDBSCAN", "DBSCAN", "leader"]
data_name = datasets[0]  # possible datasets:
flag_calc_neig = 1
flag_save = 1  # to save the labels outputs
path = "Results/results_mushrooms"  # where to save in case flag_save==1. Make sure the path exists.
run_main(algos, data_name, flag_calc_neig, flag_save, path)
